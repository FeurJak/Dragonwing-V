# Dragonwing-V

A vision inference engine written from scratch in Rust for the Qualcomm Dragonwing platform. ONNX in, detections out, with a hand-written Vulkan compute backend and a CPU reference backend that exists to keep it honest.

No ONNX Runtime, no TFLite, no intermediate array framework. The only external dependency in the GPU path is [`ash`](https://github.com/ash-rs/ash), a thin `libvulkan` binding.

## Why this exists

The target is an Arduino UNO Q: a Qualcomm QRB2210 with four Cortex-A53-class cores at 2.0 GHz, an Adreno A702 at 845 MHz, and about 1.8 GiB of RAM. On paper there are several ways to get at that GPU. On the actual stock image there is one.

- **KGSL is absent.** `/dev/kgsl-3d0` does not exist. The board runs a mainline 6.16 kernel with the upstream `msm` DRM driver, so anything that talks to Adreno through the downstream Qualcomm ioctl interface — including the TinyGrad-style direct-submission approach — is closed without rebuilding the kernel.
- **The Hexagon DSP is not exposed.** No `/dev/adsprpc-smd`, no `/dev/cdsprpc-smd`, no fastrpc nodes.
- **OpenCL exists but is thin.** Rusticl enumerates the device with a single compute unit.
- **Vulkan works out of the box.** Mesa 25.2.6 Turnip, `VK_API_VERSION_1_0.318`, conformance 1.2.7.1.

So the GPU path is Vulkan compute through Turnip, and the CPU path is NEON on A53. `docs/backend-decision.md` records all four options and why two of them are parked rather than rejected.

The CPU backend is not a fallback that nobody runs. Every GPU kernel has a CPU counterpart, and the two are compared against each other. A four-core A53 cluster will not run YOLO at a useful frame rate, but it will tell you when the GPU is wrong.

## Status

Honest summary, because the interesting parts and the incomplete parts are next to each other.

**Working and measured on device:**

- 27 SPIR-V compute shaders covering F32, FP16 and packed-INT8 paths
- Host/device parity on the ops the harness covers, 9/9 on hardware
- SiLU fusion is bit-exact against the unfused path (`max|unfused − fused| = 0.0`)
- Slab allocation reduces a 103 MB YOLOv8m load from 110 `vkAllocateMemory` calls to exactly 1, at 99.5% slab utilisation
- Whole-graph command buffer batching: one submit per `run()` unless a CPU-fallback op forces a flush

**Not working yet:**

- **YOLOv8m F32 on Vulkan does not complete.** It dies at the first mid-network convolution. The naive conv shader is roughly 1 GFLOP per dispatch at `c_in=48, c_out=96`, which exceeds MSM's 500 ms GPU hangcheck; the recovery surfaces as a translation fault in `dmesg`. The arithmetic is correct, the kernel is too slow. A tiled conv shader is the fix and is not written.
- **YOLOv8n INT8 end to end is unmeasured.** The quantisation pipeline compiles, but there is no calibration-data loader, so the benchmark skips that pass.
- The quantised graph has no `Quantize` at the input or `Dequantize` at the output, so `set_input_f32` / `get_output_f32` return an error on an I8 graph. Use `set_input_bytes` / `get_output_bytes`.
- No memory planning beyond one buffer per tensor. No lifetime analysis, no aliasing.
- No scheduling. Ops run in ONNX topological order.

`155` tests across the workspace. The Vulkan integration tests skip silently when no Vulkan device is present, which means they pass without testing anything on a dev machine — worth knowing before trusting a green run.

## Architecture

Nine crates, 29k lines of Rust, 2.1k lines of GLSL.

| Crate | Role |
|---|---|
| `dragonwing-core` | `no_std`. Dtypes (F32/F16/I8/I32), quantisation primitives, the `Backend` trait, capability structs. One dependency: `libm`. |
| `dragonwing-hal` | Hardware probing. Reads procfs/sysfs, shells out to `vulkaninfo` and `clinfo`, produces a `HardwareCapabilities` report. |
| `dragonwing-probe` | Binary that dumps that report as JSON. |
| `dragonwing-cpu` | CPU backend. 57 ops, 14 with NEON paths, hand-rolled thread pool. |
| `dragonwing-shaders` | 27 GLSL compute shaders and the build script that compiles them to SPIR-V. Zero dependencies. |
| `dragonwing-vulkan` | Vulkan 1.1 compute backend. Context, pipeline cache, slab allocator, op dispatch, command-buffer recorder. |
| `dragonwing-onnx` | Largest crate. Hand-rolled protobuf reader, ONNX loader, graph compiler, fusion passes, quantisation, two runtimes, YOLO post-processing. |
| `dragonwing-test` | Parity harness, INT8 end-to-end tests, benchmarks. |
| `dragonwing-edge` | Façade crate. |

The `Backend` trait is deliberately tiny — allocate, upload, download, synchronise. No compute operations on it. Ops are free functions in each backend crate, so adding an operator does not change a trait that every backend has to implement.

## The Vulkan backend

**Shaders.** GLSL 450 compiled to SPIR-V with `glslangValidator`. If the toolchain is not on `PATH`, the build falls back to checked-in `.spv` blobs, so the project builds on a machine with no Vulkan SDK. Edit a `.comp` and you must recompile and commit the `.spv`.

**Device selection** scores candidates rather than taking the first: Turnip-on-Adreno `+100`, integrated `+10`, discrete `+5`, llvmpipe `+1`. Compute-only queue families are preferred over compute+graphics.

**Memory** is unified, so there are no staging buffers. Buffers are persistently mapped at construction and upload/download are `memcpy` through the mapped pointer. A slab allocator sits on top: 16 MiB slabs, 256-byte alignment, first-fit with coalescing on free. Drop order matters — buffers must release their handles before the slab releases the device memory.

**Pipelines** are cached in memory per `OpKind` and the `VkPipelineCache` is persisted to `$XDG_CACHE_HOME/dragonwing/`, keyed on device UUID, driver version and a hash of every shader blob, so editing any shader invalidates it by changing the filename.

**Dispatch** has two paths. The per-op path allocates a descriptor set and a transient command pool per call and blocks on drop; it is simple and slow, and it is what the unit tests use. The recorder path opens one command buffer for the whole graph, emits a global memory barrier only where the current op consumes the previous op's output, and submits once.

Two hardware-specific things worth calling out, because both were found the hard way:

**Descriptor pool exhaustion.** Op wrappers allocated descriptor sets and never freed them, so anything past about 64 dispatches returned `ERROR_OUT_OF_POOL_MEMORY`. Invisible to the unit tests — each creates a fresh pool and issues a handful of dispatches — and only visible under benchmark load.

**Workgroup count limits.** `maxComputeWorkGroupCount[0]` is 65535 on this device. YOLOv8m's first conv output is 4.9M elements, which at 64 threads per group is 76,800 workgroups, and the dispatch faults. Six element-wise shaders now take a `gx_total` push constant and unflatten a 2D grid themselves.

## INT8

`VK_KHR_8bit_storage` is not available on Turnip for this device, so INT8 cannot be declared as shader storage. Four `i8` values are packed little-endian into each `u32` and unpacked in GLSL with explicit sign extension. Forgetting the sign extension is the obvious bug and the docs say so.

The practical consequence is that element counts, channel counts and GEMM inner dimensions must be multiples of four. The runtime validates this at dispatch and returns an error rather than producing garbage.

Quantisation is symmetric throughout, zero-point always zero. Weights are per-channel on the output dimension, activations per-tensor, accumulation in INT32, then a requantise pass back to INT8. Calibration supports min/max and percentile strategies.

## FP16

`F16` is a `repr(transparent)` wrapper over `u16` with software IEEE-754 conversion, round-to-nearest-even, handling denormals and the special cases. There is no `half` dependency.

This is not gold-plating. The A53 cores in this SoC have FP16 *storage* but not FP16 *arithmetic* — no `fphp` or `asimdhp` in `/proc/cpuinfo` — so every CPU FP16 op widens to F32 for the maths anyway. Every FP16 op accumulates in F32 on both CPU and GPU, because a 10-bit mantissa overflows at K≈256 with realistic dense-layer magnitudes.

Whether Turnip actually executes the GPU-side maths in FP16 ALUs is unverified. The A/B benchmark to find out has not been run.

## Graph pipeline

```
load_model
  → fold_batchnorm          (BatchNorm folded into the preceding Conv at load; no runtime BN op)
  → compile_model           (validate all nodes, then build; failures aggregate rather than short-circuit)
  → convert_nchw_to_nhwc    (weights permuted [2,3,1,0])
  → fold_gemm_transpose
  → [QuantizedGraphCompiler, optional]
  → apply_fusion_passes
  → VulkanGraphRuntime::new
  → run
  → postprocess_yolo        (decode + NMS on CPU)
```

Validation and compilation are separate passes over the same graph. Validation allocates nothing and reports every unsupported node at once, so you find out what a model needs in one go rather than one error at a time.

28 ONNX op types have registered builders. Ops the GPU cannot handle — `Sub`, `Div`, `Concat`, `Resize`, `Split`, `Transpose`, `Slice` — fall back to the CPU by downloading, computing and re-uploading, which flushes the command recorder.

Three fusion passes: Conv+ReLU, Sigmoid+Mul into SiLU, and an INT8 Conv+Requantize+ReLU into a single shader.

## Correctness

The CPU backend is the reference. Tolerances are absolute, `1e-4` element-wise and `1e-3` for GEMM, with deterministic LCG-generated inputs and a fixed seed so failures reproduce.

Non-zero tolerance is expected rather than conceded: the GPU uses `fma()` with a single rounding, so the CPU scalar path uses `f32::mul_add` and the NEON path uses `vfmaq_f32` specifically to match it. Where the two still differ it is instruction scheduling and FP unit behaviour, not a bug.

The parity harness currently covers four ops — fill, axpy, relu, gemm. `docs/parity-testing.md` specifies a much wider matrix across FP16, INT8, pooling, softmax and end-to-end models; that is a specification of intent, not a description of what runs today.

## Building

Rust stable, edition 2024, `rust-version = 1.93`. Targets `aarch64-unknown-linux-gnu` and `aarch64-unknown-linux-musl`.

```sh
cargo build --workspace
cargo test  --workspace
```

Building the ONNX crate with a GPU path needs the feature enabled explicitly — default features are off:

```sh
cargo build -p dragonwing-onnx --features vulkan
```

Shader compilation needs `glslangValidator`. Without it the build uses the committed SPIR-V. Force that behaviour with `DRAGONWING_FORCE_PREBUILT_SPV=1`.

On the device you also need `vulkaninfo` and `clinfo` for `dragonwing-probe` to report the GPU sections.

The device runs glibc 2.41, which is newer than the `cross` project's default images provide, so cross-compiled binaries built that way fail to resolve symbols at load. `scripts/deploy.sh` handles build and push.

### Useful environment variables

| Variable | Effect |
|---|---|
| `DRAGONWING_TRACE=1` | Print each op and synchronise after it, so a failing op is pinpointed |
| `DRAGONWING_DISABLE_SLAB=1` | One `vkAllocateMemory` per buffer instead of slab allocation |
| `DRAGONWING_FORCE_PREBUILT_SPV=1` | Skip `glslangValidator`, use committed SPIR-V |

### Binaries

| Binary | Purpose |
|---|---|
| `dragonwing-probe` | Dump hardware capabilities as JSON |
| `vulkan-test` | Seven known-answer smoke tests, on device |
| `parity-test` | CPU vs GPU parity harness |
| `device-benchmark` | Op-level timings |
| `yolo-benchmark` | Full YOLO pipeline |
| `yolo-inspect` | Model structure dump |

## Target hardware

Measured on an Arduino UNO Q, captured 2026-05-27.

| | |
|---|---|
| SoC | Qualcomm QRB2210 (QCM2290 family) |
| CPU | 4× Kryo (A53 derivative) @ 2.016 GHz, NEON, no SVE, no FP16 arithmetic |
| GPU | Adreno A702 @ 845 MHz |
| Driver | Mesa 25.2.6 Turnip, Vulkan 1.0.318 |
| RAM | 1.78 GiB LPDDR4X |
| OS | Debian 13 (trixie), glibc 2.41, kernel 6.16.7 mainline |

Adreno A702 compute limits that shaped the design: 512 max workgroup invocations, 16 KiB shared memory, subgroup size 4, 128 MiB max storage buffer range, 65535 max workgroups in X.

## License

Apache-2.0.
