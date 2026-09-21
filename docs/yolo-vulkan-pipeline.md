# YOLOv8 on Vulkan — Full Pipeline Walkthrough (Task 008)

This document is the end-to-end recipe for running a YOLOv8 ONNX model
on the Vulkan backend (Adreno A702 via Mesa/Turnip on Arduino UNO Q).
It pulls together the building blocks from tasks 004–008.

## Quickstart

```bash
# On the Arduino UNO Q
cd /home/arduino/dragonwing-edge-rs
DRAGONWING_FORCE_PREBUILT_SPV=1 cargo build --release --bin yolo-benchmark

./target/release/yolo-benchmark path/to/yolov8.onnx
```

Useful environment variables (see "Diagnostics" below):

| Var | Effect |
|-----|--------|
| `WARMUP=N` | Run N warmup iterations before timing (default 3). |
| `ITERS=N` | Run N timed iterations (default 20). |
| `MAX_OPS=N` | Truncate the graph to its first N ops — useful for bisecting GPU faults. |
| `UNBATCHED=1` | Use one-submit-per-op execution; bypasses `OpsRecorder`. |
| `DRAGONWING_DISABLE_SLAB=1` | Allocate one `vkAllocateMemory` per tensor (legacy). |
| `DRAGONWING_SKIP_BIAS=1` | Skip Conv bias_add (isolation diagnostic). |
| `DRAGONWING_TRACE=1` | Print per-op shape / device-sync per dispatch. |

## Pipeline stages

```
                            ┌────────────────────────┐
   yolov8.onnx              │   dragonwing_onnx      │
       │                    │                        │
       ├─ load_model ──────►│  Model (proto-parsed)  │
       │                    └──────────┬─────────────┘
       │                               │
       │      Pass 1: fold_batchnorm ──┤   (mutates Model: removes BN nodes,
       │                               │    absorbs scale+bias into Conv weights)
       │                               ▼
       │                    ┌────────────────────────┐
       │      Pass 2:       │ compile_model(F32)     │
       │                    │  → Graph { ops,        │
       │                    │            shapes, ... }│
       │                    └──────────┬─────────────┘
       │                               │
       │      Pass 2b (auto): fold_qdq_patterns        [Task 009]
       │                               │   (skipped when no Q/DQ nodes
       │                               │    detected; rewrites
       │                               │    DQ+DQ→Conv→Q into native INT8
       │                               │    Conv + Requantize matching the
       │                               │    shape consumed by the fusion
       │                               │    pass)
       │                               ▼
       │      Pass 3: convert_nchw_to_nhwc
       │                               │   (transposes conv weights
       │                               │    [O,I,H,W] → [H,W,I,O] and
       │                               │    activation shapes [N,C,H,W] → [N,H,W,C].
       │                               │    Task 009 added INT8/F16
       │                               │    weight-transpose paths.)
       │                               ▼
       │      Pass 4: fold_gemm_transpose
       │                               │   (pre-transposes Gemm B-weights so
       │                               │    runtime can use the non-transposed
       │                               │    gemm_f32 shader)
       │                               ▼
       │      Pass 5 (optional, INT8 path):
       │                    QuantizedGraphCompiler::compile
       │                               │   (requires calibration data —
       │                               │    not needed when the model is
       │                               │    already QDQ-quantized; in that
       │                               │    case Pass 2b does the work)
       │                               ▼
       │      Pass 6: apply_fusion_passes
       │                               │   (Conv+Relu, Sigmoid×Mul → SiLU,
       │                               │    INT8 Conv+Requant(+Relu) → fused)
       │                               ▼
       │                    ┌────────────────────────┐
       │                    │  Graph (compiled, fused│
       │                    │         + transposed)  │
       │                    └──────────┬─────────────┘
       │                               │
       │      VulkanGraphRuntime::new ──┤   (creates SlabAllocator,
       │                                │    one vkAllocateMemory for the
       │                                │    whole model, allocates VkBuffer
       │                                │    sub-ranges per tensor, uploads
       │                                │    initialisers via persistent map)
       │                                ▼
       └────────► rt.run() — single command buffer, barriers between
                  dependent ops, one queue_submit + one wait_idle.
```

## What happens inside `VulkanGraphRuntime::run()`

```rust
pub fn run(&mut self) -> Result<()> {
    let ops = self.graph.ops.clone();
    let mut recorder = None;

    for (i, op) in ops.iter().enumerate() {
        if op_needs_cpu_fallback(op) {
            // Flush any pending GPU work — CPU dispatch needs coherent buffers.
            if let Some(rec) = recorder.take() {
                rec.finish_and_submit()?;
                self.backend.synchronize()?;
            }
            self.dispatch_op(op)?;   // download → CPU op → upload
            continue;
        }
        let rec = recorder.get_or_insert_with(|| OpsRecorder::begin(&self.backend).unwrap());
        if i > 0 && needs_barrier(&ops, i) {
            rec.record_memory_barrier();
        }
        self.record_op(rec, op)?;     // bind pipeline, push, dispatch
    }

    if let Some(rec) = recorder.take() { rec.finish_and_submit()?; }
    self.backend.synchronize()?;
    Ok(())
}
```

The runtime is **dual-mode**:

* **GPU mode** for any op with a Vulkan shader (`Conv`, `Gemm`, `Add`,
  `Mul`, `Sigmoid`, `SiLU`, `MaxPool`, `Softmax`, `Relu`, `Clip`,
  `Reshape`, `Quantize/Dequantize/Requantize`, `AddQuantized`,
  `Conv2dRequantReluI8`).
* **CPU fallback** for `Sub`, `Div`, `Concat`, `Resize`, `Split`,
  `Transpose`, `Slice` (downloads inputs, runs `dragonwing_cpu::ops`,
  uploads result). These are pure data-movement / minor-arithmetic ops
  that show up in YOLO model heads.

## Memory layout

A single `VkDeviceMemory` slab (typically 16 MiB minimum, sized to fit
the model + 4 MiB headroom on construction) backs every tensor:

```
┌──────────────────────────────────────────────────────────────────┐
│  VkDeviceMemory slab (host-visible + host-coherent + device-local)│
├──────────────────────────────────────────────────────────────────┤
│  input(images)│weight_0│bias_0│act_0│weight_1│...                │
│   ↑           ↑        ↑     ↑    ↑                              │
│   VkBuffer    VkBuffer …                                          │
│   (logical    (each bound at a distinct offset via                │
│   size in     vkBindBufferMemory)                                 │
│   len_bytes)                                                      │
└──────────────────────────────────────────────────────────────────┘
```

Verified on Arduino UNO Q with a 103 MB YOLOv8m model: **1 slab,
811 MiB in use, 99.5% utilization**.

## Op coverage matrix

| ONNX op          | F32 Vulkan | INT8 Vulkan | CPU fallback |
|------------------|------------|-------------|--------------|
| Conv             | ✅ (+bias)  | ✅ fused     | —            |
| Gemm             | ✅ (+bias, trans_b) | (folded into Conv) | — |
| Add              | ✅         | ✅ (AddQuantized) | — |
| Mul              | ✅         | (via SiLU fusion) | — |
| Sigmoid          | ✅         | — | — |
| SiLU (fused)     | ✅         | — | — |
| Relu             | ✅ (in-place) | ✅ (packed) | — |
| Clip (Relu6)     | ✅ (Relu only) | — | — |
| MaxPool          | ✅         | — (CPU possible) | — |
| Softmax          | ✅         | — | — |
| Reshape/Flatten  | metadata   | metadata    | — |
| Quantize         | ✅         | ✅          | — |
| Dequantize       | ✅         | ✅          | — |
| Requantize       | ✅         | ✅ (or absorbed via fusion) | — |
| Sub              | —          | —           | ✅            |
| Div              | —          | —           | ✅            |
| Concat           | —          | —           | ✅            |
| Resize           | —          | —           | ✅ (nearest, linear) |
| Split            | —          | —           | ✅            |
| Transpose        | —          | —           | ✅            |
| Slice            | —          | —           | ✅            |
| GlobalAvgPool    | —          | —           | ⚠️ not yet    |
| AvgPool          | —          | —           | ⚠️ not yet    |

## Diagnostics

### Tracing each op

```bash
DRAGONWING_TRACE=1 UNBATCHED=1 ./target/release/yolo-benchmark model.onnx
```

prints per-op shape lines:

```
[op   0] /model.0/conv/Conv (Conv) in=["images[1, 640, 640, 3]", ...] out=[...]
[op   1] /model.0/act/Sigmoid (Sigmoid) in=[...] out=[...]
```

and inserts a `synchronize()` after each op so the failing op is
reported precisely.

### Bisecting a GPU fault

```bash
MAX_OPS=4 UNBATCHED=1 DRAGONWING_TRACE=1 ./target/release/yolo-benchmark model.onnx
```

Truncates the graph to the first 4 ops. Combine with `MAX_OPS=2`,
`MAX_OPS=3` etc. to bisect.

### Inspecting GPU kernel logs

When a hang or fault happens, check `dmesg` for entries like:

```
msm_dpu 5e01000.display-controller: [drm:hangcheck_handler [msm]] *ERROR* 7.0.2.0: hangcheck detected gpu lockup rb 0!
*** gpu fault: ttbr0=... iova=... dir=READ|WRITE type=TRANSLATION source=UCHE (0,0,0,0)
```

`TRANSLATION` faults during hang-recovery are often **secondary**:
the original problem is the dispatch ran too long for MSM's 500 ms
hangcheck timeout, and the recovery worker resets the GPU state. The
fix is to make the dispatch faster (use INT8 / fused kernels, or tile
the shader) — not to add address-space mappings.

`maxComputeWorkGroupCount[0] = 65535` is enforced by Adreno A702.
1D-dispatch shaders that exceed this on large tensors will produce a
**real** TRANSLATION fault. The 2D-wrap dispatch pattern documented in
`vulkan-runtime-integration.md` (Task 008) avoids this.

## Quantization workflow (when calibration data is available)

```rust
use dragonwing_onnx::{
    apply_fusion_passes, compile_model, convert_nchw_to_nhwc,
    fold_batchnorm, fold_gemm_transpose, load_model, Calibrator,
    QuantizedGraphCompiler, VulkanGraphRuntime,
};
use dragonwing_core::Dtype;
use dragonwing_vulkan::{VulkanBackend, VulkanConfig};

// 1. Load + graph passes (F32).
let mut model = load_model("yolov8n.onnx")?;
fold_batchnorm(&mut model)?;
let mut graph_f32 = compile_model(&model, Dtype::F32)?;
convert_nchw_to_nhwc(&mut graph_f32)?;
fold_gemm_transpose(&mut graph_f32)?;

// 2. Calibrate: feed ~100 representative images.
let mut cal = Calibrator::new(graph_f32.clone())?;
for image in calibration_images {
    cal.feed("images", &image)?;
}
let params = cal.compute_params()?;

// 3. Compile to INT8.
let compiler = QuantizedGraphCompiler::new(params);
let mut graph_i8 = compiler.compile(graph_f32)?;

// 4. Apply fusion — this collapses Conv → Requantize → Relu
//    triples into the fused Conv2dRequantReluI8 op which uses the
//    conv2d_requant_relu_i8_packed.spv shader.
apply_fusion_passes(&mut graph_i8);

// 5. Run on Vulkan.
let backend = VulkanBackend::new(VulkanConfig::default())?;
let mut rt = VulkanGraphRuntime::new(graph_i8, backend)?;

rt.set_input_f32("images", &input_640x640)?;  // F32 in, quantized internally
rt.run()?;
let out = rt.get_output_f32("output0")?;
```

## Performance targets and current state

| Metric | Task 008 target | Achieved (yolov8m F32) |
|--------|----------------|------------------------|
| `vkAllocateMemory` calls per `VulkanGraphRuntime` | <10 | **1** |
| Command-buffer submits per `run()` | 1 (or one per CPU-fallback boundary) | 1 if no CPU fallback ops, else `1 + n_cpu_ops` |
| Slab utilization | high | **99.5%** |
| YOLOv8n INT8 inference | <150 ms (<100 ms stretch) | not measured — needs calibration data |
| YOLOv8m F32 inference | not a stated target | blocked at first mid-network conv by hangcheck (naive shader too slow on c_in=48, c_out=96 shape; ~1 GFLOP/dispatch) |

## QDQ-format INT8 models (Task 009)

When the loaded model contains `QuantizeLinear` / `DequantizeLinear`
nodes (i.e., it was exported via ONNX Runtime static quantization,
TensorRT, or the bundled
[`scripts/export_int8_onnx.py`](../scripts/export_int8_onnx.py)),
`yolo-benchmark` runs an additional **Pass 2b** between
`compile_model` and `convert_nchw_to_nhwc`:

```rust
if dragonwing_onnx::graph_has_qdq(&graph) {
    let stats = dragonwing_onnx::fold_qdq_patterns(&mut graph)?;
    println!("conv_folded={} dq_removed={} q_removed={}",
        stats.conv_folded,
        stats.dequantize_removed,
        stats.quantize_removed);
}
```

The fold pass rewrites `DQ + DQ → Conv → Q` triples into native INT8
`Conv + Requantize`, producing the exact graph shape the existing
fusion pass expects. `apply_fusion_passes` then collapses each pair
into `OpParams::Conv2dRequantReluI8Nhwc`.

End-to-end workflow:

```bash
# 1. Export INT8 ONNX (one-time).
python3 scripts/export_int8_onnx.py \
    --pt artifacts/models/excavator_stone/truck.yolov8m.p640.20250512_best.pt \
    --calib /path/to/calibration/images \
    --out artifacts/models/truck.yolov8m.p640.int8.onnx \
    --imgsz 640

# 2. Run the benchmark (the QDQ-fold pass runs automatically).
./target/release/yolo-benchmark artifacts/models/truck.yolov8m.p640.int8.onnx
```

Set `DRAGONWING_QDQ_DEBUG=1` to print one stderr line per conv that
the fold pass *did not* match, with the rejection reason.

See [`docs/onnx-qdq.md`](onnx-qdq.md) for the full format spec,
fold pattern, and the v1 bias-skip limitation.

## See also

- `docs/onnx-qdq.md` — ONNX QDQ format support (Task 009).
- `docs/vulkan-runtime-integration.md` — `VulkanGraphRuntime`
  architecture, including Task 008 single-command-buffer + slab changes.
- `docs/int8-vulkan.md` — UINT32 packing, INT8 shader bindings.
- `docs/op-fusion.md` — Fusion-pass design (Conv+Relu, SiLU,
  Conv+Requant+Relu).
- `docs/quantization.md` — Calibration + `QuantizedGraphCompiler`.
- `.task/implementation-task-008.md` — Per-phase findings, the GPU
  hangcheck investigation, and acceptance matrix.
- `.task/implementation-task-009.md` — QDQ implementation findings.
