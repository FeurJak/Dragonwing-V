# INT8 Quantization

This document describes the INT8 quantization system in dragonwing-edge-rs, enabling 2-4x inference speedup and 4x memory reduction on edge devices.

## Overview

Quantization converts floating-point (F32) model weights and activations to 8-bit integers (INT8), significantly reducing memory bandwidth and leveraging integer SIMD instructions for faster inference.

### Benefits

| Metric | F32 | INT8 | Improvement |
|--------|-----|------|-------------|
| Weight size | 4 bytes | 1 byte | 4x smaller |
| Activation size | 4 bytes | 1 byte | 4x smaller |
| SIMD throughput | 4 ops (128-bit) | 16 ops (128-bit) | 4x higher |
| Memory bandwidth | Baseline | 4x reduced | 4x faster loads |

### Trade-offs

- **Accuracy loss**: Typically <2% for well-calibrated models
- **Complexity**: Requires calibration dataset and scale management
- **Accumulator size**: INT32 accumulators needed for intermediate results

## Quantization Scheme

We use **symmetric per-tensor quantization** for activations and **symmetric per-channel quantization** for weights.

### Symmetric Quantization

For a tensor with range [-max, +max]:

```
scale = max(|tensor|) / 127
quantized = round(tensor / scale)
dequantized = quantized * scale
```

Zero-point is always 0, simplifying computation:
- No bias correction in convolutions
- Simpler shader code
- Faster inference

### Per-Channel vs Per-Tensor

| Type | Granularity | Use Case |
|------|-------------|----------|
| Per-tensor | One scale per tensor | Activations (dynamic range) |
| Per-channel | One scale per output channel | Weights (different channel magnitudes) |

Per-channel quantization for weights preserves more precision by allowing each output channel to have its own scale.

## Quantization Flow

```
┌────────────────┐     ┌─────────────────┐     ┌────────────────┐
│ F32 Model      │────►│ Calibration     │────►│ Quantized      │
│ (ONNX)         │     │ (100+ samples)  │     │ Graph (INT8)   │
└────────────────┘     └─────────────────┘     └────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
    Load weights          Collect stats          Execute INT8
    Parse graph           Compute scales         ops on CPU/GPU
```

### Step 1: Compile F32 Graph

```rust
use dragonwing_onnx::compile_model;
use dragonwing_core::Dtype;

let f32_graph = compile_model(&model, Dtype::F32)?;
```

### Step 2: Calibration

Run inference on representative data to collect activation statistics:

```rust
use dragonwing_onnx::calibration::Calibrator;

let mut calibrator = Calibrator::new(f32_graph.clone())?;

// Feed calibration samples (typically 100-500 images)
for image in calibration_images {
    calibrator.feed("input", &image)?;
}

// Compute quantization parameters
let quant_params = calibrator.compute_params()?;
```

### Step 3: Quantized Compilation

Transform F32 graph to INT8:

```rust
use dragonwing_onnx::QuantizedGraphCompiler;

let compiler = QuantizedGraphCompiler::new(quant_params);
let i8_graph = compiler.compile(f32_graph)?;
```

### Step 4: INT8 Inference

```rust
use dragonwing_onnx::GraphRuntime;
use dragonwing_cpu::CpuBackend;

let backend = CpuBackend::new();
let mut runtime = GraphRuntime::new(i8_graph, backend)?;

// Input is still F32 (quantized internally)
runtime.set_input("input", &input_f32)?;
runtime.run()?;

// Output is dequantized to F32
let output_f32 = runtime.get_output("output")?;
```

## Calibration Strategies

### Min/Max (Default)

Uses the observed min and max values directly:

```rust
let strategy = CalibrationStrategy::MinMax;
```

**Pros**: Simple, deterministic
**Cons**: Sensitive to outliers

### Percentile

Uses percentile (e.g., 99.99%) instead of absolute min/max:

```rust
let strategy = CalibrationStrategy::Percentile { percentile: 9999 };
```

**Pros**: Robust to outliers
**Cons**: May clip extreme values

### Calibration Dataset Guidelines

| Factor | Recommendation |
|--------|----------------|
| Size | 100-500 samples minimum |
| Diversity | Cover all expected input variations |
| Representation | Use data similar to production |
| Reproducibility | Use fixed random seed |

## INT8 Ops

### CPU Implementation

All INT8 ops have NEON-optimized implementations with scalar fallbacks:

| Op | CPU (NEON) | CPU (Scalar) | Notes |
|----|------------|--------------|-------|
| `gemm_i8` | vmlal_s8 + vpaddl | multiply-accumulate | INT32 accumulator |
| `conv2d_i8_nhwc` | NEON unrolled | nested loops | Uses GEMM pattern |
| `add_i8` | NEON vectorized | element-wise | Scale adjustment |
| `relu_i8` | vmaxq_s8 | max(0, x) | Pass-through for positive |
| `requantize_i32_to_i8` | NEON with saturation | clamp + round | For layer transitions |
| `quantize_f32_to_i8` | NEON vectorized | scale + round | Input quantization |
| `dequantize_i8_to_f32` | NEON vectorized | scale multiply | Output dequantization |

### Vulkan Implementation

Without `VK_KHR_8bit_storage`, INT8 values are packed into UINT32:

```glsl
// Pack 4 INT8 values into UINT32
uint packed = uint(a & 0xFF) | (uint(b & 0xFF) << 8) | 
              (uint(c & 0xFF) << 16) | (uint(d & 0xFF) << 24);

// Unpack and compute
int a_signed = int(packed & 0xFF) - 128;
int b_signed = int((packed >> 8) & 0xFF) - 128;
// ... compute with signed values
```

See `docs/int8-vulkan.md` for details.

## Accuracy Guidelines

### Expected Accuracy Loss

| Model Type | Typical INT8 Accuracy Loss |
|------------|---------------------------|
| MobileNetV2 | <1% top-1 |
| ResNet-50 | <0.5% top-1 |
| YOLOv8n | <2% mAP |

### Factors Affecting Accuracy

1. **Calibration quality**: More diverse samples = better scales
2. **Model architecture**: Depth-separable convs are more sensitive
3. **Activation ranges**: Narrow ranges quantize better
4. **Outliers**: Extreme values reduce effective bit-width

### Per-Layer Sensitivity

Some layers are more sensitive to quantization:
- First and last layers (direct input/output)
- Layers with small channel counts
- Skip connections (scale mismatch)

For problematic models, keep sensitive layers in F32 (mixed precision).

## Testing

### Unit Tests

```bash
cargo test -p dragonwing-test int8
```

Tests include:
- `test_quantize_dequantize_roundtrip` - Q/DQ accuracy
- `test_int8_gemm_vs_f32` - GEMM parity with F32
- `test_int8_conv2d` - Conv2D parity with F32
- `test_int8_multi_layer_accuracy` - Multi-layer network
- `test_int8_yolo_simulation` - YOLO-like network
- `test_int8_performance_benchmark` - Speed comparison

### Tolerance Guidelines

| Test Type | Tolerance | Rationale |
|-----------|-----------|-----------|
| Single op | 10% relative | Quantization error per op |
| Multi-layer | 20% relative | Error accumulation |
| End-to-end YOLO | 25% relative | Deep networks accumulate more |

### Debugging Quantization Issues

1. **Check scale values**: Scales should be ~1e-3 to 1e-1 for [-1,1] normalized data
2. **Verify calibration stats**: Min/max should cover actual data range
3. **Compare layer-by-layer**: Run F32 and INT8 side-by-side
4. **Check for outliers**: Large |min| or |max| reduces precision

## Performance Optimization

### Memory Layout

- Keep tensors 4-byte aligned for UINT32 packing
- Pad channel dimension to multiple of 4
- Use NHWC format for efficient channel access

### Workgroup Sizing (Vulkan)

```glsl
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
```

8x8 workgroups work well for most INT8 convolutions.

### Requantization Strategy

Minimize requantization ops:
- Fuse requantize with activation (ReLU clips anyway)
- Batch requantization at layer boundaries
- Keep computation in INT32 as long as possible

## Code References

- `crates/dragonwing-core/src/quantization.rs` - Scale types, QuantizationParams
- `crates/dragonwing-core/src/dtype.rs` - Dtype::I8 definition
- `crates/dragonwing-cpu/src/ops.rs` - INT8 CPU ops (line 2949+)
- `crates/dragonwing-onnx/src/calibration.rs` - Calibrator implementation
- `crates/dragonwing-onnx/src/quantize.rs` - QuantizedGraphCompiler (internal calibration path)
- `crates/dragonwing-onnx/src/qdq.rs` - QDQ-fold pass for ONNX-format quantization (Task 009)
- `crates/dragonwing-shaders/glsl/*_i8_packed.comp` - Vulkan INT8 shaders
- `crates/dragonwing-test/src/int8_e2e.rs` - INT8 tests
- `scripts/export_int8_onnx.py` - Ultralytics → ORT QDQ exporter (Task 009)

## Two paths to INT8

dragonwing-edge supports two complementary paths to INT8 inference:

1. **Internal calibration** (`QuantizedGraphCompiler` + `Calibrator`).
   Take an F32 ONNX model and ~100 representative input frames; the
   calibrator measures activation ranges and the compiler emits an
   INT8 graph. Use this when you only have F32 weights and your
   deployment frames.

2. **ONNX QDQ import** (`fold_qdq_patterns`, Task 009). Take a
   QDQ-quantized ONNX model produced by ONNX Runtime / TensorRT /
   our [`export_int8_onnx.py`](../scripts/export_int8_onnx.py)
   helper; the fold pass rewrites the QDQ wrappers into native INT8
   ops. Use this when you already have an INT8 model from another
   toolchain or want to use ORT's mature PTQ algorithms.

Both paths feed into the same fused `Conv2dRequantReluI8Nhwc` Vulkan
shader. See [`onnx-qdq.md`](onnx-qdq.md) for the QDQ details.

## Future Work

- [x] ~~ONNX QDQ format import~~ — landed in Task 009 (see `onnx-qdq.md`).
- [ ] **Bias support in the fused INT8 conv shader.** Required to land
      end-to-end INT8 YOLOv8m; today ~98% of real-world Ultralytics
      convs carry bias and the fold pass skips them. See
      `onnx-qdq.md` §8 for the implementation sketch.
- [ ] **Per-channel requant in Vulkan.** The per-channel scales are
      already plumbed through `TensorShape::per_channel_scales`; need
      a shader that consumes them.
- [ ] **Asymmetric quantization (non-zero zero-points)** for QDQ models
      exported with default ORT settings.
- [ ] Quantization-aware training (QAT) support
- [ ] Per-group quantization for larger models
- [ ] INT4 exploration (if hardware supports)
- [ ] Dynamic quantization (per-batch scales)
