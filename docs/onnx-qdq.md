# ONNX QDQ Format Support (Task 009)

dragonwing-edge consumes the **QDQ** (Quantize-Dequantize) format used
by the major ONNX quantization toolchains — ONNX Runtime
`quantize_static`, TensorRT exporters, and the Ultralytics tooling that
wraps them.

This document covers:

- What QDQ format looks like in practice.
- Which patterns dragonwing can fold to native INT8.
- Which patterns it leaves alone (and why).
- How to produce a compatible model from a `.pt` checkpoint.
- Troubleshooting.

> See also: [`quantization.md`](quantization.md) for the internal
> calibration-based path (`QuantizedGraphCompiler`).

## 1. The QDQ pattern

A quantized op in QDQ form looks like:

```
              ┌─────────────────────────────────────────┐
              │  s_x, zp_x  (initializers)               │
              ▼                                          │
  x_i8  ── DequantizeLinear ── x_f32 ─┐                  │
                                       ├─→ Conv  ─→ y_f32  ─→ QuantizeLinear ─→ y_i8
  w_i8  ── DequantizeLinear ── w_f32 ─┘                                  ▲
                                                                          │
                                                                  s_y, zp_y
```

Every quantizable op is **sandwiched** between `QuantizeLinear` and
`DequantizeLinear` wrappers. The actual math node (`Conv`, `Add`,
`Gemm`, `MatMul`, …) still consumes F32 tensors; the wrappers
advertise the quantization metadata.

Scale and zero-point are **always** initializers (i.e., baked into the
model file at export time). dragonwing rejects dynamic scales — they
are not part of the QDQ format anyway.

## 2. Supported patterns (v1)

### 2.1 Builders

| ONNX op | Op-params produced | Notes |
|---|---|---|
| `QuantizeLinear` (scalar scale) | `OpParams::Quantize { scale }` | Per-tensor. |
| `QuantizeLinear` (1-D scale)    | `OpParams::QuantizePerChannel { scales, zero_points, axis }` | Per-channel. |
| `DequantizeLinear` (scalar)     | `OpParams::Dequantize { scale }` | Per-tensor. |
| `DequantizeLinear` (1-D)        | `OpParams::DequantizePerChannel { ... }` | Per-channel. |

Both builders honour the `axis` attribute (ONNX 13+) including
negative values; the default is `axis=1`.

### 2.2 Fold patterns

[`fold_qdq_patterns`](../crates/dragonwing-onnx/src/qdq.rs) rewrites
this exact shape:

```
DequantizeLinear(act_i8, s_x) ─┐
                                ├─→ Conv  ─→ QuantizeLinear(out_f32, s_y) → out_i8
DequantizeLinear(w_i8,   s_w) ─┘
```

into:

```
Conv(act_i8, w_i8) ─→ Requantize(_i32 → out_i8, (s_x · s_w) / s_y)
```

The downstream [`apply_fusion_passes`](op-fusion.md) then collapses
`Conv + Requantize (+ Relu)` into a single
`OpParams::Conv2dRequantReluI8Nhwc` Vulkan dispatch.

**Successfully folded:**

- Conv with two QDQ-wrapped inputs (activation + weight).
- Either an INT8 weight initializer (typical ORT export) or an F32
  weight initializer (the script quantizes it on the fly using the
  scale advertised by the weight DQ).
- Per-tensor or per-channel weight scales. Per-channel scales are
  averaged into a single scalar `requant_scale` for the fused shader;
  the per-channel vector is still stamped on the weight `TensorShape`
  so a future per-channel shader can pick it up.
- Single-consumer DequantizeLinear outputs. If a DQ feeds multiple
  branches, that conv is left F32 (residual / split convs).

### 2.3 What stays F32

- **Softmax, Sigmoid, Tanh** — numerically fragile in INT8;
  dragonwing keeps them F32 by design.
- **Convs with bias** — see §3.
- **Slice / Reshape / Concat with dynamic indices** — these are
  shape-manipulation ops, not arithmetic; the QDQ format wraps their
  outputs but dragonwing treats them as no-ops once they propagate to
  the consumer.
- **Resize / Pooling** — these are pass-through ops in the INT8 path
  (the scale doesn't change across them).

## 3. Out of scope (v1) — bias

The standard ORT QDQ recipe quantizes the F32 bias to INT32 using
`bias_i32 = round(bias_f32 / (s_x · s_w))`, then wraps it in a
DequantizeLinear so the F32 Conv receives an F32 tensor. The Vulkan
fused shader `Conv2dRequantReluI8Nhwc` does **not** accept bias —
adding it requires either:

1. A new GLSL shader that reads the INT32 bias and adds it to the I32
   accumulator before requantize. **Preferred fix.**
2. Decomposing into `Conv (i8) → AddI32 (bias) → Requantize` — three
   dispatches, lower performance but cheaper to land.

Until the bias path lands, the QDQ-fold **skips biased convs**. The
diagnostic `DRAGONWING_QDQ_DEBUG=1` env var prints a one-line reason
for every skipped conv (see §6).

> Practical impact: typical Ultralytics → ORT QDQ exports bias every
> conv. On YOLOv8m, only 1 of 84 convs folds today. The biasless case
> (mobilenetv2 pointwise convs, simple test graphs) works fully.

## 4. Producing a QDQ model

The recommended workflow is in
[`scripts/export_int8_onnx.py`](../scripts/export_int8_onnx.py):

```bash
python3 scripts/export_int8_onnx.py \
    --pt artifacts/models/excavator_stone/truck.yolov8m.p640.20250512_best.pt \
    --calib path/to/calibration/images \
    --out artifacts/models/truck.yolov8m.p640.int8.onnx \
    --imgsz 640
```

The script runs three stages:

1. **Ultralytics F32 export** (`opset=13`, `simplify=True`).
2. **ONNX Runtime static quantization** with:
   - `QuantFormat.QDQ`
   - `activation_type=QuantType.QInt8`
   - `weight_type=QuantType.QInt8`
   - `per_channel=True`
   - `extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "CalibTensorRangeSymmetric": True}`
3. **`onnxsim.simplify`** on the quantized model — folds the constant
   chains ORT inserts in detection heads.

Why we don't use `model.export(int8=True)`: Ultralytics 8.x dropped
first-party INT8 support for the `onnx` target. INT8 there now lives
in `tflite`, `openvino`, `engine`, etc. The ORT static-quantization
path is the supported way to produce ONNX QDQ models.

### 4.1 Calibration images

- ~100 representative frames at the inference resolution.
- Diverse lighting and angles.
- Same domain as deployment scenes (e.g., excavator dashboard footage
  for `truck.yolov8m.p640`).

For pipeline smoke-testing without representative data, pass
`--random-calib 32` to use random uniform tensors. The resulting QDQ
graph is structurally valid but the scales are meaningless — useful
only for compile-time verification.

### 4.2 Symmetric quantization is mandatory

ORT's default `quantize_static` produces **asymmetric** INT8
quantization with `zero_point = 128` even when `activation_type=QInt8`.
dragonwing rejects non-zero zero-points (see §5). Always set:

```python
extra_options={"ActivationSymmetric": True, "WeightSymmetric": True}
```

The script does this automatically.

## 5. Why symmetric only?

Asymmetric INT8 with non-zero zero-points requires a **bias-correction
pass** during requantize:

```
out = saturate( ((acc_i32 - zp_x · sum_w) · m_int + nudge) >> shift )
```

vs. the simpler symmetric case:

```
out = saturate( (acc_i32 · m_int + nudge) >> shift )
```

The extra `sum_w` term must be precomputed per output channel during
graph compilation. The math is well-known but adds engineering
surface; v1 keeps the implementation focused on the common symmetric
case (which is what TensorRT, GPU PTQ, and most production INT8
pipelines use anyway).

If asymmetric support becomes necessary, the Phase 1 builders already
parse the zero-point input correctly — they just reject non-zero
values today.

## 6. Troubleshooting

### 6.1 `validation error: ... uses asymmetric quantization`

Cause: the model was exported with default ORT settings.
Fix: re-export with `extra_options={"ActivationSymmetric": True,
"WeightSymmetric": True}` (or use
[`scripts/export_int8_onnx.py`](../scripts/export_int8_onnx.py)
which sets this).

### 6.2 `Slice: ... must be constant`

Cause: YOLO detection heads contain dynamic Slice ops whose
`starts/ends` are computed from `Shape → Gather → Mul`. ORT's
quantizer doesn't fold these.

Fix: run `onnxsim.simplify` on the quantized model. The export
script's stage 3 does this automatically.

### 6.3 `conv_folded=N` is much smaller than the conv count

Set `DRAGONWING_QDQ_DEBUG=1` and re-run. Each rejected conv emits a
stderr line:

```
[qdq-debug] skip Conv 'model.0.conv' (3 inputs): conv has bias '...' (v1 fold supports biasless convs only)
[qdq-debug] skip Conv 'model.4.add' (2 inputs): activation DQ output has 2 consumers (need 1)
```

Common reasons:

- **`conv has bias`** — v1 limitation (see §3). Bias support is a
  follow-up task.
- **`activation DQ output has N consumers`** — the DQ feeds a residual
  branch. Single-consumer is required to avoid breaking the other
  branch.
- **`no QuantizeLinear consumes the Conv output`** — last conv in a
  chain (rare); leave F32.

### 6.4 `Conv weight '...' must be a constant initializer`

You are running an older dragonwing build that does not have the
Phase 5 ConvBuilder fallback. Pull the latest `dev/onnx-qdq` branch.

## 7. Implementation map

| Component | File | Responsibility |
|---|---|---|
| Builders | [`builder.rs`](../crates/dragonwing-onnx/src/builder.rs) — `QuantizeLinearBuilder`, `DequantizeLinearBuilder` | Parse Q/DQ nodes into `OpParams::Quantize{,PerChannel}` and `OpParams::Dequantize{,PerChannel}`. |
| Fold pass | [`qdq.rs`](../crates/dragonwing-onnx/src/qdq.rs) — `fold_qdq_patterns` | Rewrite `DQ + DQ → Conv → Q` into native INT8 conv + requantize. |
| Pipeline hook | [`yolo_benchmark.rs`](../crates/dragonwing-test/src/bin/yolo_benchmark.rs) — Pass 2b | Conditional on `graph_has_qdq`; runs between `compile_model` and `convert_nchw_to_nhwc`. |
| Per-channel CPU kernel | [`runtime.rs`](../crates/dragonwing-onnx/src/runtime.rs) — `quantize_per_channel_f32_to_i8`, `dequantize_per_channel_i8_to_f32` | CPU fallback for non-folded per-channel QDQ ops. |
| Export tool | [`scripts/export_int8_onnx.py`](../scripts/export_int8_onnx.py) | Two-stage `.pt` → F32 ONNX → INT8 QDQ ONNX. |
| Diagnostics | `DRAGONWING_QDQ_DEBUG=1` env var | Per-conv reasons when fold rejects a candidate. |

## 8. Future work

Tracked as follow-up to task 009:

- **Bias support in the fused INT8 shader.** Reads
  `bias_i32` SSBO, adds to the I32 accumulator before requantize.
  Unblocks the ~98% of real ORT-quantized convs that carry bias.
- **Per-channel requant in Vulkan.** Replaces the scalar
  `requant_scale` push-constant with a per-channel buffer; uses the
  `per_channel_scales` already stamped on the weight `TensorShape`.
- **Asymmetric quantization (non-zero zero-points).** Compute and
  fold the per-channel `sum_w` correction term during fold; relax the
  symmetric-only rejection in the builders.
- **Gemm / MatMul QDQ folding.** Mirror the Conv fold logic for
  `OpParams::Gemm`. The detection-head MatMul is the obvious user.
