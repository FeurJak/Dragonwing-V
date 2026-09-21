//! ONNX QDQ (Quantize-Dequantize) fold pass.
//!
//! Standard ONNX quantization toolchains (Ultralytics, ONNX Runtime,
//! TensorRT) emit quantized models in **QDQ format**: every operator is
//! sandwiched between `QuantizeLinear` / `DequantizeLinear` nodes that
//! advertise the quantization parameters. The math is still expressed as
//! F32 ops; the QDQ wrappers are a contract between exporter and runtime
//! that says "this op may run in INT8 with these scales".
//!
//! ```text
//!   ... → Quantize(x, s_x) → x_i8
//!         DequantizeLinear(x_i8,   s_x) ──┐
//!         DequantizeLinear(w_i8,   s_w) ──┼─→ Conv (F32)
//!                                         │      │
//!                                         │      └→ QuantizeLinear(y, s_y) → y_i8 → ...
//! ```
//!
//! This module rewrites that pattern into the native INT8 form already
//! understood by [`apply_fusion_passes`](crate::apply_fusion_passes):
//!
//! ```text
//!   ... → Quantize(x, s_x) → x_i8
//!         Conv(x_i8, w_i8)              -- inputs swapped to the pre-DQ I8 tensors
//!           ↓ (output kept at "y" but flagged I8)
//!         Requantize(y_i32 → y_i8, (s_x*s_w)/s_y)
//! ```
//!
//! The downstream fusion pass collapses `Conv + Requantize (+Relu)` into a
//! single `OpParams::Conv2dRequantReluI8Nhwc` dispatch.
//!
//! # Supported patterns (v1)
//!
//! - **Conv-QDQ**: `DQ(act) + DQ(weight) → Conv → Q` with symmetric
//!   quantization (zero-point = 0).
//! - **Weight initializer**: any dtype. F32 weights are quantized at fold
//!   time using the scale advertised by the weight DQ. INT8 weights (the
//!   typical Ultralytics export) are kept as-is.
//! - **Optional bias**: F32 bias on the Conv flows through untouched. The
//!   bias will be reabsorbed by the existing Phase 1 fusion path or stays
//!   as a follow-on F32 add. Conv-with-bias is left F32 in v1.
//!
//! # Out of scope (v1)
//!
//! - Gemm / MatMul QDQ folding (task spec lists Gemm as a future deliverable).
//! - Branch sharing (`DQ` with multiple consumers stays in place; only the
//!   single-consumer case folds).
//! - Asymmetric quantization (already rejected by the QDQ builders).

use crate::builder::{CompiledOp, OpParams, TensorShape};
use crate::error::{Error, Result};
use crate::graph::Graph;
use dragonwing_core::{Dtype, PerChannelScale, QuantScale};
use std::collections::{HashMap, HashSet};

/// Statistics returned by [`fold_qdq_patterns`].
#[derive(Debug, Default, Clone, Copy)]
pub struct QdqFoldStats {
    /// Number of `Conv → Q` chains rewritten to INT8 conv + requant.
    pub conv_folded: usize,
    /// Number of `DequantizeLinear` nodes removed from the graph.
    pub dequantize_removed: usize,
    /// Number of `QuantizeLinear` nodes removed from the graph.
    pub quantize_removed: usize,
    /// Number of F32 weight initializers quantized on the fly.
    pub weights_quantized: usize,
    /// Number of QDQ ops left in place (could not be folded).
    pub qdq_unfolded: usize,
}

impl QdqFoldStats {
    /// Total number of QDQ nodes removed.
    pub fn nodes_removed(&self) -> usize {
        self.dequantize_removed + self.quantize_removed
    }
}

/// Detect whether the graph contains any QDQ wrappers.
///
/// Cheap check used to decide whether to run [`fold_qdq_patterns`].
pub fn graph_has_qdq(graph: &Graph) -> bool {
    graph
        .ops
        .iter()
        .any(|op| op.op_type == "QuantizeLinear" || op.op_type == "DequantizeLinear")
}

/// Fold ONNX QDQ patterns into native INT8 ops.
///
/// Returns a `QdqFoldStats` summary describing how many patterns matched
/// and how many QDQ nodes remain in the graph (which should be a small
/// number for well-formed QDQ models — typically only the initial input
/// quantize and the final output dequantize).
///
/// # Errors
///
/// Returns an error if the graph references QDQ scales that cannot be
/// resolved or if a folded pattern would leave the graph in an
/// inconsistent state. The graph is modified in place; on error it may
/// be left partially folded.
pub fn fold_qdq_patterns(graph: &mut Graph) -> Result<QdqFoldStats> {
    let mut stats = QdqFoldStats::default();

    // Index pass: producer / consumer maps used throughout.
    let producers = build_producer_map(graph);
    let consumer_counts = build_consumer_count_map(graph);

    // Collect Conv ops to fold. We snapshot indices up-front because
    // the rewrite step will mutate `graph.ops`.
    let conv_indices: Vec<usize> = graph
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == "Conv")
        .map(|(i, _)| i)
        .collect();

    // ops/inits to drop at the end (sorted in reverse before draining).
    let mut ops_to_remove: HashSet<usize> = HashSet::new();
    let mut new_ops: Vec<(usize, Vec<CompiledOp>)> = Vec::new(); // (replace_idx, replacement)
    // weight initializer renames: old_name → new_name (only when we
    // had to quantize an F32 weight ourselves).
    let mut weight_replacements: Vec<(String, Vec<u8>, TensorShape)> = Vec::new();
    // tensor shape updates (dtype flips to I8 when we wire a tensor as
    // the input/output of an INT8 conv).
    let mut shape_updates: Vec<(String, TensorShape)> = Vec::new();

    // Optional debug tracing — set DRAGONWING_QDQ_DEBUG=1 to see which
    // convs the fold pass rejected and why. Useful when an INT8 model
    // loads structurally but produces conv_folded=0.
    let debug = std::env::var("DRAGONWING_QDQ_DEBUG").is_ok();

    for conv_idx in &conv_indices {
        if debug {
            let c = &graph.ops[*conv_idx];
            let reason = trace_conv_fold_failure(graph, *conv_idx, &producers, &consumer_counts);
            if let Some(r) = reason {
                eprintln!("[qdq-debug] skip Conv '{}' ({} inputs): {}", c.name, c.inputs.len(), r);
            }
        }
        match try_fold_conv(graph, *conv_idx, &producers, &consumer_counts) {
            Some(folded) => {
                stats.conv_folded += 1;
                if folded.quantized_weight.is_some() {
                    stats.weights_quantized += 1;
                }

                // Mark DQ inputs and the output Q for removal.
                ops_to_remove.insert(folded.act_dq_idx);
                ops_to_remove.insert(folded.weight_dq_idx);
                ops_to_remove.insert(folded.output_q_idx);

                // Replace the conv with [conv_int8, requant]. The conv keeps
                // the original output name; requant writes the original Q
                // output name.
                new_ops.push((*conv_idx, vec![folded.conv_int8, folded.requant]));

                // Update shapes for the rewired conv inputs / outputs.
                for (name, shape) in folded.shape_updates {
                    shape_updates.push((name, shape));
                }

                if let Some((name, data, shape)) = folded.quantized_weight {
                    weight_replacements.push((name, data, shape));
                }
            }
            None => {
                // Pattern did not match — leave the conv alone. The
                // unfolded count is recomputed at the end based on
                // surviving QDQ ops.
            }
        }
    }

    // Apply weight initializer replacements (data + shape).
    for (name, data, shape) in weight_replacements {
        graph.initializers.insert(name.clone(), data);
        graph.shapes.insert(name, shape);
    }

    // Apply shape updates (in last-write-wins order — newer rewrites
    // win for tensors touched by multiple folds, which is what we want).
    for (name, shape) in shape_updates {
        graph.shapes.insert(name, shape);
    }

    // Replace conv ops with [conv_int8, requant] pairs.
    // Sort by index descending so later splices don't shift earlier ones.
    new_ops.sort_by(|a, b| b.0.cmp(&a.0));
    for (idx, replacement) in new_ops {
        // Remove the placeholder index from ops_to_remove (we keep this
        // op slot, but inflate it into two ops).
        ops_to_remove.remove(&idx);
        let n = replacement.len();
        graph.ops.splice(idx..=idx, replacement);
        // ops_to_remove indices that come after `idx` must be shifted
        // by (n - 1). Rebuild the set with adjusted indices.
        if n != 1 {
            let shift = n - 1;
            let adjusted: HashSet<usize> = ops_to_remove
                .iter()
                .map(|&i| if i > idx { i + shift } else { i })
                .collect();
            ops_to_remove = adjusted;
        }
    }

    // Drop the DQ / Q nodes in descending index order.
    let mut removal_indices: Vec<usize> = ops_to_remove.into_iter().collect();
    removal_indices.sort_unstable();
    removal_indices.reverse();
    for idx in removal_indices {
        // Re-classify by op_type at this surviving index for accurate stats.
        let op = &graph.ops[idx];
        match op.op_type.as_str() {
            "DequantizeLinear" => stats.dequantize_removed += 1,
            "QuantizeLinear" => stats.quantize_removed += 1,
            _ => {
                // Not a QDQ wrapper — should not happen given how we
                // populated ops_to_remove. Skip rather than panic.
                continue;
            }
        }
        graph.ops.remove(idx);
    }

    // Drop weight DQ scale/zero-point initializers + per-channel-scale
    // tensors from the QDQ wrappers if they are now unreferenced.
    prune_unused_initializers(graph);

    // Count any remaining QDQ ops (informational).
    stats.qdq_unfolded = graph
        .ops
        .iter()
        .filter(|op| op.op_type == "QuantizeLinear" || op.op_type == "DequantizeLinear")
        .count();

    Ok(stats)
}

// =============================================================================
// Internal helpers
// =============================================================================

#[derive(Debug)]
struct ConvFold {
    /// Replacement Conv op (now with I8 inputs).
    conv_int8: CompiledOp,
    /// Requantize op inserted after the Conv.
    requant: CompiledOp,
    /// Index of the activation DequantizeLinear feeding the Conv.
    act_dq_idx: usize,
    /// Index of the weight DequantizeLinear feeding the Conv.
    weight_dq_idx: usize,
    /// Index of the QuantizeLinear consuming the Conv's output.
    output_q_idx: usize,
    /// New (name, data, shape) for the weight initializer, if we had to
    /// quantize it ourselves (F32 → I8). Empty when the weight was
    /// already INT8.
    quantized_weight: Option<(String, Vec<u8>, TensorShape)>,
    /// Tensor shape updates to apply after folding.
    shape_updates: Vec<(String, TensorShape)>,
}

/// Diagnostic shadow of `try_fold_conv` — returns `Some(reason)` when the
/// fold would be skipped, or `None` when it would succeed. Only used by
/// the `DRAGONWING_QDQ_DEBUG=1` debug path.
fn trace_conv_fold_failure(
    graph: &Graph,
    conv_idx: usize,
    producers: &HashMap<String, usize>,
    consumer_counts: &HashMap<String, usize>,
) -> Option<String> {
    let conv_op = &graph.ops[conv_idx];
    if conv_op.inputs.len() > 2 && !conv_op.inputs[2].is_empty() {
        return Some(format!(
            "conv has bias '{}' (v1 fold supports biasless convs only)",
            conv_op.inputs[2]
        ));
    }
    let act_in = conv_op.inputs.first()?;
    let act_dq_idx = match producers.get(act_in) {
        Some(i) => *i,
        None => return Some(format!("no producer for activation input '{act_in}'")),
    };
    let act_dq = &graph.ops[act_dq_idx];
    if act_dq.op_type != "DequantizeLinear" {
        return Some(format!(
            "activation producer is '{}', not DequantizeLinear",
            act_dq.op_type
        ));
    }
    let act_cnt = consumer_counts.get(act_in).copied().unwrap_or(0);
    if act_cnt != 1 {
        return Some(format!("activation DQ output has {act_cnt} consumers (need 1)"));
    }
    if qdq_scalar_scale(act_dq).is_none() {
        return Some(format!(
            "activation DQ params not per-tensor: {:?}",
            act_dq.params
        ));
    }
    let w_in = conv_op.inputs.get(1)?;
    let w_dq_idx = match producers.get(w_in) {
        Some(i) => *i,
        None => return Some(format!("no producer for weight input '{w_in}'")),
    };
    let w_dq = &graph.ops[w_dq_idx];
    if w_dq.op_type != "DequantizeLinear" {
        return Some(format!(
            "weight producer is '{}', not DequantizeLinear",
            w_dq.op_type
        ));
    }
    let w_cnt = consumer_counts.get(w_in).copied().unwrap_or(0);
    if w_cnt != 1 {
        return Some(format!("weight DQ output has {w_cnt} consumers (need 1)"));
    }
    if qdq_weight_scale(w_dq, graph, w_in).is_none() {
        return Some(format!("weight DQ params not recognised: {:?}", w_dq.params));
    }
    let out = conv_op.outputs.first()?;
    let out_cnt = consumer_counts.get(out).copied().unwrap_or(0);
    if out_cnt != 1 {
        return Some(format!("conv output has {out_cnt} consumers (need 1)"));
    }
    let q_idx = graph
        .ops
        .iter()
        .position(|op| op.op_type == "QuantizeLinear" && op.inputs.first() == Some(out));
    if q_idx.is_none() {
        return Some("no QuantizeLinear consumes the Conv output".into());
    }
    if let Some(q_idx) = q_idx {
        let q = &graph.ops[q_idx];
        if qdq_scalar_scale(q).is_none() {
            return Some(format!("output Q params not per-tensor: {:?}", q.params));
        }
    }
    // Weight shape / data checks.
    let w_i8 = w_dq.inputs.first()?.clone();
    if graph.shapes.get(&w_i8).is_none() {
        return Some(format!("weight initializer '{w_i8}' has no registered shape"));
    }
    if graph.initializers.get(&w_i8).is_none() {
        return Some(format!("weight '{w_i8}' is not an initializer"));
    }
    None
}

fn try_fold_conv(
    graph: &Graph,
    conv_idx: usize,
    producers: &HashMap<String, usize>,
    consumer_counts: &HashMap<String, usize>,
) -> Option<ConvFold> {
    let conv_op = &graph.ops[conv_idx];

    // v1 limitation: only fold biasless convs. The fused INT8 shader
    // (Conv2dRequantReluI8Nhwc) does not accept bias and a bias DQ
    // would leave the Conv with mixed dtype inputs. The existing fusion
    // pass also refuses bias, so folding here would just produce a
    // dead-end op.
    if conv_op.inputs.len() > 2 && !conv_op.inputs[2].is_empty() {
        return None;
    }

    // 1. Activation DQ — producer of conv_op.inputs[0].
    let act_input_name = conv_op.inputs.first()?;
    let act_dq_idx = *producers.get(act_input_name)?;
    let act_dq = &graph.ops[act_dq_idx];
    if act_dq.op_type != "DequantizeLinear" {
        return None;
    }
    // DQ output must be single-consumer (this conv).
    if consumer_counts.get(act_input_name).copied().unwrap_or(0) != 1 {
        return None;
    }
    let (act_scale, _act_zp) = qdq_scalar_scale(act_dq)?;
    let act_i8_name = act_dq.inputs.first()?.clone();

    // 2. Weight DQ — producer of conv_op.inputs[1].
    let weight_input_name = conv_op.inputs.get(1)?;
    let weight_dq_idx = *producers.get(weight_input_name)?;
    let weight_dq = &graph.ops[weight_dq_idx];
    if weight_dq.op_type != "DequantizeLinear" {
        return None;
    }
    if consumer_counts.get(weight_input_name).copied().unwrap_or(0) != 1 {
        return None;
    }
    let weight_i8_name = weight_dq.inputs.first()?.clone();
    let weight_scale = qdq_weight_scale(weight_dq, graph, &weight_i8_name)?;

    // 3. Output Q — consumer of the Conv's output.
    let conv_out_name = conv_op.outputs.first()?;
    if consumer_counts.get(conv_out_name).copied().unwrap_or(0) != 1 {
        return None;
    }
    // Locate the Q by scanning forward; the conv_out must feed exactly
    // one op and that op must be a QuantizeLinear.
    let output_q_idx = graph
        .ops
        .iter()
        .enumerate()
        .find(|(_, op)| op.op_type == "QuantizeLinear" && op.inputs.first() == Some(conv_out_name))
        .map(|(i, _)| i)?;
    let output_q = &graph.ops[output_q_idx];
    let (output_scale, _) = qdq_scalar_scale(output_q)?;
    let final_i8_name = output_q.outputs.first()?.clone();

    // 4. Compute requant scale.
    if output_scale == 0.0 {
        return None;
    }
    let requant_scale = (act_scale * weight_scale) / output_scale;

    // 5. Handle weight initializer:
    //    - If already I8 (typical Ultralytics): keep as-is, just rewire.
    //    - If F32: quantize on the fly using the DQ scale.
    let weight_shape = graph.shapes.get(&weight_i8_name)?.clone();
    let weight_data = graph.initializers.get(&weight_i8_name)?.clone();

    let (updates, quantized_weight) = if weight_shape.dtype == Dtype::I8 {
        // Already INT8 — set per-channel scales on the shape if applicable.
        let mut new_shape = weight_shape.clone();
        if let OpParams::DequantizePerChannel { scales, .. } = &weight_dq.params {
            new_shape.per_channel_scales = Some(PerChannelScale::new(scales.clone()));
        } else if let OpParams::Dequantize { scale } = &weight_dq.params {
            new_shape.scale = Some(QuantScale::symmetric(*scale));
        }
        (vec![(weight_i8_name.clone(), new_shape)], None)
    } else if weight_shape.dtype.is_float() {
        // F32 → I8 conversion using the DQ scale(s).
        let (i8_data, new_shape) =
            quantize_weight_initializer(&weight_data, &weight_shape, &weight_dq.params).ok()?;
        let upd = vec![(weight_i8_name.clone(), new_shape.clone())];
        (upd, Some((weight_i8_name.clone(), i8_data, new_shape)))
    } else {
        // Unsupported weight dtype; leave alone.
        return None;
    };

    finalize_fold(
        conv_op,
        conv_idx,
        act_dq_idx,
        weight_dq_idx,
        output_q_idx,
        &act_i8_name,
        &weight_i8_name,
        &final_i8_name,
        requant_scale,
        graph,
        updates,
        quantized_weight,
    )
}

/// Wrapper that produces ConvFold, allowing both the fast-path (already-I8
/// weights) and the slow-path (quantize on the fly) to share construction
/// logic for the replacement ops.
fn finalize_fold(
    conv_op: &CompiledOp,
    _conv_idx: usize,
    act_dq_idx: usize,
    weight_dq_idx: usize,
    output_q_idx: usize,
    act_i8_name: &str,
    weight_i8_name: &str,
    final_i8_name: &str,
    requant_scale: f32,
    graph: &Graph,
    mut shape_updates: Vec<(String, TensorShape)>,
    quantized_weight: Option<(String, Vec<u8>, TensorShape)>,
) -> Option<ConvFold> {
    // Build the new Conv op:
    //  - inputs[0]: activation I8 (pre-DQ).
    //  - inputs[1]: weight I8 (pre-DQ).
    //  - inputs[2..]: pass through any bias / extra inputs untouched.
    let mut new_inputs = vec![act_i8_name.to_string(), weight_i8_name.to_string()];
    if conv_op.inputs.len() > 2 {
        for name in &conv_op.inputs[2..] {
            new_inputs.push(name.clone());
        }
    }

    // Both Conv and Requantize write to `final_i8_name` (the original Q
    // output). Requant's *input* is a synthesised `_i32` accumulator name.
    // This matches the exact shape produced by `QuantizedGraphCompiler::
    // create_requant_op`, which is what the existing fusion Pass 3 looks
    // for: it pairs Conv↔Requantize by matching their output names.
    //
    // The original conv's output name (`y_f32` for this example) is
    // discarded; downstream consumers were the QuantizeLinear we removed.
    let accum_name = format!("{}_i32", final_i8_name);

    let conv_int8 = CompiledOp {
        name: conv_op.name.clone(),
        op_type: "Conv".into(),
        inputs: new_inputs,
        outputs: vec![final_i8_name.to_string()],
        params: conv_op.params.clone(),
    };

    let requant = CompiledOp {
        name: format!("{}_requant", conv_op.name),
        op_type: "Requantize".into(),
        inputs: vec![accum_name],
        outputs: vec![final_i8_name.to_string()],
        params: OpParams::Requantize {
            scale: requant_scale,
        },
    };

    if let Some(s) = graph.shapes.get(final_i8_name) {
        let mut new_s = s.clone();
        new_s.dtype = Dtype::I8;
        shape_updates.push((final_i8_name.to_string(), new_s));
    }
    // Ensure the activation input is registered as I8 too (in case an
    // earlier fold didn't reach here).
    if let Some(s) = graph.shapes.get(act_i8_name) {
        if s.dtype != Dtype::I8 {
            let mut new_s = s.clone();
            new_s.dtype = Dtype::I8;
            shape_updates.push((act_i8_name.to_string(), new_s));
        }
    }

    Some(ConvFold {
        conv_int8,
        requant,
        act_dq_idx,
        weight_dq_idx,
        output_q_idx,
        quantized_weight,
        shape_updates,
    })
}

/// Extract a scalar (per-tensor) scale + zero-point from a QDQ op's
/// `OpParams`. Returns `None` for per-channel variants (which the caller
/// must handle separately).
fn qdq_scalar_scale(op: &CompiledOp) -> Option<(f32, i8)> {
    match &op.params {
        OpParams::Quantize { scale } | OpParams::Dequantize { scale } => Some((*scale, 0)),
        _ => None,
    }
}

/// Extract the (scalar approximation of the) weight scale from a weight
/// DequantizeLinear op. Per-channel scales are averaged, matching the
/// approach used by `QuantizedGraphCompiler::get_weight_scale`.
fn qdq_weight_scale(weight_dq: &CompiledOp, _graph: &Graph, _w_name: &str) -> Option<f32> {
    match &weight_dq.params {
        OpParams::Dequantize { scale } => Some(*scale),
        OpParams::DequantizePerChannel { scales, .. } => {
            if scales.is_empty() {
                None
            } else {
                let sum: f32 = scales.iter().sum();
                Some(sum / scales.len() as f32)
            }
        }
        _ => None,
    }
}

/// Quantize an F32 weight initializer to INT8 using the scale advertised
/// by the weight DQ. Per-channel scales are applied along axis 0
/// (output-channel axis for OIHW / OI).
fn quantize_weight_initializer(
    f32_bytes: &[u8],
    shape: &TensorShape,
    dq_params: &OpParams,
) -> Result<(Vec<u8>, TensorShape)> {
    if f32_bytes.len() != shape.numel() * 4 {
        return Err(Error::Compile(format!(
            "QDQ-fold: weight tensor byte length {} != numel * 4 ({})",
            f32_bytes.len(),
            shape.numel() * 4
        )));
    }

    // Read F32 weights.
    let weights: Vec<f32> = f32_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let (quantized, new_shape) = match dq_params {
        OpParams::Dequantize { scale } => {
            let inv = 1.0 / *scale;
            let q: Vec<i8> = weights
                .iter()
                .map(|w| (w * inv).round().clamp(-128.0, 127.0) as i8)
                .collect();
            let bytes = q.into_iter().map(|v| v as u8).collect();
            let mut s = shape.clone();
            s.dtype = Dtype::I8;
            s.scale = Some(QuantScale::symmetric(*scale));
            (bytes, s)
        }
        OpParams::DequantizePerChannel { scales, axis, .. } => {
            if *axis != 0 {
                return Err(Error::Compile(format!(
                    "QDQ-fold: per-channel weight axis must be 0, got {axis}"
                )));
            }
            let channels = shape.dims.first().copied().unwrap_or(0);
            if scales.len() != channels {
                return Err(Error::Compile(format!(
                    "QDQ-fold: per-channel scale count {} != channels {}",
                    scales.len(),
                    channels
                )));
            }
            let elements_per_channel = if channels == 0 {
                0
            } else {
                weights.len() / channels
            };
            let mut q = Vec::with_capacity(weights.len());
            for c in 0..channels {
                let inv = 1.0 / scales[c];
                let start = c * elements_per_channel;
                let end = start + elements_per_channel;
                for &w in &weights[start..end] {
                    q.push((w * inv).round().clamp(-128.0, 127.0) as i8 as u8);
                }
            }
            let mut s = shape.clone();
            s.dtype = Dtype::I8;
            s.per_channel_scales = Some(PerChannelScale::new(scales.clone()));
            (q, s)
        }
        other => {
            return Err(Error::Compile(format!(
                "QDQ-fold: weight DQ has unexpected params: {other:?}"
            )));
        }
    };

    Ok((quantized, new_shape))
}

fn build_producer_map(graph: &Graph) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for (i, op) in graph.ops.iter().enumerate() {
        for out in &op.outputs {
            m.insert(out.clone(), i);
        }
    }
    m
}

fn build_consumer_count_map(graph: &Graph) -> HashMap<String, usize> {
    let mut m: HashMap<String, usize> = HashMap::new();
    for op in &graph.ops {
        for inp in &op.inputs {
            *m.entry(inp.clone()).or_insert(0) += 1;
        }
    }
    // Graph outputs also count as a "consumer" — protects model output
    // tensors from being misidentified as orphaned.
    for out in &graph.outputs {
        *m.entry(out.clone()).or_insert(0) += 1;
    }
    m
}

/// Drop initializers that are no longer referenced by any op.
///
/// QDQ wrappers carry per-node scale/zero-point initializers that become
/// dead weight after folding. Removing them avoids spurious memory
/// pressure when the runtime preloads initializers.
fn prune_unused_initializers(graph: &mut Graph) {
    let mut referenced: HashSet<String> = HashSet::new();
    for op in &graph.ops {
        for inp in &op.inputs {
            referenced.insert(inp.clone());
        }
    }
    for out in &graph.outputs {
        referenced.insert(out.clone());
    }
    graph
        .initializers
        .retain(|name, _| referenced.contains(name));
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::TensorShape;

    fn shape_f32(dims: Vec<usize>) -> TensorShape {
        TensorShape::new(dims, Dtype::F32)
    }

    fn shape_i8(dims: Vec<usize>) -> TensorShape {
        TensorShape::new(dims, Dtype::I8)
    }

    fn dq_op(name: &str, input: &str, output: &str, scale: f32) -> CompiledOp {
        CompiledOp {
            name: name.into(),
            op_type: "DequantizeLinear".into(),
            inputs: vec![input.into()],
            outputs: vec![output.into()],
            params: OpParams::Dequantize { scale },
        }
    }

    fn q_op(name: &str, input: &str, output: &str, scale: f32) -> CompiledOp {
        CompiledOp {
            name: name.into(),
            op_type: "QuantizeLinear".into(),
            inputs: vec![input.into()],
            outputs: vec![output.into()],
            params: OpParams::Quantize { scale },
        }
    }

    fn conv_op(name: &str, act: &str, weight: &str, out: &str, kernel: [usize; 2]) -> CompiledOp {
        CompiledOp {
            name: name.into(),
            op_type: "Conv".into(),
            inputs: vec![act.into(), weight.into()],
            outputs: vec![out.into()],
            params: OpParams::Conv2d {
                kernel_shape: kernel,
                strides: [1, 1],
                pads: [0, 0, 0, 0],
                dilations: [1, 1],
                group: 1,
            },
        }
    }

    fn build_graph_with_qdq_conv() -> Graph {
        // Pattern:
        //   x_i8 → DQ(s_x=0.1) → x_f32 ─┐
        //                                ├─→ Conv → y_f32 → Q(s_y=0.05) → y_i8
        //   w_i8 → DQ(s_w=0.2) → w_f32 ─┘
        let mut shapes: HashMap<String, TensorShape> = HashMap::new();
        shapes.insert("x_i8".into(), shape_i8(vec![1, 4, 8, 8]));
        shapes.insert("x_f32".into(), shape_f32(vec![1, 4, 8, 8]));
        shapes.insert("w_i8".into(), shape_i8(vec![8, 4, 3, 3]));
        shapes.insert("w_f32".into(), shape_f32(vec![8, 4, 3, 3]));
        shapes.insert("y_f32".into(), shape_f32(vec![1, 8, 6, 6]));
        shapes.insert("y_i8".into(), shape_i8(vec![1, 8, 6, 6]));

        let mut initializers: HashMap<String, Vec<u8>> = HashMap::new();
        // Pre-quantized weight: 8*4*3*3 = 288 i8s.
        initializers.insert("w_i8".into(), vec![1u8; 288]);

        let ops = vec![
            dq_op("dq_x", "x_i8", "x_f32", 0.1),
            dq_op("dq_w", "w_i8", "w_f32", 0.2),
            conv_op("conv", "x_f32", "w_f32", "y_f32", [3, 3]),
            q_op("q_y", "y_f32", "y_i8", 0.05),
        ];

        Graph {
            ops,
            shapes,
            inputs: vec!["x_i8".into()],
            outputs: vec!["y_i8".into()],
            initializers,
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn fold_qdq_simple_conv() {
        let mut graph = build_graph_with_qdq_conv();
        assert!(graph_has_qdq(&graph));
        let stats = fold_qdq_patterns(&mut graph).expect("fold");

        assert_eq!(stats.conv_folded, 1);
        assert_eq!(stats.dequantize_removed, 2);
        assert_eq!(stats.quantize_removed, 1);
        // Weight was already I8 so no on-the-fly quantize.
        assert_eq!(stats.weights_quantized, 0);
        assert_eq!(stats.qdq_unfolded, 0);

        // Resulting graph: Conv → Requantize.
        assert_eq!(graph.ops.len(), 2);
        assert_eq!(graph.ops[0].op_type, "Conv");
        assert_eq!(graph.ops[0].inputs, vec!["x_i8", "w_i8"]);
        assert_eq!(graph.ops[1].op_type, "Requantize");
        // Requant scale = (0.1 * 0.2) / 0.05 = 0.4.
        match graph.ops[1].params {
            OpParams::Requantize { scale } => {
                assert!((scale - 0.4).abs() < 1e-6, "got requant scale {scale}");
            }
            _ => panic!("expected Requantize"),
        }
        // Final output name preserved.
        assert_eq!(graph.ops[1].outputs, vec!["y_i8"]);
    }

    #[test]
    fn fold_qdq_skips_dq_with_multiple_consumers() {
        // x_i8 → DQ → x_f32, consumed by both a Conv (which would fold)
        //          and a Relu (which keeps x_f32 alive). Fold should bail.
        let mut graph = build_graph_with_qdq_conv();
        graph.ops.push(CompiledOp {
            name: "relu".into(),
            op_type: "Relu".into(),
            inputs: vec!["x_f32".into()],
            outputs: vec!["relu_out".into()],
            params: OpParams::None,
        });
        graph
            .shapes
            .insert("relu_out".into(), shape_f32(vec![1, 4, 8, 8]));

        let stats = fold_qdq_patterns(&mut graph).expect("fold");
        // x_f32 now has 2 consumers; the conv fold must bail.
        assert_eq!(stats.conv_folded, 0);
        assert!(stats.qdq_unfolded > 0);
    }

    #[test]
    fn fold_qdq_quantizes_f32_weight() {
        // Same as simple_conv but the weight initializer is F32.
        let mut graph = build_graph_with_qdq_conv();
        // Replace w_i8 with an F32 weight initializer named w_f32_init
        // and rewire the weight DQ.
        graph.shapes.remove("w_i8");
        graph.initializers.remove("w_i8");
        let f32_w: Vec<f32> = (0..288).map(|i| (i as f32) * 0.01).collect();
        let f32_bytes: Vec<u8> = f32_w.iter().flat_map(|f| f.to_le_bytes()).collect();
        graph.initializers.insert("w_f32_init".into(), f32_bytes);
        graph
            .shapes
            .insert("w_f32_init".into(), shape_f32(vec![8, 4, 3, 3]));
        // Find the weight DQ and rewire its input.
        for op in &mut graph.ops {
            if op.name == "dq_w" {
                op.inputs[0] = "w_f32_init".into();
            }
        }

        let stats = fold_qdq_patterns(&mut graph).expect("fold");
        assert_eq!(stats.conv_folded, 1);
        assert_eq!(stats.weights_quantized, 1);

        // Weight initializer must now be 288 bytes (i8) under the new name.
        let w = graph.initializers.get("w_f32_init").expect("weight kept");
        assert_eq!(w.len(), 288);
        // The conv references the same name.
        let conv = graph.ops.iter().find(|o| o.op_type == "Conv").unwrap();
        assert_eq!(conv.inputs[1], "w_f32_init");
        let w_shape = graph.shapes.get("w_f32_init").unwrap();
        assert_eq!(w_shape.dtype, Dtype::I8);
    }

    #[test]
    fn fold_qdq_no_qdq_is_noop() {
        // No QDQ wrappers → fold is a no-op.
        let mut shapes: HashMap<String, TensorShape> = HashMap::new();
        shapes.insert("x".into(), shape_f32(vec![1, 4]));
        shapes.insert("y".into(), shape_f32(vec![1, 4]));
        let mut graph = Graph {
            ops: vec![CompiledOp {
                name: "relu".into(),
                op_type: "Relu".into(),
                inputs: vec!["x".into()],
                outputs: vec!["y".into()],
                params: OpParams::None,
            }],
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };
        assert!(!graph_has_qdq(&graph));
        let stats = fold_qdq_patterns(&mut graph).expect("fold");
        assert_eq!(stats.conv_folded, 0);
        assert_eq!(graph.ops.len(), 1);
        assert_eq!(graph.ops[0].op_type, "Relu");
    }

    #[test]
    fn fold_qdq_chained_convs() {
        // Two convs in series, each wrapped in QDQ. The output of conv1's Q
        // feeds conv2's input DQ — i.e. the i8 buffer between them is
        // shared. After folding, both convs should be I8 and the
        // intermediate Q/DQ pair must disappear.
        //
        //   x_i8 → DQ(s_x) → x_f32 ─┐
        //                            Conv1 → y1_f32 → Q(s_y1) → y1_i8
        //   w1_i8 → DQ(s_w1) → w1_f32 ┘
        //
        //   y1_i8 → DQ(s_y1) → y1_f32_again ─┐
        //                                     Conv2 → y2_f32 → Q(s_y2) → y2_i8
        //   w2_i8 → DQ(s_w2) → w2_f32 ────────┘
        let mut shapes: HashMap<String, TensorShape> = HashMap::new();
        for name in ["x_i8", "y1_i8", "y2_i8"] {
            shapes.insert(name.into(), shape_i8(vec![1, 4, 8, 8]));
        }
        for name in ["x_f32", "y1_f32", "y1_f32_again", "y2_f32"] {
            shapes.insert(name.into(), shape_f32(vec![1, 4, 8, 8]));
        }
        for name in ["w1_i8", "w2_i8"] {
            shapes.insert(name.into(), shape_i8(vec![4, 4, 3, 3]));
        }
        for name in ["w1_f32", "w2_f32"] {
            shapes.insert(name.into(), shape_f32(vec![4, 4, 3, 3]));
        }

        let mut initializers: HashMap<String, Vec<u8>> = HashMap::new();
        initializers.insert("w1_i8".into(), vec![1u8; 4 * 4 * 3 * 3]);
        initializers.insert("w2_i8".into(), vec![2u8; 4 * 4 * 3 * 3]);

        let ops = vec![
            dq_op("dq_x", "x_i8", "x_f32", 0.1),
            dq_op("dq_w1", "w1_i8", "w1_f32", 0.2),
            conv_op("conv1", "x_f32", "w1_f32", "y1_f32", [3, 3]),
            q_op("q_y1", "y1_f32", "y1_i8", 0.05),
            dq_op("dq_y1", "y1_i8", "y1_f32_again", 0.05),
            dq_op("dq_w2", "w2_i8", "w2_f32", 0.3),
            conv_op("conv2", "y1_f32_again", "w2_f32", "y2_f32", [3, 3]),
            q_op("q_y2", "y2_f32", "y2_i8", 0.04),
        ];

        let mut graph = Graph {
            ops,
            shapes,
            inputs: vec!["x_i8".into()],
            outputs: vec!["y2_i8".into()],
            initializers,
            dtype: Dtype::F32,
        };

        let stats = fold_qdq_patterns(&mut graph).expect("fold");
        assert_eq!(stats.conv_folded, 2);
        // 4 weight/activation DQs removed (2 per conv = activation + weight).
        assert_eq!(stats.dequantize_removed, 4);
        // Both Q ops removed (y1's Q and y2's Q).
        assert_eq!(stats.quantize_removed, 2);
        assert_eq!(stats.qdq_unfolded, 0);

        // Resulting graph: Conv, Requant, Conv, Requant (4 ops).
        assert_eq!(graph.ops.len(), 4);
        let types: Vec<&str> = graph.ops.iter().map(|o| o.op_type.as_str()).collect();
        assert_eq!(types, vec!["Conv", "Requantize", "Conv", "Requantize"]);

        // Conv2 now consumes the i8 output of the first requantize.
        let conv2 = &graph.ops[2];
        assert_eq!(conv2.inputs[0], "y1_i8");
        assert_eq!(conv2.inputs[1], "w2_i8");
    }

    #[test]
    fn fold_qdq_then_fusion_yields_fused_int8_conv() {
        // Verify that the QDQ-fold output is the exact shape expected by
        // `apply_fusion_passes` Pass 3 (Conv + Requantize → fused INT8 conv).
        use crate::fusion::apply_fusion_passes;

        let mut graph = build_graph_with_qdq_conv();
        fold_qdq_patterns(&mut graph).expect("fold");
        // Pre-fusion: 2 ops (Conv, Requantize).
        assert_eq!(graph.ops.len(), 2);

        apply_fusion_passes(&mut graph);

        // Post-fusion: 1 op (Conv2dRequantReluI8).
        assert_eq!(
            graph.ops.len(),
            1,
            "expected fusion to collapse Conv+Requant"
        );
        assert!(matches!(
            graph.ops[0].params,
            OpParams::Conv2dRequantReluI8Nhwc { .. }
        ));
        // Final output name preserved through both passes.
        assert_eq!(graph.ops[0].outputs, vec!["y_i8"]);
    }

    #[test]
    fn fold_qdq_prunes_unused_scale_initializers() {
        // Add a phantom F32 initializer that no surviving op references.
        let mut graph = build_graph_with_qdq_conv();
        graph
            .initializers
            .insert("orphan_scale".into(), vec![0u8; 4]);
        graph
            .shapes
            .insert("orphan_scale".into(), shape_f32(vec![]));

        let _ = fold_qdq_patterns(&mut graph).expect("fold");
        assert!(
            !graph.initializers.contains_key("orphan_scale"),
            "orphan initializer should be pruned"
        );
    }
}
