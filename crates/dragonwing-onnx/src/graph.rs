//! Graph compilation and execution.
//!
//! This module handles:
//! 1. Two-phase compilation (validate + compile)
//! 2. Graph execution with proper tensor lifetime management

use crate::error::{Error, Result};
use crate::model::{Model, DataType};
use crate::builder::{
    BuildContext, TensorShape, CompiledOp,
    ValidationReport, UnsupportedReason, get_builder,
};
use dragonwing_core::Dtype;
use std::collections::HashMap;

/// A compiled graph ready for execution.
#[derive(Debug)]
pub struct Graph {
    /// Compiled ops in topological order.
    pub ops: Vec<CompiledOp>,
    /// Tensor shapes (name → shape).
    pub shapes: HashMap<String, TensorShape>,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Initializer data (name → raw bytes).
    pub initializers: HashMap<String, Vec<u8>>,
    /// The dtype the graph was compiled with.
    pub dtype: Dtype,
}

impl Graph {
    /// Get the input tensor names and shapes.
    pub fn input_info(&self) -> Vec<(&str, &TensorShape)> {
        self.inputs.iter()
            .filter_map(|name| self.shapes.get(name).map(|s| (name.as_str(), s)))
            .collect()
    }

    /// Get the output tensor names and shapes.
    pub fn output_info(&self) -> Vec<(&str, &TensorShape)> {
        self.outputs.iter()
            .filter_map(|name| self.shapes.get(name).map(|s| (name.as_str(), s)))
            .collect()
    }

    /// Get the total number of ops.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }
}

/// Validate a model without compiling it.
///
/// Returns a report of supported and unsupported ops.
pub fn validate_model(model: &Model, dtype: Dtype) -> ValidationReport {
    let initializers: HashMap<String, _> = model.initializers.iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();
    
    let mut ctx = BuildContext::new(dtype, &initializers);
    let mut report = ValidationReport::default();

    // Register input shapes
    for input in &model.inputs {
        // Skip inputs that are actually initializers
        if initializers.contains_key(&input.name) {
            continue;
        }
        
        let shape = TensorShape::new(
            input.shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect(),
            dtype,
        );
        ctx.set_shape(input.name.clone(), shape);
    }

    // Register initializer shapes
    for init in &model.initializers {
        let shape = TensorShape::new(
            init.dims.iter().map(|&d| d as usize).collect(),
            onnx_dtype_to_dragonwing(init.data_type, dtype),
        );
        ctx.set_shape(init.name.clone(), shape);
    }

    // Validate each node
    // We call build() to populate constants for constant-folding ops,
    // but only report validation results
    for node in &model.nodes {
        let builder = match get_builder(&node.op_type) {
            Some(b) => b,
            None => {
                report.unsupported.push((
                    node.name.clone(),
                    UnsupportedReason::UnknownOpType,
                ));
                continue;
            }
        };

        match builder.is_supported(node, &ctx) {
            Ok(()) => {}
            Err(reason) => {
                report.unsupported.push((node.name.clone(), reason));
                continue;
            }
        }

        // Call build() to populate constants and shapes
        match builder.build(node, &mut ctx) {
            Ok(_compiled) => {
                // Op validated and built successfully
                report.supported.push(node.name.clone());
            }
            Err(e) => {
                report.unsupported.push((
                    node.name.clone(),
                    UnsupportedReason::Other(e.to_string()),
                ));
            }
        }
    }

    report
}

/// Compile a model into an executable graph.
///
/// This is the two-phase compilation:
/// 1. Validate all ops and compute shapes
/// 2. Build the compiled ops
pub fn compile_model(model: &Model, dtype: Dtype) -> Result<Graph> {
    // First, validate
    let report = validate_model(model, dtype);
    if !report.is_fully_supported() {
        let errors: Vec<String> = report.unsupported.iter()
            .map(|(name, reason)| format!("{name}: {reason}"))
            .collect();
        return Err(Error::Validation(format!(
            "Model has unsupported ops:\n  {}",
            errors.join("\n  ")
        )));
    }

    // Now compile
    let initializers: HashMap<String, _> = model.initializers.iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();
    
    let mut ctx = BuildContext::new(dtype, &initializers);
    let mut ops = Vec::new();

    // Register input shapes
    let mut graph_inputs = Vec::new();
    for input in &model.inputs {
        if initializers.contains_key(&input.name) {
            continue;
        }
        
        let shape = TensorShape::new(
            input.shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect(),
            dtype,
        );
        ctx.set_shape(input.name.clone(), shape);
        graph_inputs.push(input.name.clone());
    }

    // Register initializer shapes
    for init in &model.initializers {
        let shape = TensorShape::new(
            init.dims.iter().map(|&d| d as usize).collect(),
            onnx_dtype_to_dragonwing(init.data_type, dtype),
        );
        ctx.set_shape(init.name.clone(), shape);
    }

    // Build each node
    for node in &model.nodes {
        let builder = get_builder(&node.op_type)
            .ok_or_else(|| Error::Compile(format!("no builder for {}", node.op_type)))?;
        
        let compiled = builder.build(node, &mut ctx)?;
        
        // Skip pure metadata ops (Reshape with no actual work)
        // Actually, keep them for now - the runtime will handle them
        ops.push(compiled);
    }

    // Collect output names
    let graph_outputs: Vec<String> = model.outputs.iter()
        .map(|o| o.name.clone())
        .collect();

    // Collect all tensor names actually used by ops
    let mut used_tensors: std::collections::HashSet<String> = std::collections::HashSet::new();
    for op in &ops {
        for input in &op.inputs {
            used_tensors.insert(input.clone());
        }
        for output in &op.outputs {
            used_tensors.insert(output.clone());
        }
    }
    // Also include graph outputs
    for output in &graph_outputs {
        used_tensors.insert(output.clone());
    }

    // Collect initializer data only for tensors that are actually used
    // and that are F32 or F16 (skip INT64 shape tensors that are constant-folded)
    let initializer_data: HashMap<String, Vec<u8>> = model.initializers.iter()
        .filter(|t| {
            // Only include if it's actually used by an op
            if !used_tensors.contains(&t.name) {
                return false;
            }
            // Skip INT64 initializers - they're shape tensors for constant folding
            if t.data_type == DataType::Int64 {
                return false;
            }
            true
        })
        .map(|t| (t.name.clone(), t.data.clone()))
        .collect();

    Ok(Graph {
        ops,
        shapes: ctx.shapes,
        inputs: graph_inputs,
        outputs: graph_outputs,
        initializers: initializer_data,
        dtype,
    })
}

/// Convert a compiled graph from NCHW to NHWC layout.
///
/// ONNX models typically use NCHW format, but dragonwing kernels use NHWC
/// for efficient memory access on the target hardware.
///
/// The conversion:
/// - Transposes 4D activation shapes from [N,C,H,W] to [N,H,W,C]
/// - Transposes Conv weight data from [C_out,C_in,kH,kW] to [C_out,kH,kW,C_in]
/// - The caller must transpose input data NCHW→NHWC before feeding to the graph
/// - The caller must transpose output data NHWC→NCHW after getting results
pub fn convert_nchw_to_nhwc(graph: &mut Graph) -> Result<()> {
    // First, identify which tensors are Conv weights (they have 4D shapes and are initializers)
    let mut conv_weight_names: Vec<String> = Vec::new();
    for op in &graph.ops {
        if op.op_type == "Conv" && !op.inputs.is_empty() && op.inputs.len() >= 2 {
            // inputs[1] is the weight tensor
            let weight_name = &op.inputs[1];
            if graph.initializers.contains_key(weight_name) {
                conv_weight_names.push(weight_name.clone());
            }
        }
    }
    
    // Transpose Conv weight initializers: [C_out, C_in, kH, kW] → [kH, kW, C_in, C_out]
    for weight_name in &conv_weight_names {
        // Snapshot dtype + dims so we don't hold a borrow into graph.shapes.
        let (dtype, orig_shape) = match graph.shapes.get(weight_name) {
            Some(s) if s.dims.len() == 4 => (s.dtype, s.dims.clone()),
            _ => continue,
        };

        if let Some(data) = graph.initializers.get_mut(weight_name) {
            // perm [2, 3, 1, 0] yields [kH, kW, C_in, C_out] from OIHW.
            *data = match dtype {
                Dtype::F32 => transpose_f32_4d(data, &orig_shape, &[2, 3, 1, 0]),
                Dtype::I8 => transpose_i8_4d(data, &orig_shape, &[2, 3, 1, 0]),
                Dtype::F16 => transpose_f16_4d(data, &orig_shape, &[2, 3, 1, 0]),
                other => {
                    // Unknown weight dtype — leave data alone but log via
                    // shape mismatch. Caller's responsibility to know.
                    return Err(Error::Compile(format!(
                        "convert_nchw_to_nhwc: unsupported weight dtype {other:?} for {weight_name}"
                    )));
                }
            };
        }

        if let Some(s) = graph.shapes.get_mut(weight_name) {
            s.dims = vec![orig_shape[2], orig_shape[3], orig_shape[1], orig_shape[0]];
        }
    }
    
    // Convert all non-weight 4D activation shapes from NCHW to NHWC
    let weight_set: std::collections::HashSet<_> = conv_weight_names.iter().collect();
    for (name, shape) in graph.shapes.iter_mut() {
        if shape.dims.len() == 4 && !weight_set.contains(name) {
            let [n, c, h, w] = [shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]];
            shape.dims = vec![n, h, w, c];
        }
    }
    
    Ok(())
}

/// Transpose a 4D F32 tensor.
/// 
/// `perm` specifies the permutation: for output index [i,j,k,l],
/// read from input at [perm[0]->i, perm[1]->j, ...]
fn transpose_f32_4d(data: &[u8], shape: &[usize], perm: &[usize; 4]) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    if data.len() != numel * 4 {
        return data.to_vec();  // Size mismatch, return as-is
    }
    
    let src: Vec<f32> = data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    
    let [d0, d1, d2, d3] = [shape[0], shape[1], shape[2], shape[3]];
    let out_shape = [shape[perm[0]], shape[perm[1]], shape[perm[2]], shape[perm[3]]];
    
    let mut dst = vec![0.0f32; numel];
    
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    // Source index in original layout
                    let src_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    
                    // Destination indices after permutation
                    let indices = [i0, i1, i2, i3];
                    let mut out_idx = [0usize; 4];
                    for (out_dim, &src_dim) in perm.iter().enumerate() {
                        out_idx[out_dim] = indices[src_dim];
                    }
                    
                    let dst_idx = ((out_idx[0] * out_shape[1] + out_idx[1]) * out_shape[2] + out_idx[2]) * out_shape[3] + out_idx[3];
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    
    dst.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Transpose a 4D INT8 tensor.
///
/// Same semantics as [`transpose_f32_4d`] but for 1-byte/element tensors.
/// Used to reorder INT8 weight initializers from OIHW to HWIO during the
/// NCHW→NHWC conversion (Task 009 — needed once QDQ-folded INT8 weights
/// flow through the layout pass).
fn transpose_i8_4d(data: &[u8], shape: &[usize], perm: &[usize; 4]) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    if data.len() != numel {
        return data.to_vec();
    }

    let [d0, d1, d2, d3] = [shape[0], shape[1], shape[2], shape[3]];
    let out_shape = [shape[perm[0]], shape[perm[1]], shape[perm[2]], shape[perm[3]]];
    let mut dst = vec![0u8; numel];

    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    let src_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    let indices = [i0, i1, i2, i3];
                    let mut out_idx = [0usize; 4];
                    for (out_dim, &src_dim) in perm.iter().enumerate() {
                        out_idx[out_dim] = indices[src_dim];
                    }
                    let dst_idx = ((out_idx[0] * out_shape[1] + out_idx[1]) * out_shape[2]
                        + out_idx[2])
                        * out_shape[3]
                        + out_idx[3];
                    dst[dst_idx] = data[src_idx];
                }
            }
        }
    }

    dst
}

/// Transpose a 4D F16 tensor (2 bytes per element).
fn transpose_f16_4d(data: &[u8], shape: &[usize], perm: &[usize; 4]) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    if data.len() != numel * 2 {
        return data.to_vec();
    }

    let [d0, d1, d2, d3] = [shape[0], shape[1], shape[2], shape[3]];
    let out_shape = [shape[perm[0]], shape[perm[1]], shape[perm[2]], shape[perm[3]]];
    let mut dst = vec![0u8; numel * 2];

    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    let src_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    let indices = [i0, i1, i2, i3];
                    let mut out_idx = [0usize; 4];
                    for (out_dim, &src_dim) in perm.iter().enumerate() {
                        out_idx[out_dim] = indices[src_dim];
                    }
                    let dst_idx = ((out_idx[0] * out_shape[1] + out_idx[1]) * out_shape[2]
                        + out_idx[2])
                        * out_shape[3]
                        + out_idx[3];
                    dst[dst_idx * 2] = data[src_idx * 2];
                    dst[dst_idx * 2 + 1] = data[src_idx * 2 + 1];
                }
            }
        }
    }

    dst
}

/// Transpose input data from NCHW to NHWC format.
/// 
/// Call this on input data before feeding to a graph that was converted
/// with `convert_nchw_to_nhwc`.
pub fn transpose_nchw_to_nhwc(data: &[f32], shape: &[usize; 4]) -> Vec<f32> {
    let [n, c, h, w] = *shape;
    let mut out = vec![0.0f32; n * h * w * c];
    
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src_idx = ((ni * c + ci) * h + hi) * w + wi;
                    let dst_idx = ((ni * h + hi) * w + wi) * c + ci;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    }
    
    out
}

/// Transpose output data from NHWC back to NCHW format.
/// 
/// Call this on output data after getting results from a graph that was
/// converted with `convert_nchw_to_nhwc`.
pub fn transpose_nhwc_to_nchw(data: &[f32], shape: &[usize; 4]) -> Vec<f32> {
    let [n, h, w, c] = *shape;  // NHWC shape
    let mut out = vec![0.0f32; n * c * h * w];
    
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let src_idx = ((ni * h + hi) * w + wi) * c + ci;
                    let dst_idx = ((ni * c + ci) * h + hi) * w + wi;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    }
    
    out
}

/// Pre-transpose Gemm weights so that `trans_b` becomes `false` at runtime.
///
/// ONNX `Gemm` ops typically have `transB=1` (weights stored as `[N, K]`,
/// activations as `[M, K]`, output `C = A · Bᵀ`). The on-device Vulkan
/// `gemm_f32` shader only supports the non-transposed `C = A · B` form
/// (B stored as `[K, N]`). Rather than maintain a separate transposed
/// shader, we transpose the weight initializer once at compile time:
///
/// * For each Gemm op where `trans_b == true` and `inputs[1]` is an
///   initializer, transpose the F32/F16 weight bytes from `[N, K]` to
///   `[K, N]`.
/// * Update the weight tensor's shape entry accordingly.
/// * Flip `OpParams::Gemm::trans_b` to `false` so the runtime dispatches
///   the standard non-transposed shader.
///
/// `trans_a` is left untouched (rare in practice and out of scope for
/// task 008).
///
/// # Errors
///
/// Returns an error if a Gemm op declares `trans_b=true` but its weight
/// tensor is not an initializer (i.e. dynamic), since we can't transpose
/// runtime tensors at compile time.
pub fn fold_gemm_transpose(graph: &mut Graph) -> Result<()> {
    // Collect rewrites in a first pass to avoid borrowing graph.ops mutably
    // while inspecting initializers.
    let mut rewrites: Vec<(usize, String, usize, usize, Dtype)> = Vec::new();
    for (op_idx, op) in graph.ops.iter().enumerate() {
        if op.op_type != "Gemm" {
            continue;
        }
        let trans_b = match &op.params {
            crate::builder::OpParams::Gemm { trans_b, .. } => *trans_b,
            _ => continue,
        };
        if !trans_b {
            continue;
        }
        if op.inputs.len() < 2 {
            continue;
        }
        let weight_name = op.inputs[1].clone();
        if !graph.initializers.contains_key(&weight_name) {
            return Err(Error::Compile(format!(
                "Gemm op `{}` declares trans_b=true but its weight `{weight_name}` \
                 is not a constant initializer; runtime transposition is not \
                 supported. Pre-transpose the weight or rewrite the model.",
                op.name
            )));
        }
        // Pre-transpose semantics: weight is currently [N, K]; rewrite to [K, N].
        let shape = graph
            .shapes
            .get(&weight_name)
            .ok_or_else(|| Error::Compile(format!("shape not found for {weight_name}")))?;
        if shape.dims.len() != 2 {
            return Err(Error::Compile(format!(
                "Gemm weight {weight_name} must be 2D, got {:?}",
                shape.dims
            )));
        }
        let n_rows = shape.dims[0]; // N
        let k_cols = shape.dims[1]; // K
        rewrites.push((op_idx, weight_name, n_rows, k_cols, shape.dtype));
    }

    // Apply rewrites: transpose bytes, update shape, set trans_b=false.
    for (op_idx, weight_name, n_rows, k_cols, dtype) in rewrites {
        let data = graph
            .initializers
            .get_mut(&weight_name)
            .ok_or_else(|| Error::Compile(format!("initializer {weight_name} vanished")))?;
        match dtype {
            Dtype::F32 => {
                let element_size = 4;
                if data.len() != n_rows * k_cols * element_size {
                    return Err(Error::Compile(format!(
                        "{weight_name}: size {} != {n_rows}*{k_cols}*4",
                        data.len()
                    )));
                }
                *data = transpose_2d_bytes(data, n_rows, k_cols, element_size);
            }
            Dtype::F16 => {
                let element_size = 2;
                if data.len() != n_rows * k_cols * element_size {
                    return Err(Error::Compile(format!(
                        "{weight_name}: size {} != {n_rows}*{k_cols}*2",
                        data.len()
                    )));
                }
                *data = transpose_2d_bytes(data, n_rows, k_cols, element_size);
            }
            other => {
                return Err(Error::Compile(format!(
                    "Gemm weight {weight_name} has unsupported dtype {other:?} \
                     for compile-time transpose"
                )));
            }
        }
        // Swap shape dims: [N, K] -> [K, N].
        if let Some(shape) = graph.shapes.get_mut(&weight_name) {
            shape.dims = vec![k_cols, n_rows];
        }
        // Flip trans_b on the op.
        if let crate::builder::OpParams::Gemm { trans_b, .. } = &mut graph.ops[op_idx].params {
            *trans_b = false;
        }
    }
    Ok(())
}

/// Transpose a 2D byte buffer where each element is `element_size` bytes,
/// from row-major `[rows, cols]` to row-major `[cols, rows]`.
fn transpose_2d_bytes(data: &[u8], rows: usize, cols: usize, element_size: usize) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * element_size;
            let dst = (c * rows + r) * element_size;
            out[dst..dst + element_size].copy_from_slice(&data[src..src + element_size]);
        }
    }
    out
}

/// Convert ONNX DataType to dragonwing Dtype.
fn onnx_dtype_to_dragonwing(onnx_dtype: DataType, default: Dtype) -> Dtype {
    match onnx_dtype {
        DataType::Float => Dtype::F32,
        DataType::Float16 => Dtype::F16,
        // Task 009 — INT8 initializers (pre-quantized weights from
        // ONNX Runtime QDQ export) keep their I8 dtype so the QDQ-fold
        // pass can recognise them without trying to re-quantize.
        DataType::Int8 | DataType::Uint8 => Dtype::I8,
        // For weights stored as other types, use the graph's default dtype
        _ => default,
    }
}

/// Fold BatchNormalization into preceding Conv.
///
/// This modifies the model in place, removing BN nodes and adjusting
/// Conv weights/biases.
pub fn fold_batchnorm(model: &mut Model) -> Result<()> {
    let initializers: HashMap<String, _> = model.initializers.iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();

    // Find Conv -> BN patterns
    let mut bn_to_remove = Vec::new();
    let mut conv_updates: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();

    for (i, node) in model.nodes.iter().enumerate() {
        if node.op_type != "BatchNormalization" {
            continue;
        }

        // Check if input comes from a Conv
        let input_name = &node.inputs[0];
        let conv_idx = model.nodes.iter()
            .position(|n| n.outputs.contains(input_name) && n.op_type == "Conv");
        
        let Some(conv_idx) = conv_idx else {
            // BN not after Conv - this is an error in task 004 scope
            return Err(Error::Compile(format!(
                "BatchNorm {} not preceded by Conv - cannot fold", node.name
            )));
        };

        let conv_node = &model.nodes[conv_idx];

        // Get BN parameters: scale, bias, mean, var
        // BN inputs: X, scale, B, mean, var
        if node.inputs.len() < 5 {
            return Err(Error::Compile(format!(
                "BatchNorm {} has insufficient inputs", node.name
            )));
        }

        let scale = initializers.get(&node.inputs[1])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN scale not found: {}", node.inputs[1])))?;
        let bn_bias = initializers.get(&node.inputs[2])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN bias not found: {}", node.inputs[2])))?;
        let mean = initializers.get(&node.inputs[3])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN mean not found: {}", node.inputs[3])))?;
        let var = initializers.get(&node.inputs[4])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN var not found: {}", node.inputs[4])))?;

        let epsilon = node.get_attr_float("epsilon", 1e-5);

        // Get Conv weight
        let conv_weight = initializers.get(&conv_node.inputs[1])
            .ok_or_else(|| Error::Compile(format!("Conv weight not found: {}", conv_node.inputs[1])))?;
        let conv_weight_data = conv_weight.as_f32_slice()
            .ok_or_else(|| Error::Compile("Conv weight must be f32".into()))?;

        // Get Conv bias (optional)
        let conv_bias = if conv_node.inputs.len() > 2 && !conv_node.inputs[2].is_empty() {
            initializers.get(&conv_node.inputs[2])
                .and_then(|t| t.as_f32_slice())
                .map(|s| s.to_vec())
        } else {
            None
        };

        // Fold BN into Conv
        // weight_new[c] = weight[c] * scale[c] / sqrt(var[c] + eps)
        // bias_new[c] = (bias[c] - mean[c]) * scale[c] / sqrt(var[c] + eps) + bn_bias[c]
        
        let c_out = conv_weight.dims[0] as usize;
        let weight_per_channel = conv_weight_data.len() / c_out;
        
        let mut new_weight = Vec::with_capacity(conv_weight_data.len());
        let mut new_bias = Vec::with_capacity(c_out);

        for c in 0..c_out {
            // Use f64 for intermediate calculations to avoid precision loss
            let std_inv = 1.0 / ((var[c] as f64 + epsilon as f64).sqrt());
            let scale_factor = scale[c] as f64 * std_inv;
            
            // Scale weights for this output channel
            let start = c * weight_per_channel;
            let end = start + weight_per_channel;
            for &w in &conv_weight_data[start..end] {
                new_weight.push((w as f64 * scale_factor) as f32);
            }
            
            // Compute new bias
            let old_bias = conv_bias.as_ref().map(|b| b[c]).unwrap_or(0.0) as f64;
            let new_b = (old_bias - mean[c] as f64) * scale_factor + bn_bias[c] as f64;
            new_bias.push(new_b as f32);
        }

        conv_updates.insert(conv_node.inputs[1].clone(), (new_weight, new_bias));
        bn_to_remove.push(i);
    }

    // Apply updates
    for (weight_name, (new_weight, new_bias)) in conv_updates {
        // Update weight initializer
        if let Some(init) = model.initializers.iter_mut().find(|t| t.name == weight_name) {
            let mut bytes = Vec::with_capacity(new_weight.len() * 4);
            for w in &new_weight {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            init.data = bytes;
        }

        // Find the conv node and its bias name
        for node in &mut model.nodes {
            if node.op_type == "Conv" && node.inputs.len() > 1 && node.inputs[1] == weight_name {
                // Add or update bias
                let bias_name = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                    node.inputs[2].clone()
                } else {
                    let name = format!("{}_bias_folded", weight_name);
                    if node.inputs.len() > 2 {
                        node.inputs[2] = name.clone();
                    } else {
                        node.inputs.push(name.clone());
                    }
                    name
                };

                // Update or add bias initializer
                let bias_dims = vec![new_bias.len() as i64];
                let mut bias_bytes = Vec::with_capacity(new_bias.len() * 4);
                for b in &new_bias {
                    bias_bytes.extend_from_slice(&b.to_le_bytes());
                }

                if let Some(init) = model.initializers.iter_mut().find(|t| t.name == bias_name) {
                    init.data = bias_bytes;
                } else {
                    model.initializers.push(crate::model::OnnxTensor {
                        name: bias_name,
                        dims: bias_dims,
                        data_type: DataType::Float,
                        data: bias_bytes,
                    });
                }
                break;
            }
        }
    }

    // Remove BN nodes (in reverse order to preserve indices)
    for &idx in bn_to_remove.iter().rev() {
        let bn_node = &model.nodes[idx];
        let bn_output = bn_node.outputs[0].clone();
        let bn_input = bn_node.inputs[0].clone();

        // Redirect all references from BN output to Conv output (which is BN input)
        for node in &mut model.nodes {
            for input in &mut node.inputs {
                if *input == bn_output {
                    *input = bn_input.clone();
                }
            }
        }

        model.nodes.remove(idx);
    }

    // Update outputs if they referenced BN outputs
    // (handled by the redirection above)

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::new(vec![1, 3, 224, 224], Dtype::F32);
        assert_eq!(shape.numel(), 1 * 3 * 224 * 224);
        assert_eq!(shape.size_bytes(), 1 * 3 * 224 * 224 * 4);
    }

    #[test]
    fn test_transpose_2d_bytes_f32() {
        // 2 rows × 3 cols, F32 (4 bytes per element)
        // Row-major [2,3]: [a,b,c, d,e,f] => transposed [3,2]: [a,d, b,e, c,f]
        let a: f32 = 1.0;
        let b: f32 = 2.0;
        let c: f32 = 3.0;
        let d: f32 = 4.0;
        let e: f32 = 5.0;
        let f: f32 = 6.0;
        let mut input = Vec::new();
        for v in &[a, b, c, d, e, f] {
            input.extend_from_slice(&v.to_le_bytes());
        }
        let out = transpose_2d_bytes(&input, 2, 3, 4);
        // Expected order: a, d, b, e, c, f
        let expected = [a, d, b, e, c, f];
        for i in 0..6 {
            let bytes = &out[i * 4..i * 4 + 4];
            let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            assert!((val - expected[i]).abs() < 1e-6, "elem {i}: {val} vs {}", expected[i]);
        }
    }

    #[test]
    fn test_transpose_i8_4d_roundtrip() {
        // OIHW shape [C_out=2, C_in=2, kH=2, kW=2]; perm [2,3,1,0] → HWIO
        // [kH=2, kW=2, C_in=2, C_out=2].
        let shape = [2usize, 2, 2, 2];
        let numel: usize = shape.iter().product();
        let data: Vec<u8> = (0..numel as u8).collect();
        let out = transpose_i8_4d(&data, &shape, &[2, 3, 1, 0]);
        assert_eq!(out.len(), numel);

        // Spot check: source [oc=0, ic=0, h=0, w=0] = 0 should land at
        // dst[h=0, w=0, ic=0, oc=0] = 0 (corner). Source [oc=1, ic=0, h=0,
        // w=0] = 8 should land at dst[h=0, w=0, ic=0, oc=1] = 1.
        // dst layout strides: w stride = 2 (oc), ic stride = 4 (w*oc=2*2),
        // h stride = 8 (kW*ic*oc=2*2*2). So dst[0,0,0,0]=0, dst[0,0,0,1]=1.
        assert_eq!(out[0], 0, "[0,0,0,0]");
        assert_eq!(out[1], 8, "[0,0,0,1] should be 8 (was oc=1,ic=0,h=0,w=0)");
    }

    #[test]
    fn test_convert_nchw_to_nhwc_handles_int8_weights() {
        use crate::builder::{CompiledOp, OpParams};

        // 1 Conv op with an I8 weight [C_out=2, C_in=2, kH=2, kW=2].
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![1, 2, 4, 4], Dtype::F32));
        shapes.insert("w".into(), TensorShape::new(vec![2, 2, 2, 2], Dtype::I8));
        shapes.insert("y".into(), TensorShape::new(vec![1, 2, 3, 3], Dtype::F32));

        let mut initializers = HashMap::new();
        // 16 bytes of I8 weight data, distinguishable per element.
        initializers.insert("w".into(), (0..16u8).collect());

        let mut graph = Graph {
            ops: vec![CompiledOp {
                name: "conv".into(),
                op_type: "Conv".into(),
                inputs: vec!["x".into(), "w".into()],
                outputs: vec!["y".into()],
                params: OpParams::Conv2d {
                    kernel_shape: [2, 2],
                    strides: [1, 1],
                    pads: [0, 0, 0, 0],
                    dilations: [1, 1],
                    group: 1,
                },
            }],
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            initializers,
            dtype: Dtype::F32,
        };

        convert_nchw_to_nhwc(&mut graph).expect("nhwc convert");

        let w_shape = graph.shapes.get("w").unwrap();
        // Shape flips OIHW [2,2,2,2] → HWIO [2,2,2,2] (square in this case
        // but dimensions are reordered conceptually).
        assert_eq!(w_shape.dims, vec![2, 2, 2, 2]);
        assert_eq!(w_shape.dtype, Dtype::I8);

        // Data must be the i8 transpose; original element at OIHW
        // [oc=1, ic=0, h=0, w=0] (= byte 8) should now sit at HWIO
        // [h=0, w=0, ic=0, oc=1] = index 1.
        let w_data = graph.initializers.get("w").unwrap();
        assert_eq!(w_data.len(), 16);
        assert_eq!(w_data[1], 8);
    }

    #[test]
    fn test_fold_gemm_transpose_flips_flag_and_data() {
        use crate::builder::OpParams;

        // Build a minimal graph with one Gemm op (trans_b=true).
        // A is [M=2, K=3], B is stored as [N=4, K=3] (trans_b semantics),
        // expected post-fold: B becomes [K=3, N=4] with trans_b=false.
        let mut shapes = HashMap::new();
        shapes.insert("a".into(), TensorShape::new(vec![2, 3], Dtype::F32));
        shapes.insert("b".into(), TensorShape::new(vec![4, 3], Dtype::F32));
        shapes.insert("c".into(), TensorShape::new(vec![2, 4], Dtype::F32));

        // B = row-major [4 rows, 3 cols]:
        // row 0: 1, 2, 3
        // row 1: 4, 5, 6
        // row 2: 7, 8, 9
        // row 3: 10, 11, 12
        let b_f32: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let b_bytes: Vec<u8> = b_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut initializers = HashMap::new();
        initializers.insert("b".to_string(), b_bytes.clone());

        let mut graph = Graph {
            ops: vec![crate::builder::CompiledOp {
                name: "gemm1".into(),
                op_type: "Gemm".into(),
                inputs: vec!["a".into(), "b".into()],
                outputs: vec!["c".into()],
                params: OpParams::Gemm {
                    alpha: 1.0,
                    beta: 1.0,
                    trans_a: false,
                    trans_b: true,
                },
            }],
            shapes,
            inputs: vec!["a".into()],
            outputs: vec!["c".into()],
            initializers,
            dtype: Dtype::F32,
        };

        fold_gemm_transpose(&mut graph).expect("fold");

        // trans_b should now be false.
        match &graph.ops[0].params {
            OpParams::Gemm { trans_b, .. } => assert!(!*trans_b, "trans_b should be false"),
            _ => panic!("not a Gemm op"),
        }
        // Shape of B should now be [K, N] = [3, 4].
        let b_shape = graph.shapes.get("b").unwrap();
        assert_eq!(b_shape.dims, vec![3, 4]);
        // Data should be transposed: row-major [3, 4]:
        // row 0: 1, 4, 7, 10
        // row 1: 2, 5, 8, 11
        // row 2: 3, 6, 9, 12
        let new_bytes = graph.initializers.get("b").unwrap();
        let new_f32: Vec<f32> = new_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let expected = vec![1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0];
        for (i, (got, exp)) in new_f32.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "elem {i}: {got} vs {exp}");
        }
    }

    #[test]
    fn test_fold_gemm_transpose_skips_when_trans_b_false() {
        use crate::builder::OpParams;

        let mut shapes = HashMap::new();
        shapes.insert("b".into(), TensorShape::new(vec![3, 4], Dtype::F32));
        let mut initializers = HashMap::new();
        let b_bytes: Vec<u8> = (1..=12)
            .flat_map(|i: i32| (i as f32).to_le_bytes())
            .collect();
        initializers.insert("b".into(), b_bytes.clone());

        let mut graph = Graph {
            ops: vec![crate::builder::CompiledOp {
                name: "gemm1".into(),
                op_type: "Gemm".into(),
                inputs: vec!["a".into(), "b".into()],
                outputs: vec!["c".into()],
                params: OpParams::Gemm {
                    alpha: 1.0,
                    beta: 1.0,
                    trans_a: false,
                    trans_b: false,
                },
            }],
            shapes,
            inputs: vec!["a".into()],
            outputs: vec!["c".into()],
            initializers,
            dtype: Dtype::F32,
        };

        fold_gemm_transpose(&mut graph).expect("fold");

        // Data should be unchanged.
        assert_eq!(graph.initializers.get("b").unwrap(), &b_bytes);
    }

    #[test]
    fn test_fold_gemm_transpose_errors_on_dynamic_weight() {
        use crate::builder::OpParams;

        let mut shapes = HashMap::new();
        shapes.insert("b".into(), TensorShape::new(vec![4, 3], Dtype::F32));
        // No initializer for "b" — simulates a dynamic weight.

        let mut graph = Graph {
            ops: vec![crate::builder::CompiledOp {
                name: "gemm1".into(),
                op_type: "Gemm".into(),
                inputs: vec!["a".into(), "b".into()],
                outputs: vec!["c".into()],
                params: OpParams::Gemm {
                    alpha: 1.0,
                    beta: 1.0,
                    trans_a: false,
                    trans_b: true,
                },
            }],
            shapes,
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };

        let result = fold_gemm_transpose(&mut graph);
        assert!(result.is_err(), "expected error for dynamic gemm weight");
    }

    #[test]
    #[ignore] // Requires artifacts/models/mobilenetv2-12.onnx
    fn test_validate_mobilenetv2() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.onnx");
        let bytes = std::fs::read(model_path).expect("Failed to read model file");
        let model = crate::model::parse_model(&bytes).expect("Failed to parse model");
        
        let report = validate_model(&model, Dtype::F32);
        
        println!("Validation report:");
        println!("  Supported: {} ops", report.supported.len());
        println!("  Unsupported: {} ops", report.unsupported.len());
        
        if !report.unsupported.is_empty() {
            println!("\n  Unsupported ops:");
            for (name, reason) in &report.unsupported {
                println!("    {}: {}", name, reason);
            }
        }
        
        assert!(report.is_fully_supported(), 
            "Model validation failed with {} unsupported ops", 
            report.unsupported.len());
    }

    #[test]
    #[ignore] // Requires artifacts/models/mobilenetv2-12.onnx  
    fn test_debug_mobilenetv2_shapes() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.onnx");
        let bytes = std::fs::read(model_path).expect("Failed to read model file");
        let model = crate::model::parse_model(&bytes).expect("Failed to parse model");
        
        let initializers: HashMap<String, _> = model.initializers.iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();
        
        let mut ctx = BuildContext::new(Dtype::F32, &initializers);
        
        // Register input shapes
        for input in &model.inputs {
            if initializers.contains_key(&input.name) {
                continue;
            }
            let shape = TensorShape::new(
                input.shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect(),
                Dtype::F32,
            );
            ctx.set_shape(input.name.clone(), shape);
        }
        
        // Register initializer shapes
        for init in &model.initializers {
            let shape = TensorShape::new(
                init.dims.iter().map(|&d| d as usize).collect(),
                onnx_dtype_to_dragonwing(init.data_type, Dtype::F32),
            );
            ctx.set_shape(init.name.clone(), shape);
        }
        
        // Build nodes and print shapes for last few ops
        for node in &model.nodes {
            let builder = match crate::builder::get_builder(&node.op_type) {
                Some(b) => b,
                None => continue,
            };
            
            // Print shapes for last few nodes
            if node.name.contains("Reshape") || node.name.contains("Gemm") || 
               node.name.contains("Concat") || node.name.contains("GlobalAverage") {
                println!("\n{} ({}):", node.name, node.op_type);
                for inp in &node.inputs {
                    if let Some(shape) = ctx.get_shape(inp) {
                        println!("  input {}: {:?}", inp, shape.dims);
                    } else {
                        println!("  input {}: NOT FOUND", inp);
                    }
                }
            }
            
            if builder.is_supported(node, &ctx).is_ok() {
                if let Ok(_) = builder.build(node, &mut ctx) {
                    if node.name.contains("Reshape") || node.name.contains("Gemm") ||
                       node.name.contains("Concat") || node.name.contains("GlobalAverage") {
                        for out in &node.outputs {
                            if let Some(shape) = ctx.get_shape(out) {
                                println!("  output {}: {:?}", out, shape.dims);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore] // Requires artifacts/models/mobilenetv2-12.onnx
    fn test_compile_mobilenetv2() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.onnx");
        let bytes = std::fs::read(model_path).expect("Failed to read model file");
        let model = crate::model::parse_model(&bytes).expect("Failed to parse model");
        
        let graph = compile_model(&model, Dtype::F32).expect("Failed to compile model");
        
        println!("Compiled graph:");
        println!("  Ops: {}", graph.num_ops());
        println!("  Inputs: {:?}", graph.input_info());
        println!("  Outputs: {:?}", graph.output_info());
        
        // Check expected shapes
        let inputs = graph.input_info();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].0, "input");
        assert_eq!(inputs[0].1.dims, vec![1, 3, 224, 224]);
        
        let outputs = graph.output_info();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "output");
        assert_eq!(outputs[0].1.dims, vec![1, 1000]);
    }

    #[test]
    #[ignore] // Requires artifacts/models/mobilenetv2-12.onnx
    fn test_compile_mobilenetv2_nhwc() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.onnx");
        let bytes = std::fs::read(model_path).expect("Failed to read model file");
        let model = crate::model::parse_model(&bytes).expect("Failed to parse model");
        
        let mut graph = compile_model(&model, Dtype::F32).expect("Failed to compile model");
        
        // Convert to NHWC
        convert_nchw_to_nhwc(&mut graph).expect("Failed to convert to NHWC");
        
        println!("Compiled graph (NHWC):");
        println!("  Ops: {}", graph.num_ops());
        println!("  Inputs: {:?}", graph.input_info());
        println!("  Outputs: {:?}", graph.output_info());
        
        // Check converted shapes
        let inputs = graph.input_info();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].0, "input");
        // Input should now be NHWC: [1, 224, 224, 3]
        assert_eq!(inputs[0].1.dims, vec![1, 224, 224, 3]);
        
        // Output is 2D, should be unchanged
        let outputs = graph.output_info();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "output");
        assert_eq!(outputs[0].1.dims, vec![1, 1000]);
    }

    #[test]
    fn test_transpose_nchw_nhwc() {
        // Test small tensor: 1x2x2x3 in NCHW
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let shape = [1, 2, 2, 3];  // [N=1, C=2, H=2, W=3]
        
        let transposed = transpose_nchw_to_nhwc(&data, &shape);
        
        // Expected: [N=1, H=2, W=3, C=2]
        // In NCHW: data[n,c,h,w] at index n*C*H*W + c*H*W + h*W + w
        // In NHWC: data[n,h,w,c] at index n*H*W*C + h*W*C + w*C + c
        
        // NCHW index (0,0,0,0) -> 0  => NHWC index (0,0,0,0) -> 0
        // NCHW index (0,0,0,1) -> 1  => NHWC index (0,0,1,0) -> 2  
        // NCHW index (0,1,0,0) -> 6  => NHWC index (0,0,0,1) -> 1
        
        // At NHWC [0,0,0,:] we should have NCHW [0,:,0,0] = [0, 6]
        assert_eq!(transposed[0], 0.0);  // NHWC [0,0,0,0]
        assert_eq!(transposed[1], 6.0);  // NHWC [0,0,0,1]
        
        // At NHWC [0,0,1,:] we should have NCHW [0,:,0,1] = [1, 7]
        assert_eq!(transposed[2], 1.0);  // NHWC [0,0,1,0]
        assert_eq!(transposed[3], 7.0);  // NHWC [0,0,1,1]
    }
}
