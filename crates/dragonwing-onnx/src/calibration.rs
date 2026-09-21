//! Post-training quantization calibration.
//!
//! This module provides calibration tools to collect activation statistics
//! and compute optimal quantization scales for INT8 inference.
//!
//! # Calibration Process
//!
//! 1. Load the F32 model
//! 2. Run inference on a representative calibration dataset
//! 3. Collect min/max statistics for each activation tensor
//! 4. Compute per-tensor scales for activations
//! 5. Compute per-channel scales for weights
//!
//! # Example
//!
//! ```ignore
//! let graph = compile_graph(&model, Dtype::F32)?;
//! let calibrator = Calibrator::new(graph)?;
//!
//! // Run calibration on representative images
//! for image in calibration_images {
//!     calibrator.feed(&image)?;
//! }
//!
//! // Get quantization parameters
//! let params = calibrator.compute_params()?;
//! ```

use crate::builder::{CompiledOp, OpParams, TensorShape};
use crate::error::{Error, Result};
use crate::graph::Graph;
use dragonwing_core::{Backend, Dtype, PerChannelScale, QuantScale, QuantizationParams};
use dragonwing_cpu::CpuBuffer;
use std::collections::HashMap;

/// Statistics collected during calibration.
#[derive(Debug, Clone)]
struct TensorStats {
    /// Minimum value observed.
    min: f32,
    /// Maximum value observed.
    max: f32,
    /// Number of samples collected.
    count: usize,
}

impl TensorStats {
    fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            count: 0,
        }
    }

    fn update(&mut self, values: &[f32]) {
        for &v in values {
            if v.is_finite() {
                self.min = self.min.min(v);
                self.max = self.max.max(v);
            }
        }
        self.count += 1;
    }

    /// Compute quantization scale from collected statistics.
    fn compute_scale(&self) -> QuantScale {
        QuantScale::from_range(self.min, self.max)
    }
}

/// Calibration strategy for computing scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationStrategy {
    /// Use min/max range directly.
    /// Simple but may be sensitive to outliers.
    MinMax,
    /// Use percentile (e.g., 99.99%) instead of absolute min/max.
    /// More robust to outliers.
    Percentile { percentile: u32 }, // percentile * 100 (e.g., 9999 for 99.99%)
}

impl Default for CalibrationStrategy {
    fn default() -> Self {
        Self::MinMax
    }
}

/// Calibrator for collecting activation statistics.
pub struct Calibrator {
    /// The compiled F32 graph.
    graph: Graph,
    /// Allocated buffers.
    buffers: HashMap<String, CpuBuffer>,
    /// Statistics for each activation tensor.
    stats: HashMap<String, TensorStats>,
    /// Names of tensors to calibrate (activations, not weights).
    activation_names: Vec<String>,
    /// Calibration strategy.
    strategy: CalibrationStrategy,
    /// Number of samples fed so far.
    sample_count: usize,
}

impl Calibrator {
    /// Create a new calibrator from a compiled F32 graph.
    ///
    /// The graph must be compiled with F32 dtype.
    pub fn new(graph: Graph) -> Result<Self> {
        if graph.dtype != Dtype::F32 {
            return Err(Error::Validation("Calibration requires F32 graph".into()));
        }

        let backend = dragonwing_cpu::CpuBackend::new();
        let mut buffers = HashMap::new();

        // Allocate buffers
        for (name, shape) in &graph.shapes {
            let size_bytes = shape.size_bytes();
            let buffer = backend
                .alloc(size_bytes, dragonwing_core::BufferKind::Storage)
                .map_err(|e| Error::Runtime(format!("failed to allocate {name}: {e}")))?;
            buffers.insert(name.clone(), buffer);
        }

        // Upload weights (initializers)
        for (name, data) in &graph.initializers {
            if let Some(buffer) = buffers.get_mut(name) {
                let buf_bytes: &mut [u8] = buffer.as_bytes_mut();
                if buf_bytes.len() == data.len() {
                    buf_bytes.copy_from_slice(data);
                }
            }
        }

        // Determine which tensors are activations (not initializers)
        let activation_names: Vec<String> = graph
            .shapes
            .keys()
            .filter(|name| !graph.initializers.contains_key(*name))
            .cloned()
            .collect();

        let stats = HashMap::new();

        Ok(Self {
            graph,
            buffers,
            stats,
            activation_names,
            strategy: CalibrationStrategy::default(),
            sample_count: 0,
        })
    }

    /// Set the calibration strategy.
    pub fn with_strategy(mut self, strategy: CalibrationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Get the input tensor names and shapes.
    pub fn inputs(&self) -> Vec<(&str, &TensorShape)> {
        self.graph.input_info()
    }

    /// Feed a calibration sample (input tensor data).
    ///
    /// This runs F32 inference and collects activation statistics.
    pub fn feed(&mut self, input_name: &str, data: &[f32]) -> Result<()> {
        // Set input
        let input_buf = self
            .buffers
            .get_mut(input_name)
            .ok_or_else(|| Error::Runtime(format!("input not found: {input_name}")))?;
        let buf_f32 = input_buf.as_f32_mut();
        if buf_f32.len() != data.len() {
            return Err(Error::Runtime(format!(
                "input size mismatch: expected {}, got {}",
                buf_f32.len(),
                data.len()
            )));
        }
        buf_f32.copy_from_slice(data);

        // Run ops and collect statistics
        let ops: Vec<_> = self.graph.ops.clone();
        for op in &ops {
            self.dispatch_op(op)?;

            // After each op, collect stats for output tensors
            for output_name in &op.outputs {
                if self.activation_names.contains(output_name) {
                    self.collect_stats(output_name)?;
                }
            }
        }

        self.sample_count += 1;
        Ok(())
    }

    /// Collect statistics for a tensor.
    fn collect_stats(&mut self, name: &str) -> Result<()> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;

        let data = buffer.as_f32();

        let stats = self
            .stats
            .entry(name.to_string())
            .or_insert_with(TensorStats::new);
        stats.update(data);

        Ok(())
    }

    /// Compute quantization parameters from collected statistics.
    pub fn compute_params(&self) -> Result<QuantizationParams> {
        if self.sample_count == 0 {
            return Err(Error::Validation("No calibration samples fed".into()));
        }

        let mut params = QuantizationParams::new();

        // Compute activation scales
        for (name, stats) in &self.stats {
            let scale = stats.compute_scale();
            params.add_activation_scale(name.clone(), scale);
        }

        // Compute weight scales (per-channel)
        for (name, data) in &self.graph.initializers {
            if let Some(shape) = self.graph.shapes.get(name) {
                // Determine if this is a weight tensor (typically 4D for conv, 2D for FC)
                if shape.dims.len() >= 2 {
                    let weights: &[f32] =
                        unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len() / 4) };

                    // First dimension is output channels
                    let num_channels = shape.dims[0];
                    let per_channel = PerChannelScale::from_weights(weights, num_channels);
                    params.add_weight_scales(name.clone(), per_channel);
                }
            }
        }

        // Set input scale from first input's stats
        for input_name in &self.graph.inputs {
            if let Some(stats) = self.stats.get(input_name) {
                params.set_input_scale(stats.compute_scale());
                break;
            }
        }

        Ok(params)
    }

    /// Get the number of samples fed.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Get statistics for a specific tensor.
    pub fn get_stats(&self, name: &str) -> Option<(f32, f32, usize)> {
        self.stats.get(name).map(|s| (s.min, s.max, s.count))
    }

    // =========================================================================
    // Op dispatch (simplified for calibration - F32 only)
    // =========================================================================

    fn dispatch_op(&mut self, op: &CompiledOp) -> Result<()> {
        match &op.params {
            OpParams::None => match op.op_type.as_str() {
                "Relu" => self.dispatch_relu(op),
                "Flatten" | "Reshape" => self.dispatch_reshape(op),
                "MatMul" => self.dispatch_matmul(op),
                _ => Ok(()),
            },
            OpParams::Clip { min, max } => self.dispatch_clip(op, *min, *max),
            OpParams::Add => self.dispatch_add(op),
            OpParams::GlobalAvgPool => self.dispatch_global_avg_pool(op),
            OpParams::Gemm {
                trans_a, trans_b, ..
            } => self.dispatch_gemm(op, *trans_a, *trans_b),
            OpParams::Softmax { .. } => self.dispatch_softmax(op),
            OpParams::Reshape { .. } => self.dispatch_reshape(op),
            OpParams::Conv2d {
                kernel_shape,
                strides,
                pads,
                group,
                ..
            } => self.dispatch_conv2d(op, *kernel_shape, *strides, *pads, *group),
            OpParams::MaxPool {
                kernel_shape,
                strides,
                ..
            } => self.dispatch_maxpool(op, *kernel_shape, *strides),
            OpParams::AvgPool {
                kernel_shape,
                strides,
                ..
            } => self.dispatch_avgpool(op, *kernel_shape, *strides),
            OpParams::Sigmoid => self.dispatch_sigmoid(op),
            OpParams::Mul => self.dispatch_mul(op),
            OpParams::Sub => self.dispatch_sub(op),
            OpParams::Div => self.dispatch_div(op),
            OpParams::Concat { axis } => self.dispatch_concat(op, *axis),
            OpParams::Resize { out_h, out_w, mode } => {
                self.dispatch_resize(op, *out_h, *out_w, mode)
            }
            OpParams::Split { axis, split_sizes } => self.dispatch_split(op, *axis, split_sizes),
            OpParams::Transpose { perm } => self.dispatch_transpose(op, perm),
            OpParams::Slice {
                starts,
                ends,
                axes,
                steps,
            } => self.dispatch_slice(op, starts, ends, axes, steps),
            // Quantized ops are not used during F32 calibration
            OpParams::Requantize { .. }
            | OpParams::AddQuantized { .. }
            | OpParams::Quantize { .. }
            | OpParams::Dequantize { .. }
            | OpParams::QuantizePerChannel { .. }
            | OpParams::DequantizePerChannel { .. }
            | OpParams::Conv2dRequantReluI8Nhwc { .. } => Ok(()),
        }
    }

    fn dispatch_relu(&mut self, op: &CompiledOp) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        if input_name == output_name {
            let buffer = self
                .buffers
                .get_mut(input_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
            let data: &mut [f32] = buffer.as_f32_mut();
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        } else {
            // Copy then relu
            let input_data: Vec<f32> = self
                .buffers
                .get(input_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
                .as_f32()
                .to_vec();

            let output: &mut [f32] = self
                .buffers
                .get_mut(output_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
                .as_f32_mut();

            for (o, &i) in output.iter_mut().zip(input_data.iter()) {
                *o = f32::max(i, 0.0);
            }
        }
        Ok(())
    }

    fn dispatch_clip(&mut self, op: &CompiledOp, min: f32, max: f32) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_data: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output: &mut [f32] = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        for (o, &i) in output.iter_mut().zip(input_data.iter()) {
            *o = f32::clamp(i, min, max);
        }
        Ok(())
    }

    fn dispatch_add(&mut self, op: &CompiledOp) -> Result<()> {
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let a: Vec<f32> = self
            .buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?
            .as_f32()
            .to_vec();
        let b: Vec<f32> = self
            .buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?
            .as_f32_mut();

        if a.len() == b.len() {
            for (o, (&ai, &bi)) in output.iter_mut().zip(a.iter().zip(b.iter())) {
                *o = ai + bi;
            }
        } else if b.len() == 1 {
            let bv = b[0];
            for (o, &ai) in output.iter_mut().zip(a.iter()) {
                *o = ai + bv;
            }
        } else {
            // Broadcast
            for (i, (o, &ai)) in output.iter_mut().zip(a.iter()).enumerate() {
                *o = ai + b[i % b.len()];
            }
        }
        Ok(())
    }

    fn dispatch_global_avg_pool(&mut self, op: &CompiledOp) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self
            .graph
            .shapes
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

        // NCHW format
        let n = input_shape.dims[0];
        let c = input_shape.dims[1];
        let h = input_shape.dims[2];
        let w = input_shape.dims[3];
        let spatial = h * w;

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        for batch in 0..n {
            for ch in 0..c {
                let mut sum = 0.0f32;
                for s in 0..spatial {
                    sum += input[batch * c * spatial + ch * spatial + s];
                }
                output[batch * c + ch] = sum / spatial as f32;
            }
        }
        Ok(())
    }

    fn dispatch_gemm(&mut self, op: &CompiledOp, trans_a: bool, trans_b: bool) -> Result<()> {
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let a_shape = self
            .graph
            .shapes
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
        let b_shape = self
            .graph
            .shapes
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {b_name}")))?;

        let m = if trans_a {
            a_shape.dims[1]
        } else {
            a_shape.dims[0]
        };
        let k = if trans_a {
            a_shape.dims[0]
        } else {
            a_shape.dims[1]
        };
        let n = if trans_b {
            b_shape.dims[0]
        } else {
            b_shape.dims[1]
        };

        let a: Vec<f32> = self
            .buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?
            .as_f32()
            .to_vec();
        let b: Vec<f32> = self
            .buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?
            .as_f32()
            .to_vec();

        // Get bias first before getting mutable output
        let bias: Vec<f32> = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            self.buffers
                .get(&op.inputs[2])
                .map(|b: &CpuBuffer| b.as_f32().to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let output: &mut [f32] = self
            .buffers
            .get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?
            .as_f32_mut();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_idx = if trans_a { l * m + i } else { i * k + l };
                    let b_idx = if trans_b { j * k + l } else { l * n + j };
                    sum += a[a_idx] * b[b_idx];
                }
                output[i * n + j] = sum;
            }
        }

        // Add bias
        if !bias.is_empty() {
            for i in 0..m {
                for j in 0..n {
                    output[i * n + j] += bias[j % bias.len()];
                }
            }
        }
        Ok(())
    }

    fn dispatch_matmul(&mut self, op: &CompiledOp) -> Result<()> {
        self.dispatch_gemm(op, false, false)
    }

    fn dispatch_softmax(&mut self, op: &CompiledOp) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self
            .graph
            .shapes
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

        let last_dim = *input_shape.dims.last().unwrap_or(&1);
        let rows = input_shape.numel() / last_dim;

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        for row in 0..rows {
            let start = row * last_dim;
            let end = start + last_dim;
            let slice = &input[start..end];

            let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for v in slice {
                sum += (*v - max_val).exp();
            }

            for (i, v) in slice.iter().enumerate() {
                output[start + i] = (*v - max_val).exp() / sum;
            }
        }
        Ok(())
    }

    fn dispatch_reshape(&mut self, op: &CompiledOp) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        if input_name != output_name {
            let input: Vec<f32> = self
                .buffers
                .get(input_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
                .as_f32()
                .to_vec();

            let output = self
                .buffers
                .get_mut(output_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
                .as_f32_mut();

            let copy_len = input.len().min(output.len());
            output[..copy_len].copy_from_slice(&input[..copy_len]);
        }
        Ok(())
    }

    fn dispatch_conv2d(
        &mut self,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        group: usize,
    ) -> Result<()> {
        let input_name = &op.inputs[0];
        let weight_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let input_shape = self
            .graph
            .shapes
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
        let output_shape = self
            .graph
            .shapes
            .get(out_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {out_name}")))?;
        let weight_shape = self
            .graph
            .shapes
            .get(weight_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {weight_name}")))?;

        // NCHW format
        let n = input_shape.dims[0];
        let c_in = input_shape.dims[1];
        let h_in = input_shape.dims[2];
        let w_in = input_shape.dims[3];
        let c_out = weight_shape.dims[0];
        let k_h = kernel_shape[0];
        let k_w = kernel_shape[1];
        let h_out = output_shape.dims[2];
        let w_out = output_shape.dims[3];

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();
        let weights: Vec<f32> = self
            .buffers
            .get(weight_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {weight_name}")))?
            .as_f32()
            .to_vec();

        // Get bias before getting mutable output
        let bias: Vec<f32> = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            self.buffers
                .get(&op.inputs[2])
                .map(|b: &CpuBuffer| b.as_f32().to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let output: &mut [f32] = self
            .buffers
            .get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?
            .as_f32_mut();

        let c_in_per_group = c_in / group;
        let c_out_per_group = c_out / group;

        for batch in 0..n {
            for g in 0..group {
                for oc in 0..c_out_per_group {
                    let oc_global = g * c_out_per_group + oc;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut acc = 0.0f32;
                            for ic in 0..c_in_per_group {
                                let ic_global = g * c_in_per_group + ic;
                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let ih = (oh * strides[0] + kh) as isize - pads[0] as isize;
                                        let iw = (ow * strides[1] + kw) as isize - pads[1] as isize;
                                        if ih >= 0
                                            && ih < h_in as isize
                                            && iw >= 0
                                            && iw < w_in as isize
                                        {
                                            let ih = ih as usize;
                                            let iw = iw as usize;
                                            let in_idx = batch * c_in * h_in * w_in
                                                + ic_global * h_in * w_in
                                                + ih * w_in
                                                + iw;
                                            let w_idx = oc_global * (c_in_per_group * k_h * k_w)
                                                + ic * k_h * k_w
                                                + kh * k_w
                                                + kw;
                                            acc += input[in_idx] * weights[w_idx];
                                        }
                                    }
                                }
                            }
                            let out_idx = batch * c_out * h_out * w_out
                                + oc_global * h_out * w_out
                                + oh * w_out
                                + ow;
                            output[out_idx] = acc;
                        }
                    }
                }
            }
        }

        // Add bias
        if !bias.is_empty() {
            for batch in 0..n {
                for oc in 0..c_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let idx = batch * c_out * h_out * w_out
                                + oc * h_out * w_out
                                + oh * w_out
                                + ow;
                            output[idx] += bias[oc];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn dispatch_maxpool(
        &mut self,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
    ) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self
            .graph
            .shapes
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
        let output_shape = self
            .graph
            .shapes
            .get(output_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {output_name}")))?;

        let n = input_shape.dims[0];
        let c = input_shape.dims[1];
        let h_in = input_shape.dims[2];
        let w_in = input_shape.dims[3];
        let h_out = output_shape.dims[2];
        let w_out = output_shape.dims[3];

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        for kh in 0..kernel_shape[0] {
                            for kw in 0..kernel_shape[1] {
                                let ih = oh * strides[0] + kh;
                                let iw = ow * strides[1] + kw;
                                if ih < h_in && iw < w_in {
                                    let idx =
                                        batch * c * h_in * w_in + ch * h_in * w_in + ih * w_in + iw;
                                    max_val = max_val.max(input[idx]);
                                }
                            }
                        }
                        let out_idx =
                            batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }
        Ok(())
    }

    fn dispatch_avgpool(
        &mut self,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
    ) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self
            .graph
            .shapes
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
        let output_shape = self
            .graph
            .shapes
            .get(output_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {output_name}")))?;

        let n = input_shape.dims[0];
        let c = input_shape.dims[1];
        let h_in = input_shape.dims[2];
        let w_in = input_shape.dims[3];
        let h_out = output_shape.dims[2];
        let w_out = output_shape.dims[3];

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        let pool_size = (kernel_shape[0] * kernel_shape[1]) as f32;

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for kh in 0..kernel_shape[0] {
                            for kw in 0..kernel_shape[1] {
                                let ih = oh * strides[0] + kh;
                                let iw = ow * strides[1] + kw;
                                if ih < h_in && iw < w_in {
                                    let idx =
                                        batch * c * h_in * w_in + ch * h_in * w_in + ih * w_in + iw;
                                    sum += input[idx];
                                }
                            }
                        }
                        let out_idx =
                            batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = sum / pool_size;
                    }
                }
            }
        }
        Ok(())
    }

    fn dispatch_sigmoid(&mut self, op: &CompiledOp) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = 1.0 / (1.0 + (-i).exp());
        }
        Ok(())
    }

    fn dispatch_mul(&mut self, op: &CompiledOp) -> Result<()> {
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let a: Vec<f32> = self
            .buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?
            .as_f32()
            .to_vec();
        let b: Vec<f32> = self
            .buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?
            .as_f32_mut();

        if a.len() == b.len() {
            for (o, (&ai, &bi)) in output.iter_mut().zip(a.iter().zip(b.iter())) {
                *o = ai * bi;
            }
        } else {
            for (i, (o, &ai)) in output.iter_mut().zip(a.iter()).enumerate() {
                *o = ai * b[i % b.len()];
            }
        }
        Ok(())
    }

    fn dispatch_sub(&mut self, op: &CompiledOp) -> Result<()> {
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let a: Vec<f32> = self
            .buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?
            .as_f32()
            .to_vec();
        let b: Vec<f32> = self
            .buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?
            .as_f32_mut();

        if a.len() == b.len() {
            for (o, (&ai, &bi)) in output.iter_mut().zip(a.iter().zip(b.iter())) {
                *o = ai - bi;
            }
        } else {
            for (i, (o, &ai)) in output.iter_mut().zip(a.iter()).enumerate() {
                *o = ai - b[i % b.len()];
            }
        }
        Ok(())
    }

    fn dispatch_div(&mut self, op: &CompiledOp) -> Result<()> {
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let a: Vec<f32> = self
            .buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?
            .as_f32()
            .to_vec();
        let b: Vec<f32> = self
            .buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?
            .as_f32_mut();

        if a.len() == b.len() {
            for (o, (&ai, &bi)) in output.iter_mut().zip(a.iter().zip(b.iter())) {
                *o = ai / bi;
            }
        } else {
            for (i, (o, &ai)) in output.iter_mut().zip(a.iter()).enumerate() {
                *o = ai / b[i % b.len()];
            }
        }
        Ok(())
    }

    fn dispatch_concat(&mut self, op: &CompiledOp, axis: usize) -> Result<()> {
        // For simplicity, only handle axis=0 or axis=-1 (last dim)
        let out_name = &op.outputs[0];
        let out_shape = self
            .graph
            .shapes
            .get(out_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {out_name}")))?;

        let mut all_inputs: Vec<Vec<f32>> = Vec::new();
        for input_name in &op.inputs {
            let data: Vec<f32> = self
                .buffers
                .get(input_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
                .as_f32()
                .to_vec();
            all_inputs.push(data);
        }

        let output = self
            .buffers
            .get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?
            .as_f32_mut();

        // Simple concatenation along first axis
        if axis == 0 {
            let mut offset = 0;
            for inp in &all_inputs {
                output[offset..offset + inp.len()].copy_from_slice(inp);
                offset += inp.len();
            }
        } else {
            // Generic concat - more complex
            // For now, just copy sequentially
            let mut offset = 0;
            for inp in &all_inputs {
                let copy_len = inp.len().min(output.len() - offset);
                output[offset..offset + copy_len].copy_from_slice(&inp[..copy_len]);
                offset += copy_len;
            }
        }
        let _ = out_shape; // silence unused warning
        Ok(())
    }

    fn dispatch_resize(
        &mut self,
        op: &CompiledOp,
        out_h: usize,
        out_w: usize,
        mode: &str,
    ) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self
            .graph
            .shapes
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

        let n = input_shape.dims[0];
        let c = input_shape.dims[1];
        let h_in = input_shape.dims[2];
        let w_in = input_shape.dims[3];

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        let scale_h = h_in as f32 / out_h as f32;
        let scale_w = w_in as f32 / out_w as f32;

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = if mode == "nearest" {
                            ((oh as f32 + 0.5) * scale_h).floor() as usize
                        } else {
                            (oh as f32 * scale_h) as usize
                        }
                        .min(h_in - 1);

                        let iw = if mode == "nearest" {
                            ((ow as f32 + 0.5) * scale_w).floor() as usize
                        } else {
                            (ow as f32 * scale_w) as usize
                        }
                        .min(w_in - 1);

                        let in_idx = batch * c * h_in * w_in + ch * h_in * w_in + ih * w_in + iw;
                        let out_idx =
                            batch * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = input[in_idx];
                    }
                }
            }
        }
        Ok(())
    }

    fn dispatch_split(
        &mut self,
        op: &CompiledOp,
        axis: usize,
        split_sizes: &[usize],
    ) -> Result<()> {
        let input_name = &op.inputs[0];

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        // Simple split along first axis
        if axis == 0 {
            let mut offset = 0;
            for (i, &size) in split_sizes.iter().enumerate() {
                if i < op.outputs.len() {
                    let output = self
                        .buffers
                        .get_mut(&op.outputs[i])
                        .ok_or_else(|| {
                            Error::Runtime(format!("buffer not found: {}", op.outputs[i]))
                        })?
                        .as_f32_mut();
                    let copy_len = size.min(input.len() - offset).min(output.len());
                    output[..copy_len].copy_from_slice(&input[offset..offset + copy_len]);
                    offset += size;
                }
            }
        }
        Ok(())
    }

    fn dispatch_transpose(&mut self, op: &CompiledOp, perm: &[usize]) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self
            .graph
            .shapes
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        // Compute strides for input shape
        let ndim = input_shape.dims.len();
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input_shape.dims[i + 1];
        }

        // Compute output shape and strides
        let out_dims: Vec<usize> = perm.iter().map(|&p| input_shape.dims[p]).collect();
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }

        let total = input_shape.numel();
        for i in 0..total {
            // Convert linear index to multi-dim index in output
            let mut out_idx = vec![0usize; ndim];
            let mut remainder = i;
            for d in 0..ndim {
                out_idx[d] = remainder / out_strides[d];
                remainder %= out_strides[d];
            }

            // Map to input index
            let mut in_linear = 0;
            for d in 0..ndim {
                in_linear += out_idx[d] * in_strides[perm[d]];
            }

            output[i] = input[in_linear];
        }
        Ok(())
    }

    fn dispatch_slice(
        &mut self,
        op: &CompiledOp,
        starts: &[isize],
        ends: &[isize],
        axes: &[usize],
        steps: &[isize],
    ) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input: Vec<f32> = self
            .buffers
            .get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?
            .as_f32()
            .to_vec();

        let output = self
            .buffers
            .get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?
            .as_f32_mut();

        // Simple 1D slice for now
        if axes.len() == 1 && axes[0] == 0 {
            let start = starts[0].max(0) as usize;
            let end = (ends[0] as usize).min(input.len());
            let step = steps[0].max(1) as usize;

            let mut out_idx = 0;
            let mut in_idx = start;
            while in_idx < end && out_idx < output.len() {
                output[out_idx] = input[in_idx];
                out_idx += 1;
                in_idx += step;
            }
        } else {
            // Copy as-is for complex cases
            let copy_len = input.len().min(output.len());
            output[..copy_len].copy_from_slice(&input[..copy_len]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_stats_basic() {
        let mut stats = TensorStats::new();
        assert_eq!(stats.min, f32::INFINITY);
        assert_eq!(stats.max, f32::NEG_INFINITY);

        stats.update(&[-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(stats.min, -1.0);
        assert_eq!(stats.max, 2.0);
        assert_eq!(stats.count, 1);

        stats.update(&[-3.0, 5.0]);
        assert_eq!(stats.min, -3.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.count, 2);
    }

    #[test]
    fn tensor_stats_compute_scale() {
        let mut stats = TensorStats::new();
        stats.update(&[-1.27, 0.5, 1.0]);

        let scale = stats.compute_scale();
        // max(|min|, |max|) = 1.27, scale = 1.27 / 127 = 0.01
        assert!((scale.scale - 0.01).abs() < 1e-6);
    }
}
