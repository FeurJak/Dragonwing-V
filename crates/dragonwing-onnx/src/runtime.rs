//! Graph execution runtime.
//!
//! This module provides the runtime that executes compiled ONNX graphs
//! on a backend. It handles:
//! - Buffer allocation for activations and weights
//! - Op dispatch to the backend
//! - Input/output tensor management

use crate::error::{Error, Result};
use crate::graph::Graph;
use crate::builder::{OpParams, TensorShape, CompiledOp};
use dragonwing_core::{Backend, BackendBuffer, BufferKind, Dtype};
use std::collections::HashMap;

/// Runtime state for executing a graph on a specific backend.
pub struct GraphRuntime<B: Backend> {
    /// The compiled graph.
    graph: Graph,
    /// Backend instance.
    backend: B,
    /// Allocated buffers: tensor name → buffer.
    buffers: HashMap<String, B::Buffer>,
}

impl<B: Backend> GraphRuntime<B> {
    /// Create a new runtime for the given graph and backend.
    ///
    /// This allocates buffers for all tensors (inputs, outputs, intermediates,
    /// and initializers/weights).
    pub fn new(graph: Graph, backend: B) -> Result<Self> {
        let mut buffers = HashMap::new();

        // Allocate buffers for all shapes
        for (name, shape) in &graph.shapes {
            let size_bytes = shape.size_bytes();
            let buffer = backend.alloc(size_bytes, BufferKind::Storage)
                .map_err(|e| Error::Runtime(format!("failed to allocate {name}: {e}")))?;
            buffers.insert(name.clone(), buffer);
        }

        // Upload initializers (weights)
        for (name, data) in &graph.initializers {
            if let Some(buffer) = buffers.get_mut(name) {
                // Need to handle dtype conversion if graph dtype differs from initializer
                let shape = graph.shapes.get(name);
                let upload_data = match (shape.map(|s| s.dtype), graph.dtype) {
                    (Some(Dtype::F16), Dtype::F16) | (Some(Dtype::F32), Dtype::F32) | (None, _) => {
                        // Same dtype or unknown, upload as-is
                        data.clone()
                    }
                    (Some(Dtype::F32), Dtype::F16) => {
                        // Convert F32 weights to F16
                        convert_f32_to_f16(data)
                    }
                    (Some(Dtype::F16), Dtype::F32) => {
                        // Convert F16 weights to F32
                        convert_f16_to_f32(data)
                    }
                    _ => {
                        // Unknown dtype combination, upload as-is
                        data.clone()
                    }
                };
                
                if upload_data.len() == buffer.len_bytes() {
                    backend.upload(buffer, &upload_data)
                        .map_err(|e| Error::Runtime(format!("failed to upload {name}: {e}")))?;
                }
                // If sizes don't match, it might be a non-tensor initializer (like reshape shape)
            }
        }

        Ok(Self { graph, backend, buffers })
    }

    /// Get the input tensor names and shapes.
    pub fn inputs(&self) -> Vec<(&str, &TensorShape)> {
        self.graph.input_info()
    }

    /// Get the output tensor names and shapes.
    pub fn outputs(&self) -> Vec<(&str, &TensorShape)> {
        self.graph.output_info()
    }

    /// Run the graph with the given inputs.
    ///
    /// `inputs` maps input tensor names to their data (as bytes).
    /// Returns output tensor data.
    pub fn run(&mut self, inputs: &HashMap<&str, &[u8]>) -> Result<HashMap<String, Vec<u8>>> {
        // Upload inputs
        for (name, data) in inputs {
            let buffer = self.buffers.get_mut(*name)
                .ok_or_else(|| Error::Runtime(format!("input tensor not found: {name}")))?;
            
            if data.len() != buffer.len_bytes() {
                return Err(Error::Runtime(format!(
                    "input {name} size mismatch: expected {}, got {}",
                    buffer.len_bytes(), data.len()
                )));
            }
            
            self.backend.upload(buffer, data)
                .map_err(|e| Error::Runtime(format!("failed to upload input {name}: {e}")))?;
        }

        // Execute ops - clone ops to avoid borrow conflict
        let ops: Vec<_> = self.graph.ops.clone();
        for op in &ops {
            self.dispatch_op(op)?;
        }

        // Synchronize
        self.backend.synchronize()
            .map_err(|e| Error::Runtime(format!("synchronize failed: {e}")))?;

        // Download outputs
        let mut outputs = HashMap::new();
        for name in &self.graph.outputs {
            let buffer = self.buffers.get(name)
                .ok_or_else(|| Error::Runtime(format!("output tensor not found: {name}")))?;
            
            let mut data = vec![0u8; buffer.len_bytes()];
            self.backend.download(buffer, &mut data)
                .map_err(|e| Error::Runtime(format!("failed to download output {name}: {e}")))?;
            
            outputs.insert(name.clone(), data);
        }

        Ok(outputs)
    }

    /// Dispatch a single op to the backend.
    fn dispatch_op(&mut self, op: &CompiledOp) -> Result<()> {
        match &op.params {
            OpParams::None => {
                // Relu - dispatch based on op_type
                if op.op_type == "Relu" {
                    self.dispatch_relu(op)?;
                }
                // Other no-param ops might be metadata-only (already handled at compile time)
            }
            OpParams::Clip { min, max } => {
                self.dispatch_clip(op, *min, *max)?;
            }
            OpParams::Conv2d { kernel_shape, strides, pads, group, .. } => {
                self.dispatch_conv2d(op, *kernel_shape, *strides, *pads, *group)?;
            }
            OpParams::Add => {
                self.dispatch_add(op)?;
            }
            OpParams::GlobalAvgPool => {
                self.dispatch_global_avg_pool(op)?;
            }
            OpParams::Gemm { trans_a, trans_b, .. } => {
                self.dispatch_gemm(op, *trans_a, *trans_b)?;
            }
            OpParams::Softmax { axis } => {
                self.dispatch_softmax(op, *axis)?;
            }
            OpParams::Reshape { .. } => {
                // Reshape is metadata-only; just copy data if input != output
                self.dispatch_reshape(op)?;
            }
            OpParams::MaxPool { .. } | OpParams::AvgPool { .. } => {
                // MaxPool/AvgPool not implemented in generic runtime yet
                return Err(Error::Runtime(format!(
                    "MaxPool/AvgPool not implemented in generic runtime, use CpuGraphRuntime"
                )));
            }
            // YOLO ops (Task 005) - defer to CpuGraphRuntime for full implementation
            OpParams::Sigmoid | OpParams::Mul | OpParams::Sub | OpParams::Div |
            OpParams::Concat { .. } | OpParams::Resize { .. } | OpParams::Split { .. } | 
            OpParams::Transpose { .. } | OpParams::Slice { .. } => {
                return Err(Error::Runtime(format!(
                    "{} not implemented in generic runtime, use CpuGraphRuntime",
                    op.op_type
                )));
            }
            // Quantized ops (Task 006) - defer to specialized runtime
            OpParams::Requantize { .. } | OpParams::AddQuantized { .. } |
            OpParams::Quantize { .. } | OpParams::Dequantize { .. } |
            OpParams::QuantizePerChannel { .. } | OpParams::DequantizePerChannel { .. } => {
                return Err(Error::Runtime(format!(
                    "{} (quantized) not implemented in generic runtime",
                    op.op_type
                )));
            }
            // Fused quantized ops (Task 008) - specialized backends only.
            OpParams::Conv2dRequantReluI8Nhwc { .. } => {
                return Err(Error::Runtime(format!(
                    "{} (fused quantized) requires VulkanGraphRuntime",
                    op.op_type
                )));
            }
        }
        Ok(())
    }

    // Individual op dispatchers - these need to be specialized per backend type
    // For now, we provide placeholder implementations that will be filled in
    // when we wire up to specific backends

    fn dispatch_relu(&mut self, op: &CompiledOp) -> Result<()> {
        // Get input and output buffers
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];
        
        // For in-place ops where input == output, this is a no-copy situation
        if input_name == output_name {
            // In-place relu - need mutable access
            // This requires backend-specific handling
            return Err(Error::Runtime("in-place relu not yet implemented".into()));
        }

        // Out-of-place: copy input to output, then apply relu
        // This is a simplified approach; real implementation would use backend ops directly
        let input_shape = self.graph.shapes.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
        
        let n = input_shape.numel();
        
        // Download input, apply relu, upload to output
        let input_buf = self.buffers.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
        let mut data = vec![0u8; input_buf.len_bytes()];
        self.backend.download(input_buf, &mut data)
            .map_err(|e| Error::Runtime(format!("download failed: {e}")))?;

        // Apply relu on CPU (this is a fallback; real impl would use backend ops)
        match self.graph.dtype {
            Dtype::F32 => {
                let floats: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), n)
                };
                for v in floats.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            Dtype::F16 => {
                let halfs: &mut [dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), n)
                };
                for v in halfs.iter_mut() {
                    let f = v.to_f32();
                    *v = dragonwing_core::F16::from_f32(f.max(0.0));
                }
            }
            _ => return Err(Error::Runtime("unsupported dtype for relu".into())),
        }

        let output_buf = self.buffers.get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
        self.backend.upload(output_buf, &data)
            .map_err(|e| Error::Runtime(format!("upload failed: {e}")))?;

        Ok(())
    }

    fn dispatch_clip(&mut self, op: &CompiledOp, min: f32, max: f32) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];
        
        let input_shape = self.graph.shapes.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
        let n = input_shape.numel();

        let input_buf = self.buffers.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
        let mut data = vec![0u8; input_buf.len_bytes()];
        self.backend.download(input_buf, &mut data)
            .map_err(|e| Error::Runtime(format!("download failed: {e}")))?;

        match self.graph.dtype {
            Dtype::F32 => {
                let floats: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), n)
                };
                for v in floats.iter_mut() {
                    *v = v.clamp(min, max);
                }
            }
            Dtype::F16 => {
                let halfs: &mut [dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), n)
                };
                for v in halfs.iter_mut() {
                    let f = v.to_f32().clamp(min, max);
                    *v = dragonwing_core::F16::from_f32(f);
                }
            }
            _ => return Err(Error::Runtime("unsupported dtype for clip".into())),
        }

        let output_buf = self.buffers.get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
        self.backend.upload(output_buf, &data)
            .map_err(|e| Error::Runtime(format!("upload failed: {e}")))?;

        Ok(())
    }

    fn dispatch_conv2d(
        &mut self,
        op: &CompiledOp,
        _kernel_shape: [usize; 2],
        _strides: [usize; 2],
        _pads: [usize; 4],
        _group: usize,
    ) -> Result<()> {
        // Conv2D is complex - for now, signal that it needs backend-specific implementation
        Err(Error::Runtime(format!(
            "Conv2D dispatch not yet implemented for generic backend (op: {})",
            op.name
        )))
    }

    fn dispatch_add(&mut self, op: &CompiledOp) -> Result<()> {
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let a_shape = self.graph.shapes.get(a_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
        let n = a_shape.numel();

        // Download both inputs
        let a_buf = self.buffers.get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let mut a_data = vec![0u8; a_buf.len_bytes()];
        self.backend.download(a_buf, &mut a_data)
            .map_err(|e| Error::Runtime(format!("download A failed: {e}")))?;

        let b_buf = self.buffers.get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        let mut b_data = vec![0u8; b_buf.len_bytes()];
        self.backend.download(b_buf, &mut b_data)
            .map_err(|e| Error::Runtime(format!("download B failed: {e}")))?;

        // Add element-wise (simplified - doesn't handle broadcast)
        match self.graph.dtype {
            Dtype::F32 => {
                let a: &[f32] = unsafe {
                    std::slice::from_raw_parts(a_data.as_ptr().cast(), n)
                };
                let b: &[f32] = unsafe {
                    std::slice::from_raw_parts(b_data.as_ptr().cast(), b_data.len() / 4)
                };
                let out: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(a_data.as_mut_ptr().cast(), n)
                };
                
                // Handle broadcast: if B is smaller, broadcast
                if b.len() == n {
                    for (o, (&ai, &bi)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
                        *o = ai + bi;
                    }
                } else if b.len() == 1 {
                    // Scalar broadcast
                    let bv = b[0];
                    for (o, &ai) in out.iter_mut().zip(a.iter()) {
                        *o = ai + bv;
                    }
                } else {
                    // Channel broadcast (last dim)
                    let c = b.len();
                    for (i, (o, &ai)) in out.iter_mut().zip(a.iter()).enumerate() {
                        *o = ai + b[i % c];
                    }
                }
            }
            Dtype::F16 => {
                // Similar logic for F16
                let a: &[dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts(a_data.as_ptr().cast(), n)
                };
                let b: &[dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts(b_data.as_ptr().cast(), b_data.len() / 2)
                };
                let out: &mut [dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts_mut(a_data.as_mut_ptr().cast(), n)
                };
                
                if b.len() == n {
                    for (o, (ai, bi)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
                        *o = dragonwing_core::F16::from_f32(ai.to_f32() + bi.to_f32());
                    }
                } else {
                    let c = b.len();
                    for (i, (o, ai)) in out.iter_mut().zip(a.iter()).enumerate() {
                        *o = dragonwing_core::F16::from_f32(ai.to_f32() + b[i % c].to_f32());
                    }
                }
            }
            _ => return Err(Error::Runtime("unsupported dtype for add".into())),
        }

        let out_buf = self.buffers.get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
        self.backend.upload(out_buf, &a_data)
            .map_err(|e| Error::Runtime(format!("upload failed: {e}")))?;

        Ok(())
    }

    fn dispatch_global_avg_pool(&mut self, op: &CompiledOp) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self.graph.shapes.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
        
        // Input is NCHW: [N, C, H, W]
        if input_shape.dims.len() != 4 {
            return Err(Error::Runtime("GlobalAvgPool input must be 4D".into()));
        }
        let n = input_shape.dims[0];
        let c = input_shape.dims[1];
        let h = input_shape.dims[2];
        let w = input_shape.dims[3];
        let spatial = h * w;

        let input_buf = self.buffers.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
        let mut input_data = vec![0u8; input_buf.len_bytes()];
        self.backend.download(input_buf, &mut input_data)
            .map_err(|e| Error::Runtime(format!("download failed: {e}")))?;

        let output_shape = self.graph.shapes.get(output_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {output_name}")))?;
        let mut output_data = vec![0u8; output_shape.size_bytes()];

        match self.graph.dtype {
            Dtype::F32 => {
                let input: &[f32] = unsafe {
                    std::slice::from_raw_parts(input_data.as_ptr().cast(), n * c * h * w)
                };
                let output: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(output_data.as_mut_ptr().cast(), n * c)
                };
                
                for batch in 0..n {
                    for ch in 0..c {
                        let mut sum = 0.0f32;
                        for i in 0..spatial {
                            sum += input[((batch * c + ch) * h + i / w) * w + i % w];
                        }
                        output[batch * c + ch] = sum / spatial as f32;
                    }
                }
            }
            Dtype::F16 => {
                let input: &[dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts(input_data.as_ptr().cast(), n * c * h * w)
                };
                let output: &mut [dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts_mut(output_data.as_mut_ptr().cast(), n * c)
                };
                
                for batch in 0..n {
                    for ch in 0..c {
                        let mut sum = 0.0f32;
                        for i in 0..spatial {
                            sum += input[((batch * c + ch) * h + i / w) * w + i % w].to_f32();
                        }
                        output[batch * c + ch] = dragonwing_core::F16::from_f32(sum / spatial as f32);
                    }
                }
            }
            _ => return Err(Error::Runtime("unsupported dtype for global_avg_pool".into())),
        }

        let output_buf = self.buffers.get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
        self.backend.upload(output_buf, &output_data)
            .map_err(|e| Error::Runtime(format!("upload failed: {e}")))?;

        Ok(())
    }

    fn dispatch_gemm(&mut self, op: &CompiledOp, trans_a: bool, trans_b: bool) -> Result<()> {
        // GEMM: C = A * B (+ C if bias present)
        // For now, use a naive CPU implementation
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let a_shape = self.graph.shapes.get(a_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
        let b_shape = self.graph.shapes.get(b_name)
            .or_else(|| {
                // B might be an initializer with different stored shape
                self.graph.initializers.get(b_name).map(|_| {
                    // Try to infer from stored data
                    a_shape // fallback
                })
            })
            .ok_or_else(|| Error::Runtime(format!("shape not found: {b_name}")))?;

        if a_shape.dims.len() != 2 || b_shape.dims.len() != 2 {
            return Err(Error::Runtime("Gemm inputs must be 2D".into()));
        }

        let m = if trans_a { a_shape.dims[1] } else { a_shape.dims[0] };
        let k = if trans_a { a_shape.dims[0] } else { a_shape.dims[1] };
        let n = if trans_b { b_shape.dims[0] } else { b_shape.dims[1] };

        // Download inputs
        let a_buf = self.buffers.get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let mut a_data = vec![0u8; a_buf.len_bytes()];
        self.backend.download(a_buf, &mut a_data)
            .map_err(|e| Error::Runtime(format!("download A failed: {e}")))?;

        let b_buf = self.buffers.get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        let mut b_data = vec![0u8; b_buf.len_bytes()];
        self.backend.download(b_buf, &mut b_data)
            .map_err(|e| Error::Runtime(format!("download B failed: {e}")))?;

        // Optional bias (third input)
        let bias_data = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            let c_name = &op.inputs[2];
            if let Some(c_buf) = self.buffers.get(c_name) {
                let mut data = vec![0u8; c_buf.len_bytes()];
                self.backend.download(c_buf, &mut data)
                    .map_err(|e| Error::Runtime(format!("download C failed: {e}")))?;
                Some(data)
            } else {
                None
            }
        } else {
            None
        };

        let out_shape = self.graph.shapes.get(out_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {out_name}")))?;
        let mut out_data = vec![0u8; out_shape.size_bytes()];

        match self.graph.dtype {
            Dtype::F32 => {
                let a: &[f32] = unsafe {
                    std::slice::from_raw_parts(a_data.as_ptr().cast(), a_data.len() / 4)
                };
                let b: &[f32] = unsafe {
                    std::slice::from_raw_parts(b_data.as_ptr().cast(), b_data.len() / 4)
                };
                let out: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(out_data.as_mut_ptr().cast(), m * n)
                };

                // Naive GEMM
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            let a_idx = if trans_a { l * m + i } else { i * k + l };
                            let b_idx = if trans_b { j * k + l } else { l * n + j };
                            sum += a[a_idx] * b[b_idx];
                        }
                        out[i * n + j] = sum;
                    }
                }

                // Add bias if present
                if let Some(ref bias) = bias_data {
                    let bias_f: &[f32] = unsafe {
                        std::slice::from_raw_parts(bias.as_ptr().cast(), bias.len() / 4)
                    };
                    for i in 0..m {
                        for j in 0..n {
                            out[i * n + j] += bias_f[j % bias_f.len()];
                        }
                    }
                }
            }
            Dtype::F16 => {
                // Similar for F16, computing in F32
                let a: &[dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts(a_data.as_ptr().cast(), a_data.len() / 2)
                };
                let b: &[dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts(b_data.as_ptr().cast(), b_data.len() / 2)
                };
                let out: &mut [dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts_mut(out_data.as_mut_ptr().cast(), m * n)
                };

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            let a_idx = if trans_a { l * m + i } else { i * k + l };
                            let b_idx = if trans_b { j * k + l } else { l * n + j };
                            sum += a[a_idx].to_f32() * b[b_idx].to_f32();
                        }
                        out[i * n + j] = dragonwing_core::F16::from_f32(sum);
                    }
                }

                if let Some(ref bias) = bias_data {
                    let bias_f: &[dragonwing_core::F16] = unsafe {
                        std::slice::from_raw_parts(bias.as_ptr().cast(), bias.len() / 2)
                    };
                    for i in 0..m {
                        for j in 0..n {
                            let v = out[i * n + j].to_f32() + bias_f[j % bias_f.len()].to_f32();
                            out[i * n + j] = dragonwing_core::F16::from_f32(v);
                        }
                    }
                }
            }
            _ => return Err(Error::Runtime("unsupported dtype for gemm".into())),
        }

        let out_buf = self.buffers.get_mut(out_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
        self.backend.upload(out_buf, &out_data)
            .map_err(|e| Error::Runtime(format!("upload failed: {e}")))?;

        Ok(())
    }

    fn dispatch_softmax(&mut self, op: &CompiledOp, _axis: i64) -> Result<()> {
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        let input_shape = self.graph.shapes.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
        
        let input_buf = self.buffers.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
        let mut data = vec![0u8; input_buf.len_bytes()];
        self.backend.download(input_buf, &mut data)
            .map_err(|e| Error::Runtime(format!("download failed: {e}")))?;

        // Softmax along last axis
        let last_dim = *input_shape.dims.last().unwrap_or(&1);
        let rows = input_shape.numel() / last_dim;

        match self.graph.dtype {
            Dtype::F32 => {
                let floats: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), input_shape.numel())
                };
                
                for row in 0..rows {
                    let start = row * last_dim;
                    let end = start + last_dim;
                    let slice = &mut floats[start..end];
                    
                    // Find max for numerical stability
                    let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    
                    // exp(x - max)
                    let mut sum = 0.0f32;
                    for v in slice.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    
                    // Normalize
                    for v in slice.iter_mut() {
                        *v /= sum;
                    }
                }
            }
            Dtype::F16 => {
                let halfs: &mut [dragonwing_core::F16] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), input_shape.numel())
                };
                
                for row in 0..rows {
                    let start = row * last_dim;
                    let end = start + last_dim;
                    let slice = &mut halfs[start..end];
                    
                    let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, b| a.max(b.to_f32()));
                    
                    let mut sum = 0.0f32;
                    let temp: Vec<f32> = slice.iter().map(|v| {
                        let e = (v.to_f32() - max_val).exp();
                        sum += e;
                        e
                    }).collect();
                    
                    for (v, t) in slice.iter_mut().zip(temp.iter()) {
                        *v = dragonwing_core::F16::from_f32(t / sum);
                    }
                }
            }
            _ => return Err(Error::Runtime("unsupported dtype for softmax".into())),
        }

        let output_buf = self.buffers.get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
        self.backend.upload(output_buf, &data)
            .map_err(|e| Error::Runtime(format!("upload failed: {e}")))?;

        Ok(())
    }

    fn dispatch_reshape(&mut self, op: &CompiledOp) -> Result<()> {
        // Reshape is zero-copy if input and output are the same buffer
        // For now, just copy data
        let input_name = &op.inputs[0];
        let output_name = &op.outputs[0];

        if input_name == output_name {
            return Ok(()); // Already in place
        }

        let input_buf = self.buffers.get(input_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
        let mut data = vec![0u8; input_buf.len_bytes()];
        self.backend.download(input_buf, &mut data)
            .map_err(|e| Error::Runtime(format!("download failed: {e}")))?;

        let output_buf = self.buffers.get_mut(output_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
        self.backend.upload(output_buf, &data)
            .map_err(|e| Error::Runtime(format!("upload failed: {e}")))?;

        Ok(())
    }
}

// Helper functions

fn convert_f32_to_f16(data: &[u8]) -> Vec<u8> {
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast(), data.len() / 4)
    };
    let mut result = Vec::with_capacity(floats.len() * 2);
    for &f in floats {
        let h = dragonwing_core::F16::from_f32(f);
        // F16 stores its bits in the inner u16 field (F16.0)
        result.extend_from_slice(&h.0.to_le_bytes());
    }
    result
}

fn convert_f16_to_f32(data: &[u8]) -> Vec<u8> {
    let halfs: &[dragonwing_core::F16] = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast(), data.len() / 2)
    };
    let mut result = Vec::with_capacity(halfs.len() * 4);
    for h in halfs {
        result.extend_from_slice(&h.to_f32().to_le_bytes());
    }
    result
}

// ===========================================================================
// CPU-optimized runtime (feature-gated)
// ===========================================================================

#[cfg(feature = "cpu")]
mod cpu_runtime {
    use super::*;
    use dragonwing_cpu::{CpuBackend, CpuBuffer, ops};
    use dragonwing_core::F16;

    /// CPU-optimized graph runtime.
    ///
    /// This runtime operates directly on CPU buffer slices, avoiding the
    /// download/upload overhead of the generic runtime. It uses the optimized
    /// ops from `dragonwing_cpu::ops`.
    pub struct CpuGraphRuntime {
        /// The compiled graph.
        graph: Graph,
        /// Allocated buffers: tensor name → buffer.
        buffers: HashMap<String, CpuBuffer>,
        /// Number of threads to use for parallelizable ops.
        num_threads: usize,
    }

    impl CpuGraphRuntime {
        /// Create a new CPU runtime for the given graph.
        pub fn new(graph: Graph) -> Result<Self> {
            Self::with_threads(graph, dragonwing_cpu::num_cpus())
        }

        /// Create a new CPU runtime with a specific thread count.
        pub fn with_threads(graph: Graph, num_threads: usize) -> Result<Self> {
            let backend = CpuBackend::new();
            let mut buffers = HashMap::new();

            // Allocate buffers for all shapes
            for (name, shape) in &graph.shapes {
                let size_bytes = shape.size_bytes();
                let buffer = backend.alloc(size_bytes, BufferKind::Storage)
                    .map_err(|e| Error::Runtime(format!("failed to allocate {name}: {e}")))?;
                buffers.insert(name.clone(), buffer);
            }

            // Upload initializers (weights)
            for (name, data) in &graph.initializers {
                if let Some(buffer) = buffers.get_mut(name) {
                    let shape = graph.shapes.get(name);
                    let upload_data = match (shape.map(|s| s.dtype), graph.dtype) {
                        (Some(Dtype::F16), Dtype::F16) | (Some(Dtype::F32), Dtype::F32) | (None, _) => {
                            data.clone()
                        }
                        (Some(Dtype::F32), Dtype::F16) => {
                            convert_f32_to_f16(data)
                        }
                        (Some(Dtype::F16), Dtype::F32) => {
                            convert_f16_to_f32(data)
                        }
                        _ => data.clone(),
                    };
                    let buf_bytes = buffer.as_bytes_mut();
                    if buf_bytes.len() != upload_data.len() {
                        return Err(Error::Runtime(format!(
                            "initializer size mismatch for {name}: buffer={} bytes, data={} bytes, shape={:?}",
                            buf_bytes.len(), upload_data.len(), shape
                        )));
                    }
                    buf_bytes.copy_from_slice(&upload_data);
                }
            }

            Ok(Self { graph, buffers, num_threads })
        }

        /// Set an input tensor (F32 data).
        pub fn set_input_f32(&mut self, name: &str, data: &[f32]) -> Result<()> {
            let buffer = self.buffers.get_mut(name)
                .ok_or_else(|| Error::Runtime(format!("input not found: {name}")))?;
            
            match self.graph.dtype {
                Dtype::F32 => {
                    let buf_f32 = buffer.as_f32_mut();
                    if buf_f32.len() != data.len() {
                        return Err(Error::Runtime(format!(
                            "input size mismatch: expected {}, got {}",
                            buf_f32.len(), data.len()
                        )));
                    }
                    buf_f32.copy_from_slice(data);
                }
                Dtype::F16 => {
                    let buf_f16 = buffer.as_f16_mut();
                    if buf_f16.len() != data.len() {
                        return Err(Error::Runtime(format!(
                            "input size mismatch: expected {}, got {}",
                            buf_f16.len(), data.len()
                        )));
                    }
                    for (dst, &src) in buf_f16.iter_mut().zip(data.iter()) {
                        *dst = F16::from_f32(src);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype".into())),
            }
            Ok(())
        }

        /// Get an output tensor (as F32 data).
        pub fn get_output_f32(&self, name: &str) -> Result<Vec<f32>> {
            let buffer = self.buffers.get(name)
                .ok_or_else(|| Error::Runtime(format!("output not found: {name}")))?;
            
            match self.graph.dtype {
                Dtype::F32 => Ok(buffer.as_f32().to_vec()),
                Dtype::F16 => {
                    Ok(buffer.as_f16().iter().map(|v| v.to_f32()).collect())
                }
                _ => Err(Error::Runtime("unsupported dtype".into())),
            }
        }

        /// Run inference.
        pub fn run(&mut self) -> Result<()> {
            let ops: Vec<_> = self.graph.ops.clone();
            for op in &ops {
                self.dispatch_op(op)?;
            }
            Ok(())
        }

        fn dispatch_op(&mut self, op: &CompiledOp) -> Result<()> {
            match &op.params {
                OpParams::None => {
                    // Dispatch based on op_type
                    match op.op_type.as_str() {
                        "Relu" => self.dispatch_relu(op),
                        "Flatten" => self.dispatch_reshape(op),
                        "MatMul" => self.dispatch_matmul(op),
                        _ => Ok(()), // Other no-param ops are metadata-only
                    }
                }
                OpParams::Clip { min, max } => self.dispatch_clip(op, *min, *max),
                OpParams::Add => self.dispatch_add(op),
                OpParams::GlobalAvgPool => self.dispatch_global_avg_pool(op),
                OpParams::Gemm { trans_a, trans_b, .. } => self.dispatch_gemm(op, *trans_a, *trans_b),
                OpParams::Softmax { axis } => self.dispatch_softmax(op, *axis),
                OpParams::Reshape { .. } => self.dispatch_reshape(op),
                OpParams::Conv2d { kernel_shape, strides, pads, dilations: _, group } => {
                    self.dispatch_conv2d(op, *kernel_shape, *strides, *pads, *group)
                }
                OpParams::MaxPool { kernel_shape, strides, pads: _ } => {
                    self.dispatch_maxpool(op, *kernel_shape, *strides)
                }
                OpParams::AvgPool { kernel_shape, strides, pads: _ } => {
                    self.dispatch_avgpool(op, *kernel_shape, *strides)
                }
                // YOLO ops (Task 005)
                OpParams::Sigmoid => self.dispatch_sigmoid(op),
                OpParams::Mul => self.dispatch_mul(op),
                OpParams::Sub => self.dispatch_sub(op),
                OpParams::Div => self.dispatch_div(op),
                OpParams::Concat { axis } => self.dispatch_concat(op, *axis),
                OpParams::Resize { out_h, out_w, mode } => {
                    self.dispatch_resize(op, *out_h, *out_w, mode)
                }
                OpParams::Split { axis, split_sizes } => {
                    self.dispatch_split(op, *axis, split_sizes)
                }
                OpParams::Transpose { perm } => self.dispatch_transpose(op, perm),
                OpParams::Slice { starts, ends, axes, steps } => {
                    self.dispatch_slice(op, starts, ends, axes, steps)
                }
                // Quantized ops (Task 006)
                OpParams::Requantize { scale } => self.dispatch_requantize(op, *scale),
                OpParams::AddQuantized { scale_a_over_out, scale_b_over_out } => {
                    self.dispatch_add_quantized(op, *scale_a_over_out, *scale_b_over_out)
                }
                OpParams::Quantize { scale } => self.dispatch_quantize(op, *scale),
                OpParams::Dequantize { scale } => self.dispatch_dequantize(op, *scale),
                // Per-channel QDQ (Task 009) — implemented on CPU. Vulkan path
                // is intentionally absent for now; the QDQ-fold pass should
                // strip these before the runtime sees them in production.
                OpParams::QuantizePerChannel { scales, zero_points, axis } => {
                    self.dispatch_quantize_per_channel(op, scales, zero_points, *axis)
                }
                OpParams::DequantizePerChannel { scales, zero_points, axis } => {
                    self.dispatch_dequantize_per_channel(op, scales, zero_points, *axis)
                }
                // Fused quantized ops (Task 008) - not implemented on CPU; require
                // VulkanGraphRuntime. Emit a clear error so callers know to switch
                // backends or skip fusion when targeting CPU.
                OpParams::Conv2dRequantReluI8Nhwc { .. } => {
                    Err(Error::Runtime(format!(
                        "{} (fused Conv+Requant+Relu I8) not implemented on CPU; \
                         use VulkanGraphRuntime or skip the fusion pass",
                        op.op_type
                    )))
                }
            }
        }

        fn dispatch_relu(&mut self, op: &CompiledOp) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            // Get shapes for size calculation
            let _n = self.graph.shapes.get(input_name)
                .map(|s| s.numel())
                .unwrap_or(0);

            // Get input and output buffers
            // For in-place ops, input == output
            if input_name == output_name {
                let buffer = self.buffers.get_mut(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                
                match self.graph.dtype {
                    Dtype::F32 => {
                        let data = buffer.as_f32_mut();
                        // In-place relu
                        for v in data.iter_mut() {
                            *v = v.max(0.0);
                        }
                    }
                    Dtype::F16 => {
                        let data = buffer.as_f16_mut();
                        for v in data.iter_mut() {
                            let f = v.to_f32();
                            *v = F16::from_f32(f.max(0.0));
                        }
                    }
                    _ => return Err(Error::Runtime("unsupported dtype for relu".into())),
                }
            } else {
                // Need to borrow both buffers - use raw pointers to avoid borrow checker issues
                let (input_ptr, output_ptr) = {
                    let input_buf = self.buffers.get(input_name)
                        .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                    let output_buf = self.buffers.get(output_name)
                        .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                    (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
                };

                match self.graph.dtype {
                    Dtype::F32 => {
                        // SAFETY: We know input and output are different buffers
                        let input = unsafe { (*input_ptr).as_f32() };
                        let output = unsafe { (*output_ptr).as_f32_mut() };
                        ops::relu_f32(output, input);
                    }
                    Dtype::F16 => {
                        let input = unsafe { (*input_ptr).as_f16() };
                        let output = unsafe { (*output_ptr).as_f16_mut() };
                        ops::relu_fp16(output, input);
                    }
                    _ => return Err(Error::Runtime("unsupported dtype for relu".into())),
                }
            }
            Ok(())
        }

        fn dispatch_clip(&mut self, op: &CompiledOp, min: f32, max: f32) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            // For simplicity, copy input to output first if different
            if input_name != output_name {
                let (input_ptr, output_ptr) = {
                    let input_buf = self.buffers.get(input_name)
                        .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                    let output_buf = self.buffers.get(output_name)
                        .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                    (input_buf.as_bytes().as_ptr(), output_buf as *const CpuBuffer as *mut CpuBuffer)
                };
                let len = self.buffers.get(input_name).unwrap().as_bytes().len();
                unsafe {
                    (*output_ptr).as_bytes_mut().copy_from_slice(
                        std::slice::from_raw_parts(input_ptr, len)
                    );
                }
            }

            let buffer = self.buffers.get_mut(output_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;

            match self.graph.dtype {
                Dtype::F32 => {
                    for v in buffer.as_f32_mut().iter_mut() {
                        *v = v.clamp(min, max);
                    }
                }
                Dtype::F16 => {
                    for v in buffer.as_f16_mut().iter_mut() {
                        let f = v.to_f32().clamp(min, max);
                        *v = F16::from_f32(f);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for clip".into())),
            }
            Ok(())
        }

        fn dispatch_add(&mut self, op: &CompiledOp) -> Result<()> {
            let a_name = &op.inputs[0];
            let b_name = &op.inputs[1];
            let out_name = &op.outputs[0];

            let a_shape = self.graph.shapes.get(a_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
            let b_shape = self.graph.shapes.get(b_name);

            let n = a_shape.numel();
            let b_len = b_shape.map(|s| s.numel()).unwrap_or(n);

            // Get raw pointers to avoid borrow issues
            let (a_ptr, b_ptr, out_ptr) = {
                let a_buf = self.buffers.get(a_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
                let b_buf = self.buffers.get(b_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
                let out_buf = self.buffers.get(out_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
                (a_buf as *const CpuBuffer, b_buf as *const CpuBuffer, out_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let a = unsafe { (*a_ptr).as_f32() };
                    let b = unsafe { (*b_ptr).as_f32() };
                    let out = unsafe { (*out_ptr).as_f32_mut() };

                    if b_len == n {
                        ops::add_f32(out, a, b);
                    } else if b_len == 1 {
                        // Scalar broadcast
                        let bv = b[0];
                        for (o, &ai) in out.iter_mut().zip(a.iter()) {
                            *o = ai + bv;
                        }
                    } else {
                        // Channel broadcast
                        let c = b_len;
                        for (i, (o, &ai)) in out.iter_mut().zip(a.iter()).enumerate() {
                            *o = ai + b[i % c];
                        }
                    }
                }
                Dtype::F16 => {
                    let a = unsafe { (*a_ptr).as_f16() };
                    let b = unsafe { (*b_ptr).as_f16() };
                    let out = unsafe { (*out_ptr).as_f16_mut() };

                    if b_len == n {
                        ops::add_fp16(out, a, b);
                    } else {
                        let c = b_len;
                        for (i, (o, ai)) in out.iter_mut().zip(a.iter()).enumerate() {
                            *o = F16::from_f32(ai.to_f32() + b[i % c].to_f32());
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for add".into())),
            }
            Ok(())
        }

        fn dispatch_global_avg_pool(&mut self, op: &CompiledOp) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

            // Input is NHWC: [N, H, W, C]
            if input_shape.dims.len() != 4 {
                return Err(Error::Runtime("GlobalAvgPool input must be 4D".into()));
            }

            let n = input_shape.dims[0];
            let h = input_shape.dims[1];
            let w = input_shape.dims[2];
            let c = input_shape.dims[3];
            let spatial = h * w;

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };

                    // NHWC global avg pool: average over H,W for each N,C
                    for batch in 0..n {
                        for ch in 0..c {
                            let mut sum = 0.0f32;
                            for row in 0..h {
                                for col in 0..w {
                                    let idx = ((batch * h + row) * w + col) * c + ch;
                                    sum += input[idx];
                                }
                            }
                            // Output is [N, 1, 1, C] or [N, C]
                            output[batch * c + ch] = sum / spatial as f32;
                        }
                    }
                }
                Dtype::F16 => {
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };

                    for batch in 0..n {
                        for ch in 0..c {
                            let mut sum = 0.0f32;
                            for row in 0..h {
                                for col in 0..w {
                                    let idx = ((batch * h + row) * w + col) * c + ch;
                                    sum += input[idx].to_f32();
                                }
                            }
                            output[batch * c + ch] = F16::from_f32(sum / spatial as f32);
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for global_avg_pool".into())),
            }
            Ok(())
        }

        fn dispatch_gemm(&mut self, op: &CompiledOp, trans_a: bool, trans_b: bool) -> Result<()> {
            let a_name = &op.inputs[0];
            let b_name = &op.inputs[1];
            let out_name = &op.outputs[0];

            let a_shape = self.graph.shapes.get(a_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
            let b_shape = self.graph.shapes.get(b_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {b_name}")))?;

            if a_shape.dims.len() != 2 || b_shape.dims.len() != 2 {
                return Err(Error::Runtime("Gemm inputs must be 2D".into()));
            }

            let m = if trans_a { a_shape.dims[1] } else { a_shape.dims[0] };
            let k = if trans_a { a_shape.dims[0] } else { a_shape.dims[1] };
            let n = if trans_b { b_shape.dims[0] } else { b_shape.dims[1] };

            // Get raw pointers
            let (a_ptr, b_ptr, out_ptr) = {
                let a_buf = self.buffers.get(a_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
                let b_buf = self.buffers.get(b_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
                let out_buf = self.buffers.get(out_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
                (a_buf as *const CpuBuffer, b_buf as *const CpuBuffer, out_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            // Get optional bias
            let bias_ptr = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
                self.buffers.get(&op.inputs[2]).map(|b| b as *const CpuBuffer)
            } else {
                None
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let a = unsafe { (*a_ptr).as_f32() };
                    let b = unsafe { (*b_ptr).as_f32() };
                    let out = unsafe { (*out_ptr).as_f32_mut() };

                    // Use multi-threaded gemm if not transposed
                    if !trans_a && !trans_b {
                        ops::gemm_f32_mt(out, a, b, m, n, k, self.num_threads);
                    } else {
                        // Fall back to naive with transpose handling
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0f32;
                                for l in 0..k {
                                    let a_idx = if trans_a { l * m + i } else { i * k + l };
                                    let b_idx = if trans_b { j * k + l } else { l * n + j };
                                    sum += a[a_idx] * b[b_idx];
                                }
                                out[i * n + j] = sum;
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(bias_p) = bias_ptr {
                        let bias = unsafe { (*bias_p).as_f32() };
                        for i in 0..m {
                            for j in 0..n {
                                out[i * n + j] += bias[j % bias.len()];
                            }
                        }
                    }
                }
                Dtype::F16 => {
                    let a = unsafe { (*a_ptr).as_f16() };
                    let b = unsafe { (*b_ptr).as_f16() };
                    let out = unsafe { (*out_ptr).as_f16_mut() };

                    if !trans_a && !trans_b {
                        ops::gemm_fp16_mt(out, a, b, m, n, k, self.num_threads);
                    } else {
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0f32;
                                for l in 0..k {
                                    let a_idx = if trans_a { l * m + i } else { i * k + l };
                                    let b_idx = if trans_b { j * k + l } else { l * n + j };
                                    sum += a[a_idx].to_f32() * b[b_idx].to_f32();
                                }
                                out[i * n + j] = F16::from_f32(sum);
                            }
                        }
                    }

                    if let Some(bias_p) = bias_ptr {
                        let bias = unsafe { (*bias_p).as_f16() };
                        for i in 0..m {
                            for j in 0..n {
                                let v = out[i * n + j].to_f32() + bias[j % bias.len()].to_f32();
                                out[i * n + j] = F16::from_f32(v);
                            }
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for gemm".into())),
            }
            Ok(())
        }

        fn dispatch_matmul(&mut self, op: &CompiledOp) -> Result<()> {
            // MatMul is similar to Gemm without transpose options
            self.dispatch_gemm(op, false, false)
        }

        fn dispatch_softmax(&mut self, op: &CompiledOp, _axis: i64) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

            let last_dim = *input_shape.dims.last().unwrap_or(&1);

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::softmax_f32(output, input, last_dim);
                }
                Dtype::F16 => {
                    // Convert to F32, compute, convert back
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };

                    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
                    let mut output_f32 = vec![0.0f32; input_f32.len()];
                    ops::softmax_f32(&mut output_f32, &input_f32, last_dim);

                    for (o, &v) in output.iter_mut().zip(output_f32.iter()) {
                        *o = F16::from_f32(v);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for softmax".into())),
            }
            Ok(())
        }

        fn dispatch_reshape(&mut self, op: &CompiledOp) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            if input_name == output_name {
                return Ok(()); // In-place reshape is a no-op
            }

            // Copy data
            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf.as_bytes().as_ptr(), output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            let len = self.buffers.get(input_name).unwrap().as_bytes().len();
            unsafe {
                (*output_ptr).as_bytes_mut().copy_from_slice(
                    std::slice::from_raw_parts(input_ptr, len)
                );
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
            let output_name = &op.outputs[0];

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
            let output_shape = self.graph.shapes.get(output_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {output_name}")))?;
            let _weight_shape = self.graph.shapes.get(weight_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {weight_name}")))?;

            // Input: [N, H, W, C_in] in NHWC
            if input_shape.dims.len() != 4 {
                return Err(Error::Runtime("Conv2d input must be 4D".into()));
            }

            let n = input_shape.dims[0];
            let h_in = input_shape.dims[1];
            let w_in = input_shape.dims[2];
            let c_in = input_shape.dims[3];

            // Weight shape depends on how it was stored
            // For NHWC, weights should be [K_h, K_w, C_in/group, C_out]
            let c_out = output_shape.dims[3];
            let k_h = kernel_shape[0];
            let k_w = kernel_shape[1];

            // Padding: [top, left, bottom, right] -> symmetric
            let pad_h = pads[0];
            let pad_w = pads[1];

            // Get raw pointers
            let (input_ptr, weight_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let weight_buf = self.buffers.get(weight_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {weight_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, weight_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            // Get optional bias
            let bias_ptr = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
                self.buffers.get(&op.inputs[2]).map(|b| b as *const CpuBuffer)
            } else {
                None
            };

            // Determine if this is a depthwise convolution (group == c_in)
            let is_depthwise = group == c_in && group > 1;
            let channel_multiplier = if is_depthwise { c_out / c_in } else { 1 };

            if group != 1 && !is_depthwise {
                return Err(Error::Runtime(format!(
                    "general grouped convolution not yet supported (group={}, c_in={})",
                    group, c_in
                )));
            }

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let weight = unsafe { (*weight_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };

                    if is_depthwise {
                        ops::depthwise_conv2d_f32_nhwc_mt(
                            output, input, weight,
                            n, h_in, w_in, c_in, channel_multiplier,
                            k_h, k_w, strides[0], strides[1], pad_h, pad_w,
                            self.num_threads
                        );
                    } else {
                        ops::conv2d_f32_nhwc_mt(
                            output, input, weight,
                            n, h_in, w_in, c_in, c_out,
                            k_h, k_w, strides[0], strides[1], pad_h, pad_w,
                            self.num_threads
                        );
                    }

                    // Add bias if present
                    if let Some(bias_p) = bias_ptr {
                        let bias = unsafe { (*bias_p).as_f32() };
                        let h_out = output_shape.dims[1];
                        let w_out = output_shape.dims[2];
                        for batch in 0..n {
                            for row in 0..h_out {
                                for col in 0..w_out {
                                    for ch in 0..c_out {
                                        let idx = ((batch * h_out + row) * w_out + col) * c_out + ch;
                                        output[idx] += bias[ch];
                                    }
                                }
                            }
                        }
                    }
                }
                Dtype::F16 => {
                    let input = unsafe { (*input_ptr).as_f16() };
                    let weight = unsafe { (*weight_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };

                    if is_depthwise {
                        ops::depthwise_conv2d_fp16_nhwc_mt(
                            output, input, weight,
                            n, h_in, w_in, c_in, channel_multiplier,
                            k_h, k_w, strides[0], strides[1], pad_h, pad_w,
                            self.num_threads
                        );
                    } else {
                        ops::conv2d_fp16_nhwc_mt(
                            output, input, weight,
                            n, h_in, w_in, c_in, c_out,
                            k_h, k_w, strides[0], strides[1], pad_h, pad_w,
                            self.num_threads
                        );
                    }

                    if let Some(bias_p) = bias_ptr {
                        let bias = unsafe { (*bias_p).as_f16() };
                        let h_out = output_shape.dims[1];
                        let w_out = output_shape.dims[2];
                        for batch in 0..n {
                            for row in 0..h_out {
                                for col in 0..w_out {
                                    for ch in 0..c_out {
                                        let idx = ((batch * h_out + row) * w_out + col) * c_out + ch;
                                        let v = output[idx].to_f32() + bias[ch].to_f32();
                                        output[idx] = F16::from_f32(v);
                                    }
                                }
                            }
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for conv2d".into())),
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

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

            if input_shape.dims.len() != 4 {
                return Err(Error::Runtime("MaxPool input must be 4D".into()));
            }

            let n = input_shape.dims[0];
            let h_in = input_shape.dims[1];
            let w_in = input_shape.dims[2];
            let c = input_shape.dims[3];

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::maxpool2d_f32_nhwc(
                        output, input,
                        n, h_in, w_in, c,
                        kernel_shape[0], kernel_shape[1],
                        strides[0], strides[1]
                    );
                }
                Dtype::F16 => {
                    // No F16 maxpool in ops, convert manually
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };

                    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
                    let h_out = (h_in - kernel_shape[0]) / strides[0] + 1;
                    let w_out = (w_in - kernel_shape[1]) / strides[1] + 1;
                    let mut output_f32 = vec![0.0f32; n * h_out * w_out * c];

                    ops::maxpool2d_f32_nhwc(
                        &mut output_f32, &input_f32,
                        n, h_in, w_in, c,
                        kernel_shape[0], kernel_shape[1],
                        strides[0], strides[1]
                    );

                    for (o, &v) in output.iter_mut().zip(output_f32.iter()) {
                        *o = F16::from_f32(v);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for maxpool".into())),
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

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

            if input_shape.dims.len() != 4 {
                return Err(Error::Runtime("AvgPool input must be 4D".into()));
            }

            let n = input_shape.dims[0];
            let h_in = input_shape.dims[1];
            let w_in = input_shape.dims[2];
            let c = input_shape.dims[3];

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::avgpool2d_f32_nhwc(
                        output, input,
                        n, h_in, w_in, c,
                        kernel_shape[0], kernel_shape[1],
                        strides[0], strides[1]
                    );
                }
                Dtype::F16 => {
                    // No F16 avgpool in ops, convert manually
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };

                    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
                    let h_out = (h_in - kernel_shape[0]) / strides[0] + 1;
                    let w_out = (w_in - kernel_shape[1]) / strides[1] + 1;
                    let mut output_f32 = vec![0.0f32; n * h_out * w_out * c];

                    ops::avgpool2d_f32_nhwc(
                        &mut output_f32, &input_f32,
                        n, h_in, w_in, c,
                        kernel_shape[0], kernel_shape[1],
                        strides[0], strides[1]
                    );

                    for (o, &v) in output.iter_mut().zip(output_f32.iter()) {
                        *o = F16::from_f32(v);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for avgpool".into())),
            }
            Ok(())
        }

        // =====================================================================
        // YOLO ops dispatchers (Task 005)
        // =====================================================================

        fn dispatch_sigmoid(&mut self, op: &CompiledOp) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            let (input_ptr, output_ptr) = if input_name == output_name {
                // In-place operation
                let buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                (buf as *const CpuBuffer, buf as *const CpuBuffer as *mut CpuBuffer)
            } else {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::sigmoid_f32(output, input);
                }
                Dtype::F16 => {
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };
                    ops::sigmoid_fp16(output, input);
                }
                _ => return Err(Error::Runtime("unsupported dtype for sigmoid".into())),
            }
            Ok(())
        }

        fn dispatch_mul(&mut self, op: &CompiledOp) -> Result<()> {
            let a_name = &op.inputs[0];
            let b_name = &op.inputs[1];
            let out_name = &op.outputs[0];

            let a_shape = self.graph.shapes.get(a_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
            let n = a_shape.numel();

            let (a_ptr, b_ptr, out_ptr) = {
                let a_buf = self.buffers.get(a_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
                let b_buf = self.buffers.get(b_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
                let out_buf = self.buffers.get(out_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
                (a_buf as *const CpuBuffer, b_buf as *const CpuBuffer, 
                 out_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let a = unsafe { (*a_ptr).as_f32() };
                    let b = unsafe { (*b_ptr).as_f32() };
                    let output = unsafe { (*out_ptr).as_f32_mut() };
                    
                    // Handle broadcasting
                    if a.len() == b.len() {
                        ops::mul_f32(output, a, b);
                    } else if b.len() == 1 {
                        // Scalar broadcast
                        let bv = b[0];
                        for (o, &av) in output.iter_mut().zip(a.iter()) {
                            *o = av * bv;
                        }
                    } else {
                        // Channel broadcast
                        let c = b.len();
                        for (i, (o, &av)) in output.iter_mut().zip(a.iter()).enumerate() {
                            *o = av * b[i % c];
                        }
                    }
                }
                Dtype::F16 => {
                    let a = unsafe { (*a_ptr).as_f16() };
                    let b = unsafe { (*b_ptr).as_f16() };
                    let output = unsafe { (*out_ptr).as_f16_mut() };
                    
                    if a.len() == b.len() {
                        ops::mul_fp16(output, a, b);
                    } else {
                        let c = b.len();
                        for (i, (o, av)) in output.iter_mut().zip(a.iter()).enumerate() {
                            *o = F16::from_f32(av.to_f32() * b[i % c].to_f32());
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for mul".into())),
            }
            Ok(())
        }

        fn dispatch_sub(&mut self, op: &CompiledOp) -> Result<()> {
            let a_name = &op.inputs[0];
            let b_name = &op.inputs[1];
            let out_name = &op.outputs[0];

            let a_shape = self.graph.shapes.get(a_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
            let _n = a_shape.numel();

            let (a_ptr, b_ptr, out_ptr) = {
                let a_buf = self.buffers.get(a_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
                let b_buf = self.buffers.get(b_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
                let out_buf = self.buffers.get(out_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
                (a_buf as *const CpuBuffer, b_buf as *const CpuBuffer, 
                 out_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let a = unsafe { (*a_ptr).as_f32() };
                    let b = unsafe { (*b_ptr).as_f32() };
                    let output = unsafe { (*out_ptr).as_f32_mut() };
                    
                    if a.len() == b.len() {
                        ops::sub_f32(output, a, b);
                    } else if b.len() == 1 {
                        let bv = b[0];
                        for (o, &av) in output.iter_mut().zip(a.iter()) {
                            *o = av - bv;
                        }
                    } else {
                        let c = b.len();
                        for (i, (o, &av)) in output.iter_mut().zip(a.iter()).enumerate() {
                            *o = av - b[i % c];
                        }
                    }
                }
                Dtype::F16 => {
                    let a = unsafe { (*a_ptr).as_f16() };
                    let b = unsafe { (*b_ptr).as_f16() };
                    let output = unsafe { (*out_ptr).as_f16_mut() };
                    
                    if a.len() == b.len() {
                        ops::sub_fp16(output, a, b);
                    } else {
                        let c = b.len();
                        for (i, (o, av)) in output.iter_mut().zip(a.iter()).enumerate() {
                            *o = F16::from_f32(av.to_f32() - b[i % c].to_f32());
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for sub".into())),
            }
            Ok(())
        }

        fn dispatch_div(&mut self, op: &CompiledOp) -> Result<()> {
            let a_name = &op.inputs[0];
            let b_name = &op.inputs[1];
            let out_name = &op.outputs[0];

            let a_shape = self.graph.shapes.get(a_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
            let _n = a_shape.numel();

            let (a_ptr, b_ptr, out_ptr) = {
                let a_buf = self.buffers.get(a_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
                let b_buf = self.buffers.get(b_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
                let out_buf = self.buffers.get(out_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
                (a_buf as *const CpuBuffer, b_buf as *const CpuBuffer, 
                 out_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let a = unsafe { (*a_ptr).as_f32() };
                    let b = unsafe { (*b_ptr).as_f32() };
                    let output = unsafe { (*out_ptr).as_f32_mut() };
                    
                    if a.len() == b.len() {
                        ops::div_f32(output, a, b);
                    } else if b.len() == 1 {
                        let bv = b[0];
                        for (o, &av) in output.iter_mut().zip(a.iter()) {
                            *o = av / bv;
                        }
                    } else {
                        let c = b.len();
                        for (i, (o, &av)) in output.iter_mut().zip(a.iter()).enumerate() {
                            *o = av / b[i % c];
                        }
                    }
                }
                Dtype::F16 => {
                    let a = unsafe { (*a_ptr).as_f16() };
                    let b = unsafe { (*b_ptr).as_f16() };
                    let output = unsafe { (*out_ptr).as_f16_mut() };
                    
                    if a.len() == b.len() {
                        ops::div_fp16(output, a, b);
                    } else {
                        let c = b.len();
                        for (i, (o, av)) in output.iter_mut().zip(a.iter()).enumerate() {
                            *o = F16::from_f32(av.to_f32() / b[i % c].to_f32());
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for div".into())),
            }
            Ok(())
        }

        fn dispatch_concat(&mut self, op: &CompiledOp, axis: usize) -> Result<()> {
            let output_name = &op.outputs[0];
            
            // Gather input shapes and data
            let mut input_shapes: Vec<[usize; 4]> = Vec::new();
            let mut input_ptrs: Vec<*const CpuBuffer> = Vec::new();
            
            for input_name in &op.inputs {
                let shape = self.graph.shapes.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
                
                // Pad to 4D if needed
                let mut dims = [1usize; 4];
                let offset = 4 - shape.dims.len();
                for (i, &d) in shape.dims.iter().enumerate() {
                    dims[offset + i] = d;
                }
                input_shapes.push(dims);
                
                let buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                input_ptrs.push(buf as *const CpuBuffer);
            }

            let output_ptr = {
                let out_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                out_buf as *const CpuBuffer as *mut CpuBuffer
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let inputs: Vec<&[f32]> = input_ptrs.iter()
                        .map(|&ptr| unsafe { (*ptr).as_f32() })
                        .collect();
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::concat_f32(output, &inputs, &input_shapes, axis);
                }
                Dtype::F16 => {
                    let inputs: Vec<&[F16]> = input_ptrs.iter()
                        .map(|&ptr| unsafe { (*ptr).as_f16() })
                        .collect();
                    let output = unsafe { (*output_ptr).as_f16_mut() };
                    ops::concat_fp16(output, &inputs, &input_shapes, axis);
                }
                _ => return Err(Error::Runtime("unsupported dtype for concat".into())),
            }
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

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

            if input_shape.dims.len() != 4 {
                return Err(Error::Runtime("Resize input must be 4D".into()));
            }

            let n = input_shape.dims[0];
            let h_in = input_shape.dims[1];
            let w_in = input_shape.dims[2];
            let c = input_shape.dims[3];

            let resize_mode = match mode {
                "nearest" => ops::ResizeMode::Nearest,
                "linear" => ops::ResizeMode::Bilinear,
                _ => return Err(Error::Runtime(format!("unsupported resize mode: {mode}"))),
            };

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::resize_f32(output, input, n, h_in, w_in, out_h, out_w, c, resize_mode);
                }
                Dtype::F16 => {
                    // Convert to F32, resize, convert back
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };
                    
                    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
                    let mut output_f32 = vec![0.0f32; n * out_h * out_w * c];
                    ops::resize_f32(&mut output_f32, &input_f32, n, h_in, w_in, out_h, out_w, c, resize_mode);
                    
                    for (o, &v) in output.iter_mut().zip(output_f32.iter()) {
                        *o = F16::from_f32(v);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for resize".into())),
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
            
            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;
            
            // Pad to 4D
            let mut shape = [1usize; 4];
            let offset = 4 - input_shape.dims.len();
            for (i, &d) in input_shape.dims.iter().enumerate() {
                shape[offset + i] = d;
            }

            let input_ptr = {
                let buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                buf as *const CpuBuffer
            };

            // Get output buffer pointers
            let mut output_ptrs: Vec<*mut CpuBuffer> = Vec::new();
            for output_name in &op.outputs {
                let buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                output_ptrs.push(buf as *const CpuBuffer as *mut CpuBuffer);
            }

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let mut outputs: Vec<&mut [f32]> = output_ptrs.iter()
                        .map(|&ptr| unsafe { (*ptr).as_f32_mut() })
                        .collect();
                    let mut output_refs: Vec<&mut [f32]> = outputs.iter_mut()
                        .map(|s| &mut s[..])
                        .collect();
                    ops::split_f32(&mut output_refs, input, shape, axis, split_sizes);
                }
                Dtype::F16 => {
                    // Convert to F32, split, convert back
                    let input = unsafe { (*input_ptr).as_f16() };
                    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
                    
                    // Allocate F32 outputs
                    let mut output_f32s: Vec<Vec<f32>> = split_sizes.iter()
                        .map(|&size| {
                            let mut out_shape = shape;
                            out_shape[axis] = size;
                            vec![0.0f32; out_shape.iter().product()]
                        })
                        .collect();
                    
                    let mut output_refs: Vec<&mut [f32]> = output_f32s.iter_mut()
                        .map(|v| v.as_mut_slice())
                        .collect();
                    
                    ops::split_f32(&mut output_refs, &input_f32, shape, axis, split_sizes);
                    
                    // Convert back to F16
                    for (&ptr, f32_data) in output_ptrs.iter().zip(output_f32s.iter()) {
                        let output = unsafe { (*ptr).as_f16_mut() };
                        for (o, &v) in output.iter_mut().zip(f32_data.iter()) {
                            *o = F16::from_f32(v);
                        }
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for split".into())),
            }
            Ok(())
        }

        fn dispatch_transpose(&mut self, op: &CompiledOp, perm: &[usize]) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

            if input_shape.dims.len() != 4 {
                return Err(Error::Runtime("Transpose only supports 4D tensors".into()));
            }

            let shape: [usize; 4] = [
                input_shape.dims[0],
                input_shape.dims[1],
                input_shape.dims[2],
                input_shape.dims[3],
            ];
            let perm_arr: [usize; 4] = [perm[0], perm[1], perm[2], perm[3]];

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::transpose_f32(output, input, shape, perm_arr);
                }
                Dtype::F16 => {
                    // Convert to F32, transpose, convert back
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };
                    
                    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
                    let mut output_f32 = vec![0.0f32; input_f32.len()];
                    ops::transpose_f32(&mut output_f32, &input_f32, shape, perm_arr);
                    
                    for (o, &v) in output.iter_mut().zip(output_f32.iter()) {
                        *o = F16::from_f32(v);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for transpose".into())),
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

            let input_shape = self.graph.shapes.get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?;

            if input_shape.dims.len() != 4 {
                return Err(Error::Runtime("Slice only supports 4D tensors".into()));
            }

            let shape: [usize; 4] = [
                input_shape.dims[0],
                input_shape.dims[1],
                input_shape.dims[2],
                input_shape.dims[3],
            ];

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            match self.graph.dtype {
                Dtype::F32 => {
                    let input = unsafe { (*input_ptr).as_f32() };
                    let output = unsafe { (*output_ptr).as_f32_mut() };
                    ops::slice_f32(output, input, shape, starts, ends, axes, steps);
                }
                Dtype::F16 => {
                    // Convert to F32, slice, convert back
                    let input = unsafe { (*input_ptr).as_f16() };
                    let output = unsafe { (*output_ptr).as_f16_mut() };
                    
                    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
                    let mut output_f32 = vec![0.0f32; output.len()];
                    ops::slice_f32(&mut output_f32, &input_f32, shape, starts, ends, axes, steps);
                    
                    for (o, &v) in output.iter_mut().zip(output_f32.iter()) {
                        *o = F16::from_f32(v);
                    }
                }
                _ => return Err(Error::Runtime("unsupported dtype for slice".into())),
            }
            Ok(())
        }

        /// Get the input tensor names.
        pub fn inputs(&self) -> &[String] {
            &self.graph.inputs
        }

        /// Get the output tensor names.
        pub fn outputs(&self) -> &[String] {
            &self.graph.outputs
        }

        // =====================================================================
        // INT8 Quantized Op Dispatchers (Task 006)
        // =====================================================================

        fn dispatch_requantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            let n = self.graph.shapes.get(input_name)
                .map(|s| s.numel())
                .unwrap_or(0);

            // Get buffers
            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            // Input is INT32, output is INT8
            let input_i32 = unsafe { (*input_ptr).as_i32() };
            let output_i8 = unsafe { (*output_ptr).as_i8_mut() };

            ops::requantize_i32_to_i8(output_i8, input_i32, scale);
            Ok(())
        }

        fn dispatch_add_quantized(
            &mut self,
            op: &CompiledOp,
            scale_a_over_out: f32,
            scale_b_over_out: f32,
        ) -> Result<()> {
            let a_name = &op.inputs[0];
            let b_name = &op.inputs[1];
            let out_name = &op.outputs[0];

            // Get buffers
            let (a_ptr, b_ptr, out_ptr) = {
                let a_buf = self.buffers.get(a_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
                let b_buf = self.buffers.get(b_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
                let out_buf = self.buffers.get(out_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
                (
                    a_buf as *const CpuBuffer,
                    b_buf as *const CpuBuffer,
                    out_buf as *const CpuBuffer as *mut CpuBuffer,
                )
            };

            let a_i8 = unsafe { (*a_ptr).as_i8() };
            let b_i8 = unsafe { (*b_ptr).as_i8() };
            let out_i8 = unsafe { (*out_ptr).as_i8_mut() };

            ops::add_i8(out_i8, a_i8, b_i8, scale_a_over_out, scale_b_over_out);
            Ok(())
        }

        fn dispatch_quantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            // Get buffers
            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            let input_f32 = unsafe { (*input_ptr).as_f32() };
            let output_i8 = unsafe { (*output_ptr).as_i8_mut() };

            ops::quantize_f32_to_i8(output_i8, input_f32, scale);
            Ok(())
        }

        fn dispatch_dequantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];

            // Get buffers
            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };

            let input_i8 = unsafe { (*input_ptr).as_i8() };
            let output_f32 = unsafe { (*output_ptr).as_f32_mut() };

            ops::dequantize_i8_to_f32(output_f32, input_i8, scale);
            Ok(())
        }

        // ---- Per-channel QDQ (Task 009) ----

        fn dispatch_quantize_per_channel(
            &mut self,
            op: &CompiledOp,
            scales: &[f32],
            zero_points: &[i8],
            axis: usize,
        ) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];
            let dims = self
                .graph
                .shapes
                .get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?
                .dims
                .clone();

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };
            let input_f32 = unsafe { (*input_ptr).as_f32() };
            let output_i8 = unsafe { (*output_ptr).as_i8_mut() };

            quantize_per_channel_f32_to_i8(output_i8, input_f32, &dims, axis, scales, zero_points);
            Ok(())
        }

        fn dispatch_dequantize_per_channel(
            &mut self,
            op: &CompiledOp,
            scales: &[f32],
            zero_points: &[i8],
            axis: usize,
        ) -> Result<()> {
            let input_name = &op.inputs[0];
            let output_name = &op.outputs[0];
            let dims = self
                .graph
                .shapes
                .get(input_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {input_name}")))?
                .dims
                .clone();

            let (input_ptr, output_ptr) = {
                let input_buf = self.buffers.get(input_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {input_name}")))?;
                let output_buf = self.buffers.get(output_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {output_name}")))?;
                (input_buf as *const CpuBuffer, output_buf as *const CpuBuffer as *mut CpuBuffer)
            };
            let input_i8 = unsafe { (*input_ptr).as_i8() };
            let output_f32 = unsafe { (*output_ptr).as_f32_mut() };

            dequantize_per_channel_i8_to_f32(output_f32, input_i8, &dims, axis, scales, zero_points);
            Ok(())
        }
    }

    // ---- Per-channel QDQ kernels (Task 009) ----

    /// Quantize F32 → INT8 with one scale (and zero-point) per index along
    /// `axis`. Outer / inner strides are derived from `dims`.
    pub(super) fn quantize_per_channel_f32_to_i8(
        out: &mut [i8],
        input: &[f32],
        dims: &[usize],
        axis: usize,
        scales: &[f32],
        zero_points: &[i8],
    ) {
        assert_eq!(out.len(), input.len(), "per-channel quantize length mismatch");
        if dims.is_empty() {
            // Degenerate: treat as scalar with first scale.
            let s = scales[0];
            let zp = zero_points[0] as f32;
            let q = (input[0] / s + zp).round().clamp(-128.0, 127.0) as i8;
            out[0] = q;
            return;
        }
        let axis = axis.min(dims.len() - 1);
        let channels = dims[axis];
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product();
        assert_eq!(scales.len(), channels, "per-channel scale count mismatch");
        assert_eq!(zero_points.len(), channels, "per-channel zp count mismatch");

        for o in 0..outer {
            for c in 0..channels {
                let s = scales[c];
                let zp = zero_points[c] as f32;
                let base = (o * channels + c) * inner;
                for i in 0..inner {
                    let v = input[base + i] / s + zp;
                    out[base + i] = v.round().clamp(-128.0, 127.0) as i8;
                }
            }
        }
    }

    /// Inverse of `quantize_per_channel_f32_to_i8`.
    pub(super) fn dequantize_per_channel_i8_to_f32(
        out: &mut [f32],
        input: &[i8],
        dims: &[usize],
        axis: usize,
        scales: &[f32],
        zero_points: &[i8],
    ) {
        assert_eq!(out.len(), input.len(), "per-channel dequantize length mismatch");
        if dims.is_empty() {
            let s = scales[0];
            let zp = zero_points[0] as f32;
            out[0] = (input[0] as f32 - zp) * s;
            return;
        }
        let axis = axis.min(dims.len() - 1);
        let channels = dims[axis];
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product();
        assert_eq!(scales.len(), channels);
        assert_eq!(zero_points.len(), channels);

        for o in 0..outer {
            for c in 0..channels {
                let s = scales[c];
                let zp = zero_points[c] as f32;
                let base = (o * channels + c) * inner;
                for i in 0..inner {
                    out[base + i] = (input[base + i] as f32 - zp) * s;
                }
            }
        }
    }

    #[cfg(test)]
    mod per_channel_tests {
        use super::*;

        #[test]
        fn quantize_per_channel_applies_per_channel_scale() {
            // 1x4 (C=4): each channel gets a different scale.
            // dims=[1,4], axis=1 → outer=1, channels=4, inner=1.
            let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let scales = vec![0.5, 1.0, 1.5, 2.0]; // q = round(x / s)
            let zps = vec![0i8; 4];
            let mut out = vec![0i8; 4];

            quantize_per_channel_f32_to_i8(&mut out, &input, &[1, 4], 1, &scales, &zps);

            // Expected: [round(1/.5)=2, round(2/1)=2, round(3/1.5)=2, round(4/2)=2]
            assert_eq!(out, vec![2, 2, 2, 2]);
        }

        #[test]
        fn quantize_dequantize_round_trip_per_channel() {
            // [N=1, C=3, H=2, W=2] tensor, axis=1.
            let dims = vec![1usize, 3, 2, 2];
            let numel: usize = dims.iter().product();
            let input: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1).collect();
            let scales = vec![0.05, 0.1, 0.15];
            let zps = vec![0i8; 3];

            let mut q = vec![0i8; numel];
            quantize_per_channel_f32_to_i8(&mut q, &input, &dims, 1, &scales, &zps);

            let mut dq = vec![0f32; numel];
            dequantize_per_channel_i8_to_f32(&mut dq, &q, &dims, 1, &scales, &zps);

            // Round-trip error should be bounded by largest scale / 2.
            let max_err = scales.iter().cloned().fold(0.0, f32::max) / 2.0 + 1e-5;
            for (orig, recon) in input.iter().zip(dq.iter()) {
                let err = (orig - recon).abs();
                assert!(
                    err <= max_err,
                    "round-trip error {err} exceeds bound {max_err} (orig={orig}, recon={recon})"
                );
            }
        }

        #[test]
        fn quantize_per_channel_axis_zero() {
            // [C=2, H=3] tensor, axis=0 (per-row scaling).
            let dims = vec![2usize, 3];
            let input: Vec<f32> = vec![
                1.0, 2.0, 3.0,    // channel 0, scale=0.5 → q=2,4,6
                10.0, 20.0, 30.0, // channel 1, scale=10  → q=1,2,3
            ];
            let scales = vec![0.5, 10.0];
            let zps = vec![0i8; 2];
            let mut out = vec![0i8; 6];

            quantize_per_channel_f32_to_i8(&mut out, &input, &dims, 0, &scales, &zps);
            assert_eq!(out, vec![2, 4, 6, 1, 2, 3]);
        }

        #[test]
        fn quantize_per_channel_saturates() {
            // Overflow values clamp to i8 range.
            let scales = vec![0.01f32];
            let zps = vec![0i8];
            let input = vec![10.0f32, -10.0, 0.0];
            let mut out = vec![0i8; 3];

            quantize_per_channel_f32_to_i8(&mut out, &input, &[1, 3], 0, &scales, &zps);
            assert_eq!(out, vec![127, -128, 0]);
        }
    }
}

/// Re-export the CPU-optimized runtime when the `cpu` feature is enabled.
#[cfg(feature = "cpu")]
pub use self::cpu_runtime::CpuGraphRuntime;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_conversion() {
        let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = f32_data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        let f16_bytes = convert_f32_to_f16(&bytes);
        assert_eq!(f16_bytes.len(), 8); // 4 f16 values = 8 bytes
        
        let back = convert_f16_to_f32(&f16_bytes);
        let result: Vec<f32> = back.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        for (orig, converted) in f32_data.iter().zip(result.iter()) {
            assert!((orig - converted).abs() < 0.01);
        }
    }

    #[cfg(feature = "cpu")]
    mod cpu_tests {
        use super::*;
        use crate::graph::{compile_model, convert_nchw_to_nhwc, transpose_nchw_to_nhwc};
        use dragonwing_core::Dtype;

        #[test]
        #[ignore] // Requires artifacts/models/mobilenetv2-12.onnx
        fn test_mobilenetv2_inference() {
            // Load model
            let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.onnx");
            let bytes = std::fs::read(model_path).expect("Failed to read model file");
            let model = crate::model::parse_model(&bytes).expect("Failed to parse model");
            
            // Compile and convert to NHWC
            let mut graph = compile_model(&model, Dtype::F32).expect("Failed to compile model");
            convert_nchw_to_nhwc(&mut graph).expect("Failed to convert to NHWC");
            
            println!("Graph compiled:");
            println!("  Inputs: {:?}", graph.input_info());
            println!("  Outputs: {:?}", graph.output_info());
            
            // Create runtime
            let mut runtime = CpuGraphRuntime::new(graph).expect("Failed to create runtime");
            
            // Load input (NCHW format from file)
            let input_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.input.bin");
            let input_bytes = std::fs::read(input_path).expect("Failed to read input file");
            let input_nchw: Vec<f32> = input_bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            
            // Convert input to NHWC
            let input_nhwc = transpose_nchw_to_nhwc(&input_nchw, &[1, 3, 224, 224]);
            
            println!("Input loaded: {} elements", input_nhwc.len());
            println!("  First 5 values: {:?}", &input_nhwc[..5.min(input_nhwc.len())]);
            
            // Set input
            runtime.set_input_f32("input", &input_nhwc).expect("Failed to set input");
            
            // Run inference
            println!("Running inference...");
            let start = std::time::Instant::now();
            runtime.run().expect("Inference failed");
            let elapsed = start.elapsed();
            println!("Inference completed in {:?}", elapsed);
            
            // Get output
            let output = runtime.get_output_f32("output").expect("Failed to get output");
            println!("Output shape: {} elements", output.len());
            
            // Find top-5 predictions
            let mut indexed: Vec<(usize, f32)> = output.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
            
            println!("\nTop-5 predictions:");
            for (i, (class_id, logit)) in top5.iter().enumerate() {
                println!("  {}. Class {}: logit={:.4}", i + 1, class_id, logit);
            }
            
            // Load reference
            let ref_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.reference.json");
            let ref_json = std::fs::read_to_string(ref_path).expect("Failed to read reference");
            
            // Parse expected top-5 classes (simple JSON parsing)
            // Expected: "top5_classes": [549, 418, 645, 954, 818]
            let expected_classes: Vec<usize> = vec![549, 418, 645, 954, 818];
            
            println!("\nExpected top-5 classes: {:?}", expected_classes);
            println!("Actual top-5 classes: {:?}", top5.iter().map(|(c, _)| c).collect::<Vec<_>>());
            
            // Check if top-1 matches
            assert_eq!(top5[0].0, expected_classes[0], 
                "Top-1 class mismatch: expected {}, got {}", expected_classes[0], top5[0].0);
            
            println!("\n✓ Top-1 class matches reference!");
        }
    }
}
