//! Vulkan-backed graph runtime (Task 007).
//!
//! This module provides `VulkanGraphRuntime`, which executes compiled ONNX
//! graphs on the Vulkan compute backend. Unlike the generic `GraphRuntime`
//! (which falls back to host-side compute), this runtime dispatches every
//! op directly to a `dragonwing_vulkan::ops::*` wrapper.
//!
//! # Architecture
//!
//! ```text
//!   VulkanGraphRuntime
//!     ├── Graph                  (compiled, dtype-validated)
//!     ├── VulkanBackend          (Arc<Context> + Arc<PipelineCache>)
//!     ├── HashMap<String,        VulkanBuffer per tensor (allocated up-front)
//!     │       VulkanBuffer>
//!     └── dispatch_op() match    routes CompiledOp → dragonwing_vulkan::ops::*
//! ```
//!
//! # Buffer lifecycle
//!
//! All tensors (inputs, intermediates, outputs, initializers) get one
//! `VulkanBuffer` at `new()` time. Initializers are uploaded once. Inputs are
//! re-uploaded via `set_input_f32()`. Outputs are read back via
//! `get_output_f32()`. Intermediates are written/read by the GPU only — they
//! are never touched on the host between `run()` calls.
//!
//! # Synchronisation
//!
//! Each op currently records its own one-shot command buffer + submit + wait
//! (mirroring the existing `dragonwing_vulkan::ops::*` API). The `run()`
//! method waits for the queue to go idle after the last op so that
//! `get_output_*()` reads coherent data. Phase 6 may move to a single batched
//! command buffer per graph if measurements show submit overhead is high.
//!
//! # Quickstart
//!
//! ```ignore
//! use dragonwing_onnx::{compile_model, VulkanGraphRuntime};
//! use dragonwing_vulkan::{VulkanBackend, VulkanConfig};
//! use dragonwing_core::Dtype;
//!
//! let model = dragonwing_onnx::load_model("model.onnx")?;
//! let graph = compile_model(&model, Dtype::F32)?;
//! let backend = VulkanBackend::new(VulkanConfig::default())?;
//! let mut rt = VulkanGraphRuntime::new(graph, backend)?;
//!
//! rt.set_input_f32("input", &input_data)?;
//! rt.run()?;
//! let out = rt.get_output_f32("output")?;
//! ```

use crate::builder::{CompiledOp, OpParams, TensorShape};
use crate::error::{Error, Result};
use crate::graph::Graph;
use dragonwing_core::{Backend, BackendBuffer, BufferKind, Dtype};
use dragonwing_vulkan::{SlabAllocator, VulkanBackend, VulkanBuffer};
use std::collections::HashMap;
use std::sync::Arc;

/// Graph runtime that dispatches every op to the Vulkan compute backend.
///
/// Phase 1 implementation: skeleton, buffer management, and op-dispatch
/// scaffolding. Most op dispatchers are stubs that return
/// `Error::Runtime("not yet implemented")` and will be filled in in
/// subsequent phases:
///
/// * **Phase 2** wires up the F32 ops that already have Vulkan shaders
///   (`gemm_f32`, `conv2d_f32_nhwc`, `relu_f32`, `add_f32`, `maxpool2d_f32`,
///   `softmax_f32`, `sigmoid_f32`, `mul_f32`).
/// * **Phase 3** wires up the INT8 packed ops.
/// * **Phase 4** wires up fused kernels.
pub struct VulkanGraphRuntime {
    /// The compiled graph.
    graph: Graph,
    /// Vulkan backend (cheap to clone — internally `Arc`).
    backend: VulkanBackend,
    /// Tensor-name → `VulkanBuffer`. Allocated up-front in `new()`.
    buffers: HashMap<String, VulkanBuffer>,
    /// Slab allocator backing every tensor buffer. Holds the shared
    /// `VkDeviceMemory` allocations that buffers reference.
    ///
    /// **Drop order matters**: `buffers` must be dropped before `slab`
    /// so the slab-owned `VkDeviceMemory` outlives every buffer that
    /// points into it. Rust drops struct fields in declaration order,
    /// so `buffers` is declared above `slab` deliberately.
    #[allow(dead_code)]
    slab: Option<Arc<SlabAllocator>>,
    /// Stats: number of `vkAllocateMemory` calls issued for this runtime
    /// (i.e. slab count). Useful for Phase 5 verification.
    slab_count_at_init: usize,
}

impl std::fmt::Debug for VulkanGraphRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanGraphRuntime")
            .field("device", &self.backend.device_name())
            .field("op_count", &self.graph.ops.len())
            .field("buffer_count", &self.buffers.len())
            .finish()
    }
}

impl VulkanGraphRuntime {
    /// Allocate a Vulkan buffer for every tensor in `graph.shapes` and upload
    /// all initializers (weights, biases).
    ///
    /// Buffer sizes use `TensorShape::size_bytes()`, which accounts for the
    /// graph's dtype (4 B for F32, 2 B for F16, 1 B for I8, 4 B for I32).
    ///
    /// # Errors
    ///
    /// * `Error::Runtime` if any individual `backend.alloc()` fails.
    /// * `Error::Runtime` if an initializer's byte length does not match the
    ///   allocated buffer (typically indicates a build-context bug).
    pub fn new(graph: Graph, backend: VulkanBackend) -> Result<Self> {
        // Task 008 Phase 5: pack all tensor buffers into a shared slab
        // allocator. Total memory required is the sum of every tensor's
        // aligned size, plus per-buffer alignment headroom. We aim for
        // <10 vkAllocateMemory calls for YOLOv8n (~100 tensors).
        //
        // Disable the slab path by setting `DRAGONWING_DISABLE_SLAB=1`.
        // The per-buffer `vkAllocateMemory` path is slower at init but
        // is sometimes useful for isolating GPU-side faults that are
        // suspected to be slab-binding bugs.
        let disable_slab = std::env::var("DRAGONWING_DISABLE_SLAB").is_ok();

        // First pass: compute total bytes to allocate.
        let total_bytes: usize = graph
            .shapes
            .iter()
            .filter_map(|(_, s)| {
                let sz = s.size_bytes();
                if sz == 0 { None } else { Some(sz) }
            })
            .map(|sz| (sz + 255) & !255usize) // +alignment slack
            .sum();

        // Pick a slab size: max(default 16 MiB, total + headroom).
        // Headroom = 4 MiB to absorb fragmentation; if the model is small
        // enough we still only allocate one slab.
        const HEADROOM: usize = 4 * 1024 * 1024;
        let slab_size = (total_bytes + HEADROOM).max(16 * 1024 * 1024);

        let slab = if disable_slab {
            None
        } else {
            // Discover the right memory type once.
            let mem_type = backend
                .find_storage_memory_type()
                .map_err(|e| Error::Runtime(format!("find_storage_memory_type: {e}")))?;
            Some(Arc::new(SlabAllocator::new(
                backend.context().clone(),
                mem_type,
                Some(slab_size),
            )))
        };

        let mut buffers: HashMap<String, VulkanBuffer> = HashMap::new();

        for (name, shape) in &graph.shapes {
            let size_bytes = shape.size_bytes();
            if size_bytes == 0 {
                continue;
            }
            let buffer = if let Some(slab_ref) = &slab {
                let (buffer, _alloc) = backend
                    .alloc_slab_buffer(slab_ref, size_bytes)
                    .map_err(|e| Error::Runtime(format!("Vulkan slab alloc {name} failed: {e}")))?;
                buffer
            } else {
                backend
                    .alloc(size_bytes, BufferKind::Storage)
                    .map_err(|e| Error::Runtime(format!("Vulkan alloc {name} failed: {e}")))?
            };
            buffers.insert(name.clone(), buffer);
        }

        // Upload initializers (weights, biases, etc.).
        for (name, data) in &graph.initializers {
            if let Some(buffer) = buffers.get_mut(name) {
                if buffer.len_bytes() == data.len() {
                    backend
                        .upload(buffer, data)
                        .map_err(|e| Error::Runtime(format!("Vulkan upload {name}: {e}")))?;
                }
            }
        }

        let slab_count_at_init = slab.as_ref().map(|s| s.stats().num_slabs).unwrap_or(0);

        Ok(Self {
            graph,
            backend,
            buffers,
            slab,
            slab_count_at_init,
        })
    }

    /// Number of `vkAllocateMemory` calls issued by the slab allocator
    /// at the time of construction. Phase 5 acceptance criterion:
    /// `<10` for YOLOv8n.
    pub fn slab_count(&self) -> usize {
        self.slab_count_at_init
    }

    /// Slab allocator statistics (if a slab is in use).
    pub fn slab_stats(&self) -> Option<dragonwing_vulkan::SlabStats> {
        self.slab.as_ref().map(|s| s.stats())
    }

    /// Input tensor names + shapes (for caller introspection).
    pub fn inputs(&self) -> Vec<(&str, &TensorShape)> {
        self.graph.input_info()
    }

    /// Output tensor names + shapes.
    pub fn outputs(&self) -> Vec<(&str, &TensorShape)> {
        self.graph.output_info()
    }

    /// Borrow the underlying backend (cheap clone if needed).
    pub fn backend(&self) -> &VulkanBackend {
        &self.backend
    }

    /// Borrow the compiled graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Upload F32 data into the named input buffer.
    ///
    /// For graphs compiled in F32, this is a direct memcpy. For F16 graphs we
    /// convert via `F16::from_f32`. INT8 graphs require quantization at the
    /// caller side (see Phase 3 wiring); for now `set_input_f32` returns an
    /// error if the graph dtype is `I8`.
    pub fn set_input_f32(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(name)
            .ok_or_else(|| Error::Runtime(format!("input not found: {name}")))?;

        match self.graph.dtype {
            Dtype::F32 => {
                let expected = data.len() * 4;
                if buffer.len_bytes() != expected {
                    return Err(Error::Runtime(format!(
                        "input {name} size mismatch: buffer={} bytes, data={} bytes",
                        buffer.len_bytes(),
                        expected,
                    )));
                }
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), expected)
                };
                self.backend
                    .upload(buffer, bytes)
                    .map_err(|e| Error::Runtime(format!("upload {name}: {e}")))
            }
            Dtype::F16 => {
                use dragonwing_core::F16;
                let expected = data.len() * 2;
                if buffer.len_bytes() != expected {
                    return Err(Error::Runtime(format!(
                        "input {name} size mismatch: buffer={} bytes, data={} bytes",
                        buffer.len_bytes(),
                        expected,
                    )));
                }
                let f16_data: Vec<F16> = data.iter().map(|&v| F16::from_f32(v)).collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f16_data.as_ptr().cast::<u8>(), expected)
                };
                self.backend
                    .upload(buffer, bytes)
                    .map_err(|e| Error::Runtime(format!("upload {name}: {e}")))
            }
            Dtype::I8 => Err(Error::Runtime(
                "set_input_f32 on I8 graph requires explicit Quantize op or set_input_i8 \
                 (see Phase 3)"
                    .into(),
            )),
            _ => Err(Error::Runtime(format!(
                "set_input_f32: unsupported graph dtype {:?}",
                self.graph.dtype
            ))),
        }
    }

    /// Download the named output buffer as F32.
    ///
    /// Implicitly synchronises the backend (waits for the last submission to
    /// complete) so the readback observes the result of the most recent
    /// `run()`.
    pub fn get_output_f32(&self, name: &str) -> Result<Vec<f32>> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("output not found: {name}")))?;

        // `download` calls `synchronize` internally — see `VulkanBackend::download`.
        match self.graph.dtype {
            Dtype::F32 => {
                let n = buffer.len_bytes() / 4;
                let mut bytes = vec![0u8; buffer.len_bytes()];
                self.backend
                    .download(buffer, &mut bytes)
                    .map_err(|e| Error::Runtime(format!("download {name}: {e}")))?;
                let mut out = vec![0.0f32; n];
                for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(out)
            }
            Dtype::F16 => {
                use dragonwing_core::F16;
                let n = buffer.len_bytes() / 2;
                let mut bytes = vec![0u8; buffer.len_bytes()];
                self.backend
                    .download(buffer, &mut bytes)
                    .map_err(|e| Error::Runtime(format!("download {name}: {e}")))?;
                let mut out = vec![0.0f32; n];
                for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out[i] = F16(bits).to_f32();
                }
                Ok(out)
            }
            Dtype::I8 => Err(Error::Runtime(
                "get_output_f32 on I8 graph requires explicit Dequantize op or get_output_i8 \
                 (see Phase 3)"
                    .into(),
            )),
            _ => Err(Error::Runtime(format!(
                "get_output_f32: unsupported graph dtype {:?}",
                self.graph.dtype
            ))),
        }
    }

    /// Execute all ops in graph order.
    ///
    /// Task 008 Phase 4: Ops with native Vulkan shaders are recorded into
    /// a **single command buffer** via [`dragonwing_vulkan::OpsRecorder`],
    /// with `cmd_pipeline_barrier` inserted between dependent ops. The
    /// batch is submitted once, drastically reducing per-op
    /// `vkQueueSubmit` overhead.
    ///
    /// Task 008 Phase 6: Ops that don't have Vulkan shaders yet (Sub,
    /// Div, Concat, Resize, Split, Transpose, Slice) **flush** the
    /// current recorder, run a CPU fallback (`download → compute →
    /// upload`), and then a new recorder is started for the next run of
    /// GPU ops. This lets the full YOLOv8 graph run end-to-end without
    /// blocking on shader implementation of the layout-shuffling ops.
    ///
    /// The previous strictly-per-op submit path remains available as
    /// [`Self::run_unbatched`].
    pub fn run(&mut self) -> Result<()> {
        let ops: Vec<CompiledOp> = self.graph.ops.clone();
        let mut recorder_opt: Option<dragonwing_vulkan::OpsRecorder> = None;

        for (i, op) in ops.iter().enumerate() {
            let cpu_fallback = matches!(
                &op.params,
                OpParams::Sub
                    | OpParams::Div
                    | OpParams::Concat { .. }
                    | OpParams::Resize { .. }
                    | OpParams::Split { .. }
                    | OpParams::Transpose { .. }
                    | OpParams::Slice { .. }
            );

            if cpu_fallback {
                // Flush any pending GPU work first so the CPU dispatch
                // reads coherent data.
                if let Some(rec) = recorder_opt.take() {
                    rec.finish_and_submit()
                        .map_err(|e| Error::Runtime(format!("recorder flush: {e}")))?;
                    self.backend
                        .synchronize()
                        .map_err(|e| Error::Runtime(format!("flush synchronize: {e}")))?;
                }
                // Dispatch on host. The CPU fallback paths
                // (dispatch_cpu_*) themselves take care of download +
                // upload around the actual compute.
                self.dispatch_op(op)?;
                continue;
            }

            // GPU op — open a recorder lazily on demand.
            let rec = match recorder_opt.as_mut() {
                Some(r) => r,
                None => {
                    let r = dragonwing_vulkan::OpsRecorder::begin(&self.backend)
                        .map_err(|e| Error::Runtime(format!("recorder begin: {e}")))?;
                    recorder_opt = Some(r);
                    recorder_opt.as_mut().unwrap()
                }
            };

            if i > 0 && Self::needs_barrier(&ops, i) {
                rec.record_memory_barrier();
            }
            self.record_op(rec, op)?;
        }

        // Submit anything left in the current recorder.
        if let Some(rec) = recorder_opt.take() {
            rec.finish_and_submit()
                .map_err(|e| Error::Runtime(format!("recorder submit: {e}")))?;
        }

        self.backend
            .synchronize()
            .map_err(|e| Error::Runtime(format!("synchronize: {e}")))?;
        Ok(())
    }

    /// Legacy per-op submit path — one `OneShot` per op. Used for tests
    /// that exercise individual ops via `dispatch_op()`.
    pub fn run_unbatched(&mut self) -> Result<()> {
        let ops: Vec<CompiledOp> = self.graph.ops.clone();
        let trace = std::env::var("DRAGONWING_TRACE").is_ok();
        for (i, op) in ops.iter().enumerate() {
            if trace {
                let in_shapes: Vec<String> = op
                    .inputs
                    .iter()
                    .map(|n| {
                        self.graph
                            .shapes
                            .get(n)
                            .map(|s| format!("{n}{:?}", s.dims))
                            .unwrap_or_else(|| format!("{n}(?)"))
                    })
                    .collect();
                let out_shapes: Vec<String> = op
                    .outputs
                    .iter()
                    .map(|n| {
                        self.graph
                            .shapes
                            .get(n)
                            .map(|s| format!("{n}{:?}", s.dims))
                            .unwrap_or_else(|| format!("{n}(?)"))
                    })
                    .collect();
                eprintln!(
                    "[op {i:3}] {} ({}) in={:?} out={:?}",
                    op.name, op.op_type, in_shapes, out_shapes
                );
            }
            self.dispatch_op(op).map_err(|e| {
                Error::Runtime(format!(
                    "op[{i}] {} ({}) failed: {e}",
                    op.name, op.op_type
                ))
            })?;
            if trace {
                self.backend.synchronize().ok();
            }
        }
        self.backend
            .synchronize()
            .map_err(|e| Error::Runtime(format!("synchronize: {e}")))?;
        Ok(())
    }

    /// Decide if a barrier is needed before op `idx`.
    ///
    /// We insert a barrier when any of `ops[idx].inputs` was written by
    /// `ops[idx - 1]` (the immediately preceding op). This catches the
    /// common case where the runtime emits a strictly sequential chain
    /// `A → B → C` with no parallelism between dispatches.
    ///
    /// Two simplifying assumptions:
    /// 1. **Single-producer ordering.** Graph ops appear in topological
    ///    order; producers always run before consumers. So we only need
    ///    to check the immediate predecessor.
    /// 2. **Conservative over-barrier.** When in doubt we insert a
    ///    barrier — it's cheaper than missing one (which causes silent
    ///    data corruption on Turnip / Adreno A702).
    fn needs_barrier(ops: &[CompiledOp], idx: usize) -> bool {
        let prev = &ops[idx - 1];
        let cur = &ops[idx];
        cur.inputs.iter().any(|inp| prev.outputs.contains(inp))
    }

    /// Get two buffers (one shared, one mut) from the `buffers` map without
    /// borrowing `self` entirely. Used by dispatchers that also need
    /// `&self.backend`.
    ///
    /// Borrows `buffers: &mut HashMap<String, VulkanBuffer>` so it
    /// doesn't conflict with shared borrows of other `self` fields.
    fn get_two_buffers<'a>(
        buffers: &'a mut HashMap<String, VulkanBuffer>,
        a_name: &str,
        b_name: &str,
    ) -> Result<(&'a VulkanBuffer, &'a mut VulkanBuffer)> {
        if a_name == b_name {
            return Err(Error::Runtime(format!(
                "aliasing not supported here: {a_name}"
            )));
        }
        // SAFETY: a_name != b_name (checked above) so the two raw pointers
        // refer to disjoint entries. We split the &mut HashMap borrow.
        let a_ptr: *const VulkanBuffer = buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let b_ptr: *mut VulkanBuffer = buffers
            .get_mut(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        unsafe { Ok((&*a_ptr, &mut *b_ptr)) }
    }

    /// Three buffers: two shared inputs + one mut output.
    fn get_three_buffers<'a>(
        buffers: &'a mut HashMap<String, VulkanBuffer>,
        a_name: &str,
        b_name: &str,
        c_name: &str,
    ) -> Result<(&'a VulkanBuffer, &'a VulkanBuffer, &'a mut VulkanBuffer)> {
        if a_name == c_name || b_name == c_name {
            return Err(Error::Runtime(format!(
                "output {c_name} aliases an input"
            )));
        }
        let a_ptr: *const VulkanBuffer = buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let b_ptr: *const VulkanBuffer = buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        let c_ptr: *mut VulkanBuffer = buffers
            .get_mut(c_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {c_name}")))?;
        unsafe { Ok((&*a_ptr, &*b_ptr, &mut *c_ptr)) }
    }

    /// Dispatch a single op to the Vulkan backend.
    ///
    /// In Phase 1 most arms are stubs that return `Error::Runtime("not yet
    /// implemented for Vulkan")`. Subsequent phases fill them in.
    fn dispatch_op(&mut self, op: &CompiledOp) -> Result<()> {
        match &op.params {
            // ----- metadata-only / no-op ----------------------------------
            OpParams::None => match op.op_type.as_str() {
                // "Relu" with OpParams::None happens for in-place relus that
                // didn't get folded into a Clip; treat as no-op for now —
                // Phase 2 will route to relu_f32.
                "Flatten" | "Reshape" => self.dispatch_reshape(op),
                "Relu" => self.dispatch_relu(op),
                // Fused SiLU op produced by the task 005 fusion pass.
                // Runs as a single shader instead of separate sigmoid + mul.
                "SiLU" => self.dispatch_silu(op),
                _ => Ok(()),
            },
            OpParams::Reshape { .. } => self.dispatch_reshape(op),

            // ----- F32 element-wise (Phase 2) -----------------------------
            OpParams::Add => self.dispatch_add(op),
            OpParams::Mul => self.dispatch_mul(op),
            OpParams::Sigmoid => self.dispatch_sigmoid(op),
            OpParams::Clip { min, max } => self.dispatch_clip(op, *min, *max),

            // ----- F32 GEMM / Conv (Phase 2) ------------------------------
            OpParams::Gemm {
                trans_a, trans_b, ..
            } => self.dispatch_gemm(op, *trans_a, *trans_b),
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
            OpParams::Softmax { axis } => self.dispatch_softmax(op, *axis),

            // ----- INT8 ops (Phase 3) ------------------------------------
            OpParams::Requantize { scale } => self.dispatch_requantize(op, *scale),
            OpParams::AddQuantized {
                scale_a_over_out,
                scale_b_over_out,
            } => self.dispatch_add_quantized(op, *scale_a_over_out, *scale_b_over_out),
            OpParams::Quantize { scale } => self.dispatch_quantize(op, *scale),
            OpParams::Dequantize { scale } => self.dispatch_dequantize(op, *scale),
            // ----- Per-channel QDQ (Task 009) ---------------------------
            // No native Vulkan shader; the QDQ-fold pass should strip these
            // before runtime. Falling back via the CPU path is possible but
            // expensive; for now emit a clear error so the user knows the
            // fold pass either did not run or could not handle the model.
            OpParams::QuantizePerChannel { .. } | OpParams::DequantizePerChannel { .. } => {
                Err(Error::Runtime(format!(
                    "{} (per-channel QDQ) reached VulkanGraphRuntime; \
                     ensure fold_qdq_patterns() was applied before runtime construction",
                    op.op_type
                )))
            }

            // ----- Fused INT8 ops (Phase 1 of Task 008) ------------------
            OpParams::Conv2dRequantReluI8Nhwc {
                kernel_shape,
                strides,
                pads,
                group,
                requant_scale,
                has_relu,
                ..
            } => self.dispatch_conv2d_requant_relu_i8(
                op,
                *kernel_shape,
                *strides,
                *pads,
                *group,
                *requant_scale,
                *has_relu,
            ),

            // ----- ops without Vulkan shaders yet — CPU fallback ---------
            // We download inputs to host, run a CPU op, and upload the
            // result. This kills batched-recorder performance for the
            // affected ops but lets the full YOLO graph run end-to-end.
            // Task 009 should add native Vulkan shaders for these.
            OpParams::Sub => self.dispatch_cpu_sub(op),
            OpParams::Div => self.dispatch_cpu_div(op),
            OpParams::Concat { axis } => self.dispatch_cpu_concat(op, *axis),
            OpParams::Resize { out_h, out_w, mode } => {
                self.dispatch_cpu_resize(op, *out_h, *out_w, mode)
            }
            OpParams::Split { axis, split_sizes } => {
                self.dispatch_cpu_split(op, *axis, split_sizes)
            }
            OpParams::Transpose { perm } => self.dispatch_cpu_transpose(op, perm),
            OpParams::Slice {
                starts,
                ends,
                axes,
                steps,
            } => self.dispatch_cpu_slice(op, starts, ends, axes, steps),
            OpParams::GlobalAvgPool | OpParams::AvgPool { .. } => Err(Error::Runtime(format!(
                "op {:?} not yet supported by VulkanGraphRuntime",
                op.op_type
            ))),
        }
    }

    // =========================================================================
    // Task 008 Phase 4 — record_op: parallel to dispatch_op, but records into
    // a shared OpsRecorder instead of submitting per op. Used by the new
    // run() to batch the whole graph into a single command buffer.
    //
    // Shape resolution + buffer lookup logic is shared with dispatch_op via
    // helpers; only the final "send dispatch" call differs.
    // =========================================================================

    fn record_op(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
    ) -> Result<()> {
        match &op.params {
            OpParams::None => match op.op_type.as_str() {
                "Flatten" | "Reshape" => self.record_reshape(rec, op),
                "Relu" => self.record_relu(rec, op),
                "SiLU" => self.record_silu(rec, op),
                _ => Ok(()),
            },
            OpParams::Reshape { .. } => self.record_reshape(rec, op),
            OpParams::Add => self.record_add(rec, op),
            OpParams::Mul => self.record_mul(rec, op),
            OpParams::Sigmoid => self.record_sigmoid(rec, op),
            OpParams::Clip { min, max } => {
                if *min == 0.0 && max.is_infinite() {
                    self.record_relu(rec, op)
                } else {
                    Err(Error::Runtime(
                        "vk record_clip: general clip not supported".into(),
                    ))
                }
            }
            OpParams::Gemm {
                trans_a, trans_b, ..
            } => self.record_gemm(rec, op, *trans_a, *trans_b),
            OpParams::Conv2d {
                kernel_shape,
                strides,
                pads,
                group,
                ..
            } => self.record_conv2d(rec, op, *kernel_shape, *strides, *pads, *group),
            OpParams::MaxPool {
                kernel_shape,
                strides,
                ..
            } => self.record_maxpool(rec, op, *kernel_shape, *strides),
            OpParams::Softmax { axis } => self.record_softmax(rec, op, *axis),
            OpParams::Requantize { scale } => self.record_requantize(rec, op, *scale),
            OpParams::AddQuantized {
                scale_a_over_out,
                scale_b_over_out,
            } => self.record_add_quantized(rec, op, *scale_a_over_out, *scale_b_over_out),
            OpParams::Quantize { scale } => self.record_quantize(rec, op, *scale),
            OpParams::Dequantize { scale } => self.record_dequantize(rec, op, *scale),
            OpParams::Conv2dRequantReluI8Nhwc {
                kernel_shape,
                strides,
                pads,
                group,
                requant_scale,
                has_relu,
                ..
            } => self.record_conv2d_requant_relu_i8(
                rec,
                op,
                *kernel_shape,
                *strides,
                *pads,
                *group,
                *requant_scale,
                *has_relu,
            ),
            OpParams::Sub
            | OpParams::Div
            | OpParams::GlobalAvgPool
            | OpParams::AvgPool { .. }
            | OpParams::Concat { .. }
            | OpParams::Resize { .. }
            | OpParams::Split { .. }
            | OpParams::Transpose { .. }
            | OpParams::Slice { .. } => Err(Error::Runtime(format!(
                "op {:?} not yet supported by VulkanGraphRuntime (record path)",
                op.op_type
            ))),
            // Per-channel QDQ should be folded away before the runtime sees them.
            OpParams::QuantizePerChannel { .. } | OpParams::DequantizePerChannel { .. } => {
                Err(Error::Runtime(format!(
                    "{} (per-channel QDQ) reached VulkanGraphRuntime (record path); \
                     run fold_qdq_patterns() before constructing the runtime",
                    op.op_type
                )))
            }
        }
    }

    // -------------------------------------------------------------------------
    // record_* implementations — thin wrappers that mirror dispatch_* but
    // call rec.record_* instead of dragonwing_vulkan::ops::*.
    // -------------------------------------------------------------------------

    fn record_reshape(
        &mut self,
        _rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
    ) -> Result<()> {
        // Reshape is a no-op when in==out names. When they differ, copy
        // via the GPU. For now, mirror the dispatch path's host bounce.
        // (TODO: switch to vkCmdCopyBuffer once we have a record_copy_buffer.)
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Ok(());
        }
        if op.inputs[0] == op.outputs[0] {
            return Ok(());
        }
        // Host-side bounce: this forces a partial sync (download requires the
        // batch be empty). To avoid breaking the batched recorder semantics,
        // we leave Reshape unsupported in the batched path for now. Real-world
        // ONNX models typically have Reshape only at the boundaries where
        // shape-changes happen between inference stages.
        Err(Error::Runtime(
            "record_reshape with input != output not supported in batched run; \
             use run_unbatched, or rewrite the graph to alias the reshape"
                .into(),
        ))
    }

    fn record_relu(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_relu: missing input/output".into()));
        }
        if op.inputs[0] != op.outputs[0] {
            return Err(Error::Runtime(
                "record_relu: out-of-place not supported in batched path".into(),
            ));
        }
        let name = op.inputs[0].clone();
        let buf = self
            .buffers
            .get_mut(&name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;
        rec.record_relu_f32(buf)
            .map_err(|e| Error::Runtime(format!("vk record relu_f32: {e}")))
    }

    fn record_add(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("record_add: needs 2 inputs and 1 output".into()));
        }
        let (a, b, y) = Self::get_three_buffers(
            &mut self.buffers,
            &op.inputs[0],
            &op.inputs[1],
            &op.outputs[0],
        )?;
        rec.record_add_f32(a, b, y)
            .map_err(|e| Error::Runtime(format!("vk record add_f32: {e}")))
    }

    fn record_mul(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("record_mul: needs 2 inputs and 1 output".into()));
        }
        let (a, b, y) = Self::get_three_buffers(
            &mut self.buffers,
            &op.inputs[0],
            &op.inputs[1],
            &op.outputs[0],
        )?;
        rec.record_mul_f32(a, b, y)
            .map_err(|e| Error::Runtime(format!("vk record mul_f32: {e}")))
    }

    fn record_sigmoid(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_sigmoid: missing input/output".into()));
        }
        if op.inputs[0] == op.outputs[0] {
            return Err(Error::Runtime(
                "record_sigmoid: in-place not supported".into(),
            ));
        }
        let (input, output) =
            Self::get_two_buffers(&mut self.buffers, &op.inputs[0], &op.outputs[0])?;
        rec.record_sigmoid_f32(input, output)
            .map_err(|e| Error::Runtime(format!("vk record sigmoid_f32: {e}")))
    }

    fn record_silu(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_silu: missing input/output".into()));
        }
        if op.inputs[0] == op.outputs[0] {
            return Err(Error::Runtime("record_silu: in-place not supported".into()));
        }
        let (input, output) =
            Self::get_two_buffers(&mut self.buffers, &op.inputs[0], &op.outputs[0])?;
        rec.record_silu_f32(input, output)
            .map_err(|e| Error::Runtime(format!("vk record silu_f32: {e}")))
    }

    fn record_gemm(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("record_gemm: needs ≥2 inputs and 1 output".into()));
        }
        let a_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?;
        let b_shape = self
            .graph
            .shapes
            .get(&op.inputs[1])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[1])))?;
        if a_shape.dims.len() != 2 || b_shape.dims.len() != 2 {
            return Err(Error::Runtime("record_gemm: inputs must be 2D".into()));
        }
        let m = if trans_a { a_shape.dims[1] } else { a_shape.dims[0] };
        let k = if trans_a { a_shape.dims[0] } else { a_shape.dims[1] };
        let n = if trans_b { b_shape.dims[0] } else { b_shape.dims[1] };
        if trans_a || trans_b {
            return Err(Error::Runtime(format!(
                "record_gemm: trans_a/trans_b not supported (call fold_gemm_transpose first); \
                 trans_a={trans_a}, trans_b={trans_b}"
            )));
        }
        let a_name = op.inputs[0].clone();
        let b_name = op.inputs[1].clone();
        let c_name = op.outputs[0].clone();
        let bias_name: Option<String> = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            Some(op.inputs[2].clone())
        } else {
            None
        };

        let (a, b, c) =
            Self::get_three_buffers(&mut self.buffers, &a_name, &b_name, &c_name)?;
        rec.record_gemm_f32(a, b, c, m, n, k)
            .map_err(|e| Error::Runtime(format!("vk record gemm_f32: {e}")))?;

        if let Some(bias_name) = bias_name {
            let bias_n = self
                .buffers
                .get(&bias_name)
                .map(|b| b.len_bytes() / 4)
                .unwrap_or(0);
            if bias_n == n {
                // Bias must read the just-written gemm output, so insert a barrier.
                rec.record_memory_barrier();
                let (bias, c) = Self::get_two_buffers(&mut self.buffers, &bias_name, &c_name)?;
                rec.record_bias_add_f32_nhwc(c, bias, n)
                    .map_err(|e| Error::Runtime(format!("vk record gemm bias_add: {e}")))?;
            }
        }
        Ok(())
    }

    fn record_conv2d(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        group: usize,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("record_conv2d: needs ≥2 inputs and 1 output".into()));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?;
        let out_shape = self
            .graph
            .shapes
            .get(&op.outputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.outputs[0])))?;
        if in_shape.dims.len() != 4 {
            return Err(Error::Runtime("record_conv2d input must be 4D NHWC".into()));
        }
        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c_in = in_shape.dims[3];
        let c_out = out_shape.dims[3];
        if group != 1 {
            return Err(Error::Runtime(format!(
                "record_conv2d: group={group} not supported"
            )));
        }
        let bias_name: Option<String> = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            Some(op.inputs[2].clone())
        } else {
            None
        };
        let in_name = op.inputs[0].clone();
        let w_name = op.inputs[1].clone();
        let out_name = op.outputs[0].clone();
        let (input, kernel, output) =
            Self::get_three_buffers(&mut self.buffers, &in_name, &w_name, &out_name)?;
        rec.record_conv2d_f32_nhwc(
            input, kernel, output, n, h_in, w_in, c_in, c_out, kernel_shape[0], kernel_shape[1],
            strides[0], strides[1], pads[0], pads[1],
        )
        .map_err(|e| Error::Runtime(format!("vk record conv2d_f32_nhwc: {e}")))?;

        if let Some(bias_name) = bias_name {
            // See dispatch_conv2d for the bias-shape sanity check rationale.
            let bias_n = self
                .buffers
                .get(&bias_name)
                .map(|b| b.len_bytes() / 4)
                .unwrap_or(0);
            if bias_n == c_out {
                rec.record_memory_barrier();
                let (bias, output) =
                    Self::get_two_buffers(&mut self.buffers, &bias_name, &out_name)?;
                rec.record_bias_add_f32_nhwc(output, bias, c_out)
                    .map_err(|e| Error::Runtime(format!("vk record conv2d bias_add: {e}")))?;
            }
        }
        Ok(())
    }

    fn record_maxpool(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_maxpool: missing input/output".into()));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?;
        if in_shape.dims.len() != 4 {
            return Err(Error::Runtime("record_maxpool input must be 4D NHWC".into()));
        }
        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c = in_shape.dims[3];
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        rec.record_maxpool2d_f32(
            input,
            output,
            n,
            h_in,
            w_in,
            c,
            kernel_shape[0],
            kernel_shape[1],
            strides[0],
            strides[1],
        )
        .map_err(|e| Error::Runtime(format!("vk record maxpool2d_f32: {e}")))
    }

    fn record_softmax(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        _axis: i64,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_softmax: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("record_softmax: in-place not supported".into()));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        let last = *in_shape.dims.last().unwrap_or(&1);
        let rows = in_shape.numel() / last;
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        rec.record_softmax_f32(input, output, rows, last)
            .map_err(|e| Error::Runtime(format!("vk record softmax_f32: {e}")))
    }

    fn record_requantize(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        scale: f32,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_requantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime(
                "record_requantize: in-place not supported".into(),
            ));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        rec.record_requantize_i32_to_i8_packed(input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk record requantize: {e}")))
    }

    fn record_add_quantized(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        scale_a_over_out: f32,
        scale_b_over_out: f32,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("record_add_q: needs 2 inputs + 1 output".into()));
        }
        let (a, b, y) = Self::get_three_buffers(
            &mut self.buffers,
            &op.inputs[0],
            &op.inputs[1],
            &op.outputs[0],
        )?;
        rec.record_add_i8_packed(a, b, y, scale_a_over_out, scale_b_over_out)
            .map_err(|e| Error::Runtime(format!("vk record add_i8: {e}")))
    }

    fn record_quantize(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        scale: f32,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_quantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("record_quantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        rec.record_quantize_f32_to_i8_packed(input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk record quantize: {e}")))
    }

    fn record_dequantize(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        scale: f32,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("record_dequantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("record_dequantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        rec.record_dequantize_i8_packed_to_f32(input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk record dequantize: {e}")))
    }

    #[allow(clippy::too_many_arguments)]
    fn record_conv2d_requant_relu_i8(
        &mut self,
        rec: &mut dragonwing_vulkan::OpsRecorder,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        group: usize,
        requant_scale: f32,
        has_relu: bool,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime(
                "record_conv2d_requant_relu_i8: needs ≥2 inputs and 1 output".into(),
            ));
        }
        if group != 1 {
            return Err(Error::Runtime(format!(
                "record_conv2d_requant_relu_i8: group={group} not supported"
            )));
        }
        if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            return Err(Error::Runtime(
                "record_conv2d_requant_relu_i8: bias must be folded at compile time".into(),
            ));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?;
        let out_shape = self
            .graph
            .shapes
            .get(&op.outputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.outputs[0])))?;
        if in_shape.dims.len() != 4 || out_shape.dims.len() != 4 {
            return Err(Error::Runtime(
                "record_conv2d_requant_relu_i8: shapes must be 4D NHWC".into(),
            ));
        }
        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c_in = in_shape.dims[3];
        let c_out = out_shape.dims[3];
        let in_name = op.inputs[0].clone();
        let w_name = op.inputs[1].clone();
        let out_name = op.outputs[0].clone();
        let (input, kernel, output) =
            Self::get_three_buffers(&mut self.buffers, &in_name, &w_name, &out_name)?;
        rec.record_conv2d_requant_relu_i8_packed(
            input,
            kernel,
            output,
            n,
            h_in,
            w_in,
            c_in,
            c_out,
            kernel_shape[0],
            kernel_shape[1],
            strides[0],
            strides[1],
            pads[0],
            pads[1],
            requant_scale,
            has_relu,
        )
        .map_err(|e| Error::Runtime(format!("vk record conv2d_requant_relu_i8: {e}")))
    }

    // =========================================================================
    // Phase 1 dispatch stubs — filled in by later phases.
    // =========================================================================

    fn dispatch_reshape(&mut self, op: &CompiledOp) -> Result<()> {
        // Reshape is metadata-only; the underlying storage layout is the same.
        // If input and output are different tensors, copy via GPU.
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Ok(());
        }
        if op.inputs[0] == op.outputs[0] {
            return Ok(());
        }
        // For Phase 1 we use download + upload as a placeholder. Phase 2
        // can swap this for `vkCmdCopyBuffer`.
        let src_bytes = {
            let src = self
                .buffers
                .get(&op.inputs[0])
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {}", op.inputs[0])))?;
            let mut bytes = vec![0u8; src.len_bytes()];
            self.backend
                .download(src, &mut bytes)
                .map_err(|e| Error::Runtime(format!("reshape download: {e}")))?;
            bytes
        };
        let dst = self
            .buffers
            .get_mut(&op.outputs[0])
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {}", op.outputs[0])))?;
        let n = dst.len_bytes().min(src_bytes.len());
        self.backend
            .upload(dst, &src_bytes[..n])
            .map_err(|e| Error::Runtime(format!("reshape upload: {e}")))?;
        Ok(())
    }

    // =========================================================================
    // Phase 2 dispatchers — F32 ops wired to dragonwing_vulkan::ops::*.
    // =========================================================================
    //
    // Conventions:
    //   - Input names come from op.inputs[0..]; output name is op.outputs[0].
    //   - For in-place ops (input_name == output_name), we just borrow the
    //     buffer mutably; the GPU shader treats it as inout.
    //   - For out-of-place ops we use get_two_buffers / get_three_buffers
    //     to avoid HashMap borrow conflicts.
    //   - All shape info comes from self.graph.shapes (set by compile_model).
    //   - Optional biases (Conv/Gemm input[2]) are handled by adding a
    //     separate Add pass when present — Phase 2 keeps this simple.

    fn dispatch_relu(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("relu: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();

        if in_name == out_name {
            // In-place ReLU on a single buffer.
            let buf = self
                .buffers
                .get_mut(&in_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {in_name}")))?;
            dragonwing_vulkan::ops::relu_f32(&self.backend, buf)
                .map_err(|e| Error::Runtime(format!("vk relu_f32: {e}")))
        } else {
            // Copy input → output, then in-place relu on output.
            // (relu_f32 is single-buffer; no out-of-place variant exists.)
            let src_bytes = {
                let src = self
                    .buffers
                    .get(&in_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {in_name}")))?;
                let mut b = vec![0u8; src.len_bytes()];
                self.backend
                    .download(src, &mut b)
                    .map_err(|e| Error::Runtime(format!("relu copy download: {e}")))?;
                b
            };
            let dst = self
                .buffers
                .get_mut(&out_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
            self.backend
                .upload(dst, &src_bytes)
                .map_err(|e| Error::Runtime(format!("relu copy upload: {e}")))?;
            dragonwing_vulkan::ops::relu_f32(&self.backend, dst)
                .map_err(|e| Error::Runtime(format!("vk relu_f32: {e}")))
        }
    }

    fn dispatch_add(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("add: needs 2 inputs and 1 output".into()));
        }
        let (a, b, y) = Self::get_three_buffers(&mut self.buffers, &op.inputs[0], &op.inputs[1], &op.outputs[0])?;
        dragonwing_vulkan::ops::add_f32(&self.backend, a, b, y)
            .map_err(|e| Error::Runtime(format!("vk add_f32: {e}")))
    }

    fn dispatch_mul(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("mul: needs 2 inputs and 1 output".into()));
        }
        let (a, b, y) = Self::get_three_buffers(&mut self.buffers, &op.inputs[0], &op.inputs[1], &op.outputs[0])?;
        dragonwing_vulkan::ops::mul_f32(&self.backend, a, b, y)
            .map_err(|e| Error::Runtime(format!("vk mul_f32: {e}")))
    }

    fn dispatch_sigmoid(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("sigmoid: missing input/output".into()));
        }
        if op.inputs[0] == op.outputs[0] {
            return Err(Error::Runtime(
                "vk sigmoid_f32 does not support in-place; compile_model should split"
                    .into(),
            ));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &op.inputs[0], &op.outputs[0])?;
        dragonwing_vulkan::ops::sigmoid_f32(&self.backend, input, output)
            .map_err(|e| Error::Runtime(format!("vk sigmoid_f32: {e}")))
    }

    /// Fused SiLU (`x * sigmoid(x)`) dispatcher — one shader replaces two.
    ///
    /// The op_type is the string "SiLU" produced by the fusion pass in
    /// `crates/dragonwing-onnx/src/fusion.rs`. The op has a single input
    /// (x) and a single output (z = x * σ(x)).
    fn dispatch_silu(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("silu: missing input/output".into()));
        }
        if op.inputs[0] == op.outputs[0] {
            return Err(Error::Runtime(
                "vk silu_f32 does not support in-place".into(),
            ));
        }
        let (input, output) =
            Self::get_two_buffers(&mut self.buffers, &op.inputs[0], &op.outputs[0])?;
        dragonwing_vulkan::ops::silu_f32(&self.backend, input, output)
            .map_err(|e| Error::Runtime(format!("vk silu_f32: {e}")))
    }

    fn dispatch_clip(&mut self, op: &CompiledOp, min: f32, max: f32) -> Result<()> {
        // No clip shader exists. For the common ReLU6 case (min=0, max=6) we
        // could approximate via relu_f32 + saturate; but for general clip we
        // need a dedicated shader. Phase 2 routes the common ReLU (min=0,
        // max=+inf) to relu_f32 and errors otherwise.
        if min == 0.0 && max.is_infinite() {
            return self.dispatch_relu(op);
        }
        let _ = (min, max);
        Err(Error::Runtime(
            "vk clip: only min=0/max=inf (ReLU) supported in Phase 2; \
             general clip shader is future work"
                .into(),
        ))
    }

    fn dispatch_gemm(&mut self, op: &CompiledOp, trans_a: bool, trans_b: bool) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("gemm: needs ≥2 inputs and 1 output".into()));
        }

        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let c_name = &op.outputs[0];

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

        if a_shape.dims.len() != 2 || b_shape.dims.len() != 2 {
            return Err(Error::Runtime("Gemm inputs must be 2D".into()));
        }

        let m = if trans_a { a_shape.dims[1] } else { a_shape.dims[0] };
        let k = if trans_a { a_shape.dims[0] } else { a_shape.dims[1] };
        let n = if trans_b { b_shape.dims[0] } else { b_shape.dims[1] };

        // Vulkan gemm_f32 shader is non-transposed only. For ONNX models that
        // store FC weights with trans_b=true (the typical case), we'd need a
        // transposed shader. Phase 2 errors out and the caller can either:
        //   (a) pre-transpose B when loading the model, or
        //   (b) fall back to CpuGraphRuntime for this op.
        if trans_a || trans_b {
            return Err(Error::Runtime(format!(
                "vk gemm: trans_a/trans_b not supported (yet). \
                 trans_a={trans_a}, trans_b={trans_b}, MxNxK={}x{}x{}",
                m, n, k
            )));
        }

        // Buffers
        let a_name_str = a_name.clone();
        let b_name_str = b_name.clone();
        let c_name_str = c_name.clone();
        let (a, b, c) = Self::get_three_buffers(&mut self.buffers, &a_name_str, &b_name_str, &c_name_str)?;
        dragonwing_vulkan::ops::gemm_f32(&self.backend, a, b, c, m, n, k)
            .map_err(|e| Error::Runtime(format!("vk gemm_f32: {e}")))?;

        // Bias handling (op.inputs[2]) — Task 008 Phase 2: insert a
        // broadcasting bias-add pass after the gemm. bias has shape [N];
        // C has shape [M, N]; we broadcast bias over rows by using c_out=N.
        if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            let bias_name = op.inputs[2].clone();
            let bias_n = self
                .buffers
                .get(&bias_name)
                .map(|b| b.len_bytes() / 4)
                .unwrap_or(0);
            if bias_n == n {
                let (bias, c) =
                    Self::get_two_buffers(&mut self.buffers, &bias_name, &c_name_str)?;
                dragonwing_vulkan::ops::bias_add_f32_nhwc(&self.backend, c, bias, n)
                    .map_err(|e| Error::Runtime(format!("vk gemm bias_add: {e}")))?;
            }
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
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("conv2d: needs ≥2 inputs and 1 output".into()));
        }

        let in_name = &op.inputs[0];
        let w_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let in_shape = self
            .graph
            .shapes
            .get(in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        let out_shape = self
            .graph
            .shapes
            .get(out_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {out_name}")))?;

        if in_shape.dims.len() != 4 {
            return Err(Error::Runtime("conv2d input must be 4D NHWC".into()));
        }

        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c_in = in_shape.dims[3];
        let c_out = out_shape.dims[3];
        let k_h = kernel_shape[0];
        let k_w = kernel_shape[1];
        let pad_h = pads[0];
        let pad_w = pads[1];

        // Depthwise / grouped not supported by the standard vk conv shader.
        if group != 1 {
            return Err(Error::Runtime(format!(
                "vk conv2d: group={group} not supported (need depthwise shader)"
            )));
        }

        let bias_name: Option<String> = if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            Some(op.inputs[2].clone())
        } else {
            None
        };

        let in_name = in_name.clone();
        let w_name = w_name.clone();
        let out_name = out_name.clone();
        let (input, kernel, output) = Self::get_three_buffers(&mut self.buffers, &in_name, &w_name, &out_name)?;

        dragonwing_vulkan::ops::conv2d_f32_nhwc(
            &self.backend,
            input,
            kernel,
            output,
            n,
            h_in,
            w_in,
            c_in,
            c_out,
            k_h,
            k_w,
            strides[0],
            strides[1],
            pad_h,
            pad_w,
        )
        .map_err(|e| Error::Runtime(format!("vk conv2d_f32_nhwc: {e}")))?;

        // Bias handling (op.inputs[2]) — Task 008 Phase 2: broadcasting
        // bias-add pass after the conv. Bias shape is [C_out]; output shape
        // is [N, H_out, W_out, C_out]. The bias_add_f32_nhwc shader uses
        // i % c_out as the bias index, which is correct for NHWC layout.
        if let Some(bias_name) = bias_name {
            // Verify the bias buffer's element count matches c_out. ONNX
            // sometimes carries scalar/empty bias initializers when BN
            // folding has already absorbed the bias; in that case we
            // skip the add to avoid a shape mismatch (and a GPU read of
            // uninitialised memory, which Turnip translates to a
            // device-lost on Adreno A702).
            let bias_n = self
                .buffers
                .get(&bias_name)
                .map(|b| b.len_bytes() / 4)
                .unwrap_or(0);
            if bias_n != c_out {
                eprintln!(
                    "  conv2d: skipping bias_add for `{out_name}` — bias {bias_name} \
                     has {bias_n} elems, c_out={c_out}"
                );
            } else if std::env::var("DRAGONWING_SKIP_BIAS").is_ok() {
                // Debug toggle: skip bias_add to isolate conv from bias.
            } else {
                let (bias, output) =
                    Self::get_two_buffers(&mut self.buffers, &bias_name, &out_name)?;
                dragonwing_vulkan::ops::bias_add_f32_nhwc(&self.backend, output, bias, c_out)
                    .map_err(|e| Error::Runtime(format!("vk conv2d bias_add: {e}")))?;
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
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("maxpool: missing input/output".into()));
        }

        let in_name = &op.inputs[0];
        let out_name = &op.outputs[0];

        let in_shape = self
            .graph
            .shapes
            .get(in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        if in_shape.dims.len() != 4 {
            return Err(Error::Runtime("maxpool input must be 4D NHWC".into()));
        }

        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c = in_shape.dims[3];

        let in_name = in_name.clone();
        let out_name = out_name.clone();
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::maxpool2d_f32(
            &self.backend,
            input,
            output,
            n,
            h_in,
            w_in,
            c,
            kernel_shape[0],
            kernel_shape[1],
            strides[0],
            strides[1],
        )
        .map_err(|e| Error::Runtime(format!("vk maxpool2d_f32: {e}")))
    }

    fn dispatch_softmax(&mut self, op: &CompiledOp, _axis: i64) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("softmax: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime(
                "vk softmax does not support in-place".into(),
            ));
        }

        let in_shape = self
            .graph
            .shapes
            .get(&in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        let last = *in_shape.dims.last().unwrap_or(&1);
        let rows = in_shape.numel() / last;

        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::softmax_f32(&self.backend, input, output, rows, last)
            .map_err(|e| Error::Runtime(format!("vk softmax_f32: {e}")))
    }

    // =========================================================================
    // Phase 3 dispatchers — INT8 packed ops.
    // =========================================================================
    //
    // INT8 ops use UINT32 packing (4× I8 per U32). Tensor sizes are derived
    // from `graph.shapes` via `numel()`. The buffer for a packed I8 tensor
    // has byte length == `numel()` (1 byte per element). The buffer for an
    // INT32 accumulator has byte length == `numel() * 4`.

    fn dispatch_requantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("requantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("requantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::requantize_i32_to_i8_packed(&self.backend, input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk requantize: {e}")))
    }

    fn dispatch_add_quantized(
        &mut self,
        op: &CompiledOp,
        scale_a_over_out: f32,
        scale_b_over_out: f32,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("add_q: needs 2 inputs + 1 output".into()));
        }
        let (a, b, y) =
            Self::get_three_buffers(&mut self.buffers, &op.inputs[0], &op.inputs[1], &op.outputs[0])?;
        dragonwing_vulkan::ops::add_i8_packed(
            &self.backend,
            a,
            b,
            y,
            scale_a_over_out,
            scale_b_over_out,
        )
        .map_err(|e| Error::Runtime(format!("vk add_i8: {e}")))
    }

    fn dispatch_quantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("quantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("quantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::quantize_f32_to_i8_packed(&self.backend, input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk quantize: {e}")))
    }

    fn dispatch_dequantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("dequantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("dequantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::dequantize_i8_packed_to_f32(&self.backend, input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk dequantize: {e}")))
    }

    // =========================================================================
    // Task 008 — Fused INT8 Conv + Requantize + (optional) Relu dispatcher
    // =========================================================================
    //
    // Emitted by the fusion pass when a `Conv → Requantize (→ Relu)` chain is
    // detected in an INT8 graph. The fused shader
    // (`conv2d_requant_relu_i8_packed.comp`) reads packed I8 inputs and
    // weights, accumulates in I32 inside the shader, requantises with the
    // provided scale, and writes packed I8 directly to the output buffer —
    // skipping the intermediate I32 tensor that the unfused path would
    // allocate.
    //
    // Constraints (enforced by the underlying op wrapper):
    //   - `c_in % 4 == 0` and `c_out % 4 == 0` (UINT32 packing).
    //   - `group == 1` (no depthwise).
    //   - bias not supported here (must be folded at compile time).

    #[allow(clippy::too_many_arguments)]
    fn dispatch_conv2d_requant_relu_i8(
        &mut self,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        group: usize,
        requant_scale: f32,
        has_relu: bool,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime(
                "conv2d_requant_relu_i8: needs ≥2 inputs and 1 output".into(),
            ));
        }
        if group != 1 {
            return Err(Error::Runtime(format!(
                "conv2d_requant_relu_i8: group={group} not supported"
            )));
        }
        if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            return Err(Error::Runtime(
                "conv2d_requant_relu_i8: bias must be folded at compile time".into(),
            ));
        }

        let in_name = &op.inputs[0];
        let w_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let in_shape = self
            .graph
            .shapes
            .get(in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        let out_shape = self
            .graph
            .shapes
            .get(out_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {out_name}")))?;

        if in_shape.dims.len() != 4 || out_shape.dims.len() != 4 {
            return Err(Error::Runtime(
                "conv2d_requant_relu_i8: input and output must be 4D NHWC".into(),
            ));
        }

        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c_in = in_shape.dims[3];
        let c_out = out_shape.dims[3];

        let k_h = kernel_shape[0];
        let k_w = kernel_shape[1];
        let stride_h = strides[0];
        let stride_w = strides[1];
        let pad_h = pads[0];
        let pad_w = pads[1];

        let in_name = in_name.clone();
        let w_name = w_name.clone();
        let out_name = out_name.clone();
        let (input, kernel, output) =
            Self::get_three_buffers(&mut self.buffers, &in_name, &w_name, &out_name)?;

        dragonwing_vulkan::ops::conv2d_requant_relu_i8_packed(
            &self.backend,
            input,
            kernel,
            output,
            n,
            h_in,
            w_in,
            c_in,
            c_out,
            k_h,
            k_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            requant_scale,
            has_relu,
        )
        .map_err(|e| Error::Runtime(format!("vk conv2d_requant_relu_i8: {e}")))
    }

    // =========================================================================
    // Task 008 Phase 6 — CPU fallback dispatchers.
    //
    // For ops that don't yet have Vulkan shaders (Sub, Div, Concat,
    // Resize, Split, Transpose, Slice) we download the inputs, run a
    // CPU implementation from `dragonwing-cpu`, and upload the result.
    // The caller (run()) has already flushed/synchronised any pending
    // GPU work before invoking these.
    // =========================================================================

    fn download_f32(&self, name: &str) -> Result<Vec<f32>> {
        let buf = self
            .buffers
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;
        let n_bytes = buf.len_bytes();
        if !n_bytes.is_multiple_of(4) {
            return Err(Error::Runtime(format!(
                "{name}: byte length {n_bytes} not multiple of 4"
            )));
        }
        let mut bytes = vec![0u8; n_bytes];
        self.backend
            .download(buf, &mut bytes)
            .map_err(|e| Error::Runtime(format!("download {name}: {e}")))?;
        let mut out = vec![0f32; n_bytes / 4];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(out)
    }

    fn upload_f32(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let buf = self
            .buffers
            .get_mut(name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;
        let expected = data.len() * 4;
        if buf.len_bytes() != expected {
            return Err(Error::Runtime(format!(
                "{name} upload size mismatch: buf={} data={}",
                buf.len_bytes(),
                expected,
            )));
        }
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), expected) };
        self.backend
            .upload(buf, bytes)
            .map_err(|e| Error::Runtime(format!("upload {name}: {e}")))
    }

    fn dispatch_cpu_sub(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("cpu_sub: needs 2 inputs + 1 output".into()));
        }
        let a = self.download_f32(&op.inputs[0])?;
        let b = self.download_f32(&op.inputs[1])?;
        let out_size = self
            .graph
            .shapes
            .get(&op.outputs[0])
            .map(|s| s.numel())
            .unwrap_or(a.len());
        let mut out = vec![0f32; out_size];
        if a.len() == b.len() {
            dragonwing_cpu::ops::sub_f32(&mut out, &a, &b);
        } else if b.len() == 1 {
            let bv = b[0];
            for (o, &av) in out.iter_mut().zip(a.iter()) {
                *o = av - bv;
            }
        } else {
            let c = b.len();
            for (i, (o, &av)) in out.iter_mut().zip(a.iter()).enumerate() {
                *o = av - b[i % c];
            }
        }
        self.upload_f32(&op.outputs[0], &out)
    }

    fn dispatch_cpu_div(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("cpu_div: needs 2 inputs + 1 output".into()));
        }
        let a = self.download_f32(&op.inputs[0])?;
        let b = self.download_f32(&op.inputs[1])?;
        let out_size = self
            .graph
            .shapes
            .get(&op.outputs[0])
            .map(|s| s.numel())
            .unwrap_or(a.len());
        let mut out = vec![0f32; out_size];
        if a.len() == b.len() {
            dragonwing_cpu::ops::div_f32(&mut out, &a, &b);
        } else if b.len() == 1 {
            let bv = b[0];
            for (o, &av) in out.iter_mut().zip(a.iter()) {
                *o = av / bv;
            }
        } else {
            let c = b.len();
            for (i, (o, &av)) in out.iter_mut().zip(a.iter()).enumerate() {
                *o = av / b[i % c];
            }
        }
        self.upload_f32(&op.outputs[0], &out)
    }

    fn dispatch_cpu_concat(&mut self, op: &CompiledOp, axis: usize) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("cpu_concat: missing input/output".into()));
        }
        let mut input_shapes: Vec<[usize; 4]> = Vec::with_capacity(op.inputs.len());
        let mut input_data: Vec<Vec<f32>> = Vec::with_capacity(op.inputs.len());
        for in_name in &op.inputs {
            let shape = self
                .graph
                .shapes
                .get(in_name)
                .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
            let mut dims = [1usize; 4];
            let offset = 4 - shape.dims.len();
            for (i, &d) in shape.dims.iter().enumerate() {
                dims[offset + i] = d;
            }
            input_shapes.push(dims);
            input_data.push(self.download_f32(in_name)?);
        }
        let inputs: Vec<&[f32]> = input_data.iter().map(|v| v.as_slice()).collect();
        let out_size = self
            .graph
            .shapes
            .get(&op.outputs[0])
            .map(|s| s.numel())
            .ok_or_else(|| Error::Runtime("output shape not found".into()))?;
        let mut out = vec![0f32; out_size];
        dragonwing_cpu::ops::concat_f32(&mut out, &inputs, &input_shapes, axis);
        self.upload_f32(&op.outputs[0], &out)
    }

    fn dispatch_cpu_resize(
        &mut self,
        op: &CompiledOp,
        out_h: usize,
        out_w: usize,
        mode: &str,
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("cpu_resize: missing input/output".into()));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?
            .clone();
        if in_shape.dims.len() != 4 {
            return Err(Error::Runtime("cpu_resize: input must be 4D".into()));
        }
        // After convert_nchw_to_nhwc the dims are [N, H, W, C].
        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c = in_shape.dims[3];
        let resize_mode = match mode {
            "nearest" => dragonwing_cpu::ops::ResizeMode::Nearest,
            "linear" => dragonwing_cpu::ops::ResizeMode::Bilinear,
            other => {
                return Err(Error::Runtime(format!(
                    "cpu_resize: unsupported mode {other}"
                )));
            }
        };
        let input = self.download_f32(&op.inputs[0])?;
        let mut out = vec![0f32; n * out_h * out_w * c];
        dragonwing_cpu::ops::resize_f32(
            &mut out, &input, n, h_in, w_in, out_h, out_w, c, resize_mode,
        );
        self.upload_f32(&op.outputs[0], &out)
    }

    fn dispatch_cpu_split(
        &mut self,
        op: &CompiledOp,
        axis: usize,
        split_sizes: &[usize],
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("cpu_split: missing input/output".into()));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?
            .clone();
        let mut dims = [1usize; 4];
        let offset = 4 - in_shape.dims.len();
        for (i, &d) in in_shape.dims.iter().enumerate() {
            dims[offset + i] = d;
        }
        let input = self.download_f32(&op.inputs[0])?;
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(op.outputs.len());
        for &split_size in split_sizes {
            let mut split_dims = dims;
            split_dims[axis] = split_size;
            outputs.push(vec![0f32; split_dims.iter().product()]);
        }
        let mut output_slices: Vec<&mut [f32]> =
            outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
        dragonwing_cpu::ops::split_f32(&mut output_slices, &input, dims, axis, split_sizes);
        for (i, out_name) in op.outputs.iter().enumerate() {
            self.upload_f32(out_name, &outputs[i])?;
        }
        Ok(())
    }

    fn dispatch_cpu_transpose(&mut self, op: &CompiledOp, perm: &[usize]) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("cpu_transpose: missing input/output".into()));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?
            .clone();
        let input = self.download_f32(&op.inputs[0])?;
        let mut out = vec![0f32; input.len()];
        // Pad shape and perm to 4D.
        let mut dims4 = [1usize; 4];
        let offset = 4 - in_shape.dims.len();
        for (i, &d) in in_shape.dims.iter().enumerate() {
            dims4[offset + i] = d;
        }
        let mut perm4 = [0usize, 1, 2, 3];
        // ONNX perm length matches input rank; shift by offset.
        for (i, &p) in perm.iter().enumerate() {
            perm4[offset + i] = p + offset;
        }
        dragonwing_cpu::ops::transpose_f32(&mut out, &input, dims4, perm4);
        self.upload_f32(&op.outputs[0], &out)
    }

    fn dispatch_cpu_slice(
        &mut self,
        op: &CompiledOp,
        starts: &[isize],
        ends: &[isize],
        axes: &[usize],
        steps: &[isize],
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("cpu_slice: missing input/output".into()));
        }
        let in_shape = self
            .graph
            .shapes
            .get(&op.inputs[0])
            .ok_or_else(|| Error::Runtime(format!("shape not found: {}", op.inputs[0])))?
            .clone();
        let out_size = self
            .graph
            .shapes
            .get(&op.outputs[0])
            .map(|s| s.numel())
            .ok_or_else(|| Error::Runtime("cpu_slice: output shape not found".into()))?;
        let input = self.download_f32(&op.inputs[0])?;
        let mut out = vec![0f32; out_size];
        // Pad shape to 4D.
        let mut dims4 = [1usize; 4];
        let offset = 4 - in_shape.dims.len();
        for (i, &d) in in_shape.dims.iter().enumerate() {
            dims4[offset + i] = d;
        }
        dragonwing_cpu::ops::slice_f32(&mut out, &input, dims4, starts, ends, axes, steps);
        self.upload_f32(&op.outputs[0], &out)
    }

    /// INT8 ReLU on packed tensors. Not in `OpParams` directly — exposed via a
    /// helper for ops that combine ReLU with another quantized step. Phase 3
    /// includes it for testing.
    #[allow(dead_code)]
    fn dispatch_relu_i8(&mut self, in_name: &str, out_name: &str) -> Result<()> {
        if in_name == out_name {
            return Err(Error::Runtime("relu_i8: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, in_name, out_name)?;
        dragonwing_vulkan::ops::relu_i8_packed(&self.backend, input, output)
            .map_err(|e| Error::Runtime(format!("vk relu_i8: {e}")))
    }

    /// Vulkan-side INT8 GEMM (low-level helper, not used by graph dispatch).
    /// Useful for parity tests that build packed buffers directly.
    #[allow(dead_code)]
    fn dispatch_gemm_i8(
        &mut self,
        a_name: &str,
        b_name: &str,
        c_name: &str,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (a, b, c) = Self::get_three_buffers(&mut self.buffers, a_name, b_name, c_name)?;
        dragonwing_vulkan::ops::gemm_i8_packed(&self.backend, a, b, c, m, n, k)
            .map_err(|e| Error::Runtime(format!("vk gemm_i8: {e}")))
    }

    /// Set a Vulkan buffer by raw bytes (Phase 3 helper for INT8 testing).
    ///
    /// Bypasses dtype conversion; the caller is responsible for matching the
    /// buffer's byte length exactly.
    pub fn set_input_bytes(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;
        if buffer.len_bytes() != data.len() {
            return Err(Error::Runtime(format!(
                "set_input_bytes {name}: size mismatch buffer={} bytes, data={} bytes",
                buffer.len_bytes(),
                data.len(),
            )));
        }
        self.backend
            .upload(buffer, data)
            .map_err(|e| Error::Runtime(format!("upload {name}: {e}")))
    }

    /// Read a Vulkan buffer's raw bytes (Phase 3 helper for INT8 testing).
    pub fn get_output_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;
        let mut out = vec![0u8; buffer.len_bytes()];
        self.backend
            .download(buffer, &mut out)
            .map_err(|e| Error::Runtime(format!("download {name}: {e}")))?;
        Ok(out)
    }
}

// =========================================================================
// Tests — Phase 1 only validates the skeleton (construction, no ops).
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::TensorShape;
    use dragonwing_vulkan::VulkanConfig;

    fn try_make_backend() -> Option<VulkanBackend> {
        VulkanBackend::new(VulkanConfig::default()).ok()
    }

    fn empty_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert(
            "input".to_string(),
            TensorShape::new(vec![1, 4], Dtype::F32),
        );
        shapes.insert(
            "output".to_string(),
            TensorShape::new(vec![1, 4], Dtype::F32),
        );
        Graph {
            ops: Vec::new(),
            shapes,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn empty_graph_constructs_and_runs() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let graph = empty_graph();
        let mut rt = VulkanGraphRuntime::new(graph, backend).expect("new");

        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        rt.set_input_f32("input", &input).expect("set_input");

        // Empty op list — run is a no-op + synchronize.
        rt.run().expect("run");

        // Output buffer was never written, but get_output_f32 should still
        // succeed (reads whatever zeros / garbage is in the buffer; we just
        // assert the length).
        let out = rt.get_output_f32("output").expect("get_output");
        assert_eq!(out.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Phase 2 tests: F32 op dispatch parity
    // -----------------------------------------------------------------------

    /// Build a graph with a single Relu op (in-place).
    fn relu_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![1, 8], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "relu1".into(),
            op_type: "Relu".into(),
            inputs: vec!["x".into()],
            outputs: vec!["x".into()], // in-place
            params: OpParams::None,
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["x".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn relu_inplace_matches_cpu() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(relu_graph(), backend).expect("new");

        let input = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, -0.001, 100.0];
        rt.set_input_f32("x", &input).expect("set_input");
        rt.run().expect("run");
        let out = rt.get_output_f32("x").expect("get_output");

        let expected: Vec<f32> = input.iter().map(|&v| v.max(0.0)).collect();
        for (i, (a, b)) in expected.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
    }

    /// Build a graph: y = a + b (out-of-place).
    fn add_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("a".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("b".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("y".into(), TensorShape::new(vec![8], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "add1".into(),
            op_type: "Add".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["y".into()],
            params: OpParams::Add,
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["y".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn add_matches_cpu() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(add_graph(), backend).expect("new");

        let a = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, 0.0, 10.0];
        let b = vec![0.5f32, 0.5, 0.5, 0.5, 1.5, 2.5, 3.5, -5.0];
        rt.set_input_f32("a", &a).expect("set a");
        rt.set_input_f32("b", &b).expect("set b");
        rt.run().expect("run");
        let out = rt.get_output_f32("y").expect("get y");

        for i in 0..a.len() {
            let exp = a[i] + b[i];
            assert!((exp - out[i]).abs() < 1e-6, "mismatch at {i}: {exp} vs {}", out[i]);
        }
    }

    /// Build a graph: y = sigmoid(x), then z = x * y (SiLU).
    fn silu_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("s".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("z".into(), TensorShape::new(vec![8], Dtype::F32));
        let ops = vec![
            CompiledOp {
                name: "sig1".into(),
                op_type: "Sigmoid".into(),
                inputs: vec!["x".into()],
                outputs: vec!["s".into()],
                params: OpParams::Sigmoid,
            },
            CompiledOp {
                name: "mul1".into(),
                op_type: "Mul".into(),
                inputs: vec!["x".into(), "s".into()],
                outputs: vec!["z".into()],
                params: OpParams::Mul,
            },
        ];
        Graph {
            ops,
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["z".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    /// Fused SiLU graph: same input/output as `silu_graph`, but the
    /// fusion pass has collapsed sigmoid + mul into a single SiLU op.
    fn fused_silu_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("z".into(), TensorShape::new(vec![8], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "silu1".into(),
            op_type: "SiLU".into(),
            inputs: vec!["x".into()],
            outputs: vec!["z".into()],
            params: OpParams::None,
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["z".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn fused_silu_matches_reference() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(fused_silu_graph(), backend).expect("new");

        let x = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5];
        rt.set_input_f32("x", &x).expect("set");
        rt.run().expect("run");
        let z = rt.get_output_f32("z").expect("get");

        for (i, &xi) in x.iter().enumerate() {
            let sig = 1.0 / (1.0 + (-xi).exp());
            let expected = xi * sig;
            assert!(
                (expected - z[i]).abs() < 1e-4,
                "fused silu mismatch at {i}: {expected} vs {}",
                z[i]
            );
        }
    }

    #[test]
    fn silu_two_op_graph_matches_cpu() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(silu_graph(), backend).expect("new");

        let x = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5];
        rt.set_input_f32("x", &x).expect("set");
        rt.run().expect("run");
        let z = rt.get_output_f32("z").expect("get");

        for i in 0..x.len() {
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let exp = x[i] * sig;
            assert!((exp - z[i]).abs() < 1e-4, "silu mismatch at {i}: {exp} vs {}", z[i]);
        }
    }

    /// Build a tiny gemm graph: c = a × b, with M=2, K=3, N=2.
    fn gemm_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("a".into(), TensorShape::new(vec![2, 3], Dtype::F32));
        shapes.insert("b".into(), TensorShape::new(vec![3, 2], Dtype::F32));
        shapes.insert("c".into(), TensorShape::new(vec![2, 2], Dtype::F32));
        let ops = vec![CompiledOp {
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
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3 tests: INT8 packed op dispatch
    // -----------------------------------------------------------------------

    /// Pack `i8` slice into little-endian UINT32s as bytes (4× I8 → 1× U32).
    fn pack_i8_bytes(values: &[i8]) -> Vec<u8> {
        // `as u8` gives the two's-complement byte representation, which
        // matches the shader's `int(packed & 0xFF) - (if >= 128) 256` decoding.
        values.iter().map(|&v| v as u8).collect()
    }

    /// Build a graph that just quantizes F32 to I8 packed.
    fn quantize_graph() -> Graph {
        let mut shapes = HashMap::new();
        // Note: dtype on the graph as a whole still drives buffer allocation.
        // For mixed-dtype graphs we set per-tensor shapes accordingly.
        shapes.insert("x_f32".into(), TensorShape::new(vec![16], Dtype::F32));
        shapes.insert("x_i8".into(), TensorShape::new(vec![16], Dtype::I8));
        let ops = vec![CompiledOp {
            name: "q1".into(),
            op_type: "Quantize".into(),
            inputs: vec!["x_f32".into()],
            outputs: vec!["x_i8".into()],
            params: OpParams::Quantize { scale: 0.1 },
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["x_f32".into()],
            outputs: vec!["x_i8".into()],
            initializers: HashMap::new(),
            // Even for a mixed-dtype graph we have to pick a "primary" dtype;
            // F32 makes set_input_f32 use the right path for the F32 input.
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn quantize_roundtrip_via_dequantize() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };

        // We build a Q → DQ chain manually since the graph types don't yet
        // model two-shape graphs cleanly.
        let mut shapes = HashMap::new();
        shapes.insert("x_f32".into(), TensorShape::new(vec![16], Dtype::F32));
        shapes.insert("x_i8".into(), TensorShape::new(vec![16], Dtype::I8));
        shapes.insert("y_f32".into(), TensorShape::new(vec![16], Dtype::F32));
        let scale = 0.1f32;
        let ops = vec![
            CompiledOp {
                name: "q1".into(),
                op_type: "Quantize".into(),
                inputs: vec!["x_f32".into()],
                outputs: vec!["x_i8".into()],
                params: OpParams::Quantize { scale },
            },
            CompiledOp {
                name: "dq1".into(),
                op_type: "Dequantize".into(),
                inputs: vec!["x_i8".into()],
                outputs: vec!["y_f32".into()],
                params: OpParams::Dequantize { scale },
            },
        ];
        let graph = Graph {
            ops,
            shapes,
            inputs: vec!["x_f32".into()],
            outputs: vec!["y_f32".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };

        let mut rt = VulkanGraphRuntime::new(graph, backend).expect("new");
        let input = vec![
            0.0f32, 0.1, -0.2, 0.5, -0.5, 1.0, -1.0, 12.7, // 12.7/0.1=127, clamps OK
            -12.7, 6.3, -6.3, 0.05, -0.05, 0.3, -0.7, 2.5,
        ];
        rt.set_input_f32("x_f32", &input).expect("set");
        rt.run().expect("run");
        let out = rt.get_output_f32("y_f32").expect("get");

        // Each value should be within one quantization step (scale = 0.1)
        // of the original. The roundtrip clamps values outside [-12.8, 12.7].
        for (i, (a, b)) in input.iter().zip(out.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(err <= scale, "qdq mismatch at {i}: {a} -> {b}, err={err}");
        }
    }

    /// Build a graph that does only requantize: I32 input → I8 packed output.
    /// Used to validate the shader directly with manually-prepared buffers.
    fn requantize_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("acc_i32".into(), TensorShape::new(vec![16], Dtype::I32));
        shapes.insert("out_i8".into(), TensorShape::new(vec![16], Dtype::I8));
        let ops = vec![CompiledOp {
            name: "rq1".into(),
            op_type: "Requantize".into(),
            inputs: vec!["acc_i32".into()],
            outputs: vec!["out_i8".into()],
            params: OpParams::Requantize { scale: 0.01 },
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["acc_i32".into()],
            outputs: vec!["out_i8".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32, // doesn't matter for raw byte I/O
        }
    }

    #[test]
    fn requantize_clamps_and_scales() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };

        let mut rt = VulkanGraphRuntime::new(requantize_graph(), backend).expect("new");

        // input INT32 values
        let acc: Vec<i32> = vec![
            0, 100, -100, 12700, // 12700 * 0.01 = 127
            -12700, 25400,  // saturates to +127
            -25400, // saturates to -128
            500, -500, 1234, -1234, 5, -5, 9999, -9999, 12700,
        ];
        let acc_bytes: Vec<u8> = acc
            .iter()
            .flat_map(|v| v.to_le_bytes().into_iter())
            .collect();
        rt.set_input_bytes("acc_i32", &acc_bytes).expect("set");
        rt.run().expect("run");
        let out_bytes = rt.get_output_bytes("out_i8").expect("get");
        let out: Vec<i8> = out_bytes.iter().map(|&b| b as i8).collect();

        for (i, &v) in acc.iter().enumerate() {
            let expected = ((v as f32) * 0.01).round().clamp(-128.0, 127.0) as i8;
            assert_eq!(
                out[i], expected,
                "requantize[{i}]: input={v}, expected={expected}, got={}",
                out[i]
            );
        }
    }

    /// INT8 ReLU graph.
    fn relu_i8_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![8], Dtype::I8));
        shapes.insert("y".into(), TensorShape::new(vec![8], Dtype::I8));
        // OpParams::None is fine; dispatcher routes by op_type for Quantize/Relu
        // but we use the helper directly.
        Graph {
            ops: Vec::new(),
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn relu_i8_clamps_negatives() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };

        let mut rt = VulkanGraphRuntime::new(relu_i8_graph(), backend).expect("new");

        let input: Vec<i8> = vec![-128, -1, 0, 1, 127, -50, 50, -10];
        rt.set_input_bytes("x", &pack_i8_bytes(&input))
            .expect("set");

        rt.dispatch_relu_i8("x", "y").expect("relu_i8");
        rt.backend.synchronize().expect("sync");

        let out_bytes = rt.get_output_bytes("y").expect("get");
        let out: Vec<i8> = out_bytes.iter().map(|&b| b as i8).collect();

        let expected: Vec<i8> = input.iter().map(|&v| v.max(0)).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn gemm_matches_reference() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(gemm_graph(), backend).expect("new");

        // a is 2×3, b is 3×2
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        rt.set_input_f32("a", &a).expect("set a");
        rt.set_input_f32("b", &b).expect("set b");
        rt.run().expect("run");
        let c = rt.get_output_f32("c").expect("get c");

        // c[0,0] = 1*1 + 2*0 + 3*1 = 4
        // c[0,1] = 1*0 + 2*1 + 3*1 = 5
        // c[1,0] = 4*1 + 5*0 + 6*1 = 10
        // c[1,1] = 4*0 + 5*1 + 6*1 = 11
        let expected = [4.0f32, 5.0, 10.0, 11.0];
        for (i, (exp, got)) in expected.iter().zip(c.iter()).enumerate() {
            assert!((exp - got).abs() < 1e-4, "gemm[{i}]: {exp} vs {got}");
        }
    }

    // -----------------------------------------------------------------------
    // Task 008 Phase 2 tests: F32 bias broadcasting for Conv/Gemm.
    // -----------------------------------------------------------------------

    /// Build a graph: c = a × b + bias, with M=2, K=3, N=4 and bias of shape [4].
    fn gemm_with_bias_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("a".into(), TensorShape::new(vec![2, 3], Dtype::F32));
        shapes.insert("b".into(), TensorShape::new(vec![3, 4], Dtype::F32));
        shapes.insert("bias".into(), TensorShape::new(vec![4], Dtype::F32));
        shapes.insert("c".into(), TensorShape::new(vec![2, 4], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "gemm_bias".into(),
            op_type: "Gemm".into(),
            inputs: vec!["a".into(), "b".into(), "bias".into()],
            outputs: vec!["c".into()],
            params: OpParams::Gemm {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: false,
            },
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["a".into(), "b".into(), "bias".into()],
            outputs: vec!["c".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn gemm_with_bias_matches_reference() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(gemm_with_bias_graph(), backend).expect("new");

        // a is 2×3
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // b is 3×4, identity-ish so the matmul produces a recognisable result
        // Use a structured pattern: b[k,n] = (k+1) * (n+1)
        let b: Vec<f32> = {
            let mut v = vec![0.0; 12];
            for k in 0..3 {
                for n in 0..4 {
                    v[k * 4 + n] = (k as f32 + 1.0) * (n as f32 + 1.0);
                }
            }
            v
        };
        let bias: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

        rt.set_input_f32("a", &a).expect("set a");
        rt.set_input_f32("b", &b).expect("set b");
        rt.set_input_f32("bias", &bias).expect("set bias");
        rt.run().expect("run");
        let c = rt.get_output_f32("c").expect("get c");

        // Reference CPU compute: c[m,n] = sum_k a[m,k]*b[k,n] + bias[n]
        for m in 0..2 {
            for n in 0..4 {
                let mut acc = 0.0f32;
                for k in 0..3 {
                    acc += a[m * 3 + k] * b[k * 4 + n];
                }
                let exp = acc + bias[n];
                let got = c[m * 4 + n];
                assert!(
                    (exp - got).abs() < 1e-4,
                    "gemm_bias[{m},{n}]: exp={exp} got={got}"
                );
            }
        }
    }

    /// Build a conv graph with bias: 1×3×3×4 input, 1×1 kernel, 4→2 channels, bias[2].
    fn conv_with_bias_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![1, 3, 3, 4], Dtype::F32));
        // Kernel for conv2d_f32_nhwc: layout [k_h, k_w, c_in, c_out]
        shapes.insert("w".into(), TensorShape::new(vec![1, 1, 4, 2], Dtype::F32));
        shapes.insert("bias".into(), TensorShape::new(vec![2], Dtype::F32));
        shapes.insert("y".into(), TensorShape::new(vec![1, 3, 3, 2], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "conv_bias".into(),
            op_type: "Conv".into(),
            inputs: vec!["x".into(), "w".into(), "bias".into()],
            outputs: vec!["y".into()],
            params: OpParams::Conv2d {
                kernel_shape: [1, 1],
                strides: [1, 1],
                pads: [0, 0, 0, 0],
                dilations: [1, 1],
                group: 1,
            },
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["x".into(), "w".into(), "bias".into()],
            outputs: vec!["y".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    // -----------------------------------------------------------------------
    // Task 008 Phase 4 tests: single-command-buffer (batched) execution.
    // -----------------------------------------------------------------------

    /// Build a graph with two dependent ops: sigmoid → mul, so we can verify
    /// that the inserted barrier correctly serialises them.
    fn two_op_chain_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("s".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("z".into(), TensorShape::new(vec![8], Dtype::F32));
        let ops = vec![
            CompiledOp {
                name: "sig1".into(),
                op_type: "Sigmoid".into(),
                inputs: vec!["x".into()],
                outputs: vec!["s".into()],
                params: OpParams::Sigmoid,
            },
            CompiledOp {
                name: "mul1".into(),
                op_type: "Mul".into(),
                inputs: vec!["x".into(), "s".into()],
                outputs: vec!["z".into()],
                params: OpParams::Mul,
            },
        ];
        Graph {
            ops,
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["z".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn slab_packing_yields_one_allocation_for_small_graph() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        // 10 tensors of ~32 bytes each: total <1 MiB → must fit in one slab.
        let mut shapes = HashMap::new();
        for i in 0..10 {
            shapes.insert(
                format!("t{i}"),
                TensorShape::new(vec![1, 8], Dtype::F32),
            );
        }
        let graph = Graph {
            ops: Vec::new(),
            shapes,
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };
        let rt = VulkanGraphRuntime::new(graph, backend).expect("new");
        // Phase 5 acceptance: <10 vkAllocateMemory calls (i.e. <10 slabs).
        // For a tiny graph this should be exactly 1.
        let slab_count = rt.slab_count();
        assert!(
            slab_count <= 1,
            "expected ≤1 slab for 10-tensor graph, got {slab_count}"
        );
        if let Some(stats) = rt.slab_stats() {
            assert_eq!(stats.num_slabs, slab_count);
            assert!(
                stats.bytes_in_use >= 10 * 256,
                "bytes_in_use {} too small",
                stats.bytes_in_use
            );
        }
    }

    #[test]
    fn batched_run_matches_reference() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(two_op_chain_graph(), backend).expect("new");
        let x = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5];
        rt.set_input_f32("x", &x).expect("set");
        // run() now uses the single-command-buffer recorder + barriers.
        rt.run().expect("batched run");
        let z = rt.get_output_f32("z").expect("get");
        for (i, &xi) in x.iter().enumerate() {
            let sig = 1.0 / (1.0 + (-xi).exp());
            let expected = xi * sig;
            assert!(
                (expected - z[i]).abs() < 1e-4,
                "batched silu mismatch at {i}: {expected} vs {}",
                z[i]
            );
        }
    }

    #[test]
    fn needs_barrier_detects_immediate_dependency() {
        // Build two-op chain: op0 outputs "a", op1 reads "a".
        let ops = vec![
            CompiledOp {
                name: "op0".into(),
                op_type: "Relu".into(),
                inputs: vec!["x".into()],
                outputs: vec!["a".into()],
                params: OpParams::None,
            },
            CompiledOp {
                name: "op1".into(),
                op_type: "Relu".into(),
                inputs: vec!["a".into()],
                outputs: vec!["y".into()],
                params: OpParams::None,
            },
        ];
        assert!(VulkanGraphRuntime::needs_barrier(&ops, 1));
    }

    #[test]
    fn needs_barrier_skips_independent_ops() {
        // op0 outputs "a", op1 reads "x" (input only). No barrier needed.
        let ops = vec![
            CompiledOp {
                name: "op0".into(),
                op_type: "Relu".into(),
                inputs: vec!["x".into()],
                outputs: vec!["a".into()],
                params: OpParams::None,
            },
            CompiledOp {
                name: "op1".into(),
                op_type: "Relu".into(),
                inputs: vec!["x".into()],
                outputs: vec!["b".into()],
                params: OpParams::None,
            },
        ];
        assert!(!VulkanGraphRuntime::needs_barrier(&ops, 1));
    }

    #[test]
    fn conv2d_with_bias_matches_reference() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt =
            VulkanGraphRuntime::new(conv_with_bias_graph(), backend).expect("new");

        // Input: 1×3×3×4 = 36 elements, deterministic pattern.
        let mut x = vec![0.0f32; 36];
        for i in 0..36 {
            x[i] = (i as f32) * 0.1 - 1.0;
        }
        // Kernel: 1×1×4×2 = 8 elements. Pattern: w[k_h, k_w, c_in, c_out].
        // For a 1×1 conv, this reduces to a [4×2] matmul per spatial.
        let w: Vec<f32> = vec![
            // c_in=0: c_out 0, 1
            0.5, -0.25,
            // c_in=1
            0.1, 0.2,
            // c_in=2
            -0.3, 0.4,
            // c_in=3
            0.7, 0.0,
        ];
        let bias: Vec<f32> = vec![100.0, -50.0];

        rt.set_input_f32("x", &x).expect("set x");
        rt.set_input_f32("w", &w).expect("set w");
        rt.set_input_f32("bias", &bias).expect("set bias");
        rt.run().expect("run");
        let y = rt.get_output_f32("y").expect("get y");

        // Reference: for each spatial position s in 0..9, for each oc in 0..2,
        //   y[s, oc] = sum_{ic=0..4} x[s, ic] * w[ic, oc] + bias[oc]
        for s in 0..9 {
            for oc in 0..2 {
                let mut acc = 0.0f32;
                for ic in 0..4 {
                    acc += x[s * 4 + ic] * w[ic * 2 + oc];
                }
                let exp = acc + bias[oc];
                let got = y[s * 2 + oc];
                assert!(
                    (exp - got).abs() < 1e-4,
                    "conv_bias[s={s},oc={oc}]: exp={exp} got={got}"
                );
            }
        }
    }
}
