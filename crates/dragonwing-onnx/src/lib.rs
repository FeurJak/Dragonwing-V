//! `dragonwing-onnx` — Minimal ONNX model loader for dragonwing-edge.
//!
//! This crate provides a zero-external-dependency ONNX parser and graph builder
//! for static-shape inference graphs. It targets opset 12 (MobileNetV2's version)
//! and supports the subset of ops needed for typical vision classifiers.
//!
//! # Design
//!
//! The loader uses a hand-rolled protobuf reader (~300 lines) instead of `prost`
//! or `onnx-rs`. This keeps the dependency count at zero and simplifies debugging.
//! ONNX uses only varint + length-delimited wire types, so the reader is minimal.
//!
//! # Two-Phase Compilation
//!
//! Following the QNN EP pattern from task 002's analysis:
//!
//! 1. **Validate** — walks the graph, checks each op is supported, reports errors.
//! 2. **Compile** — allocates buffers, builds the execution plan.
//!
//! This separation ensures no resources are leaked if validation fails.
//!
//! # Usage
//!
//! ```ignore
//! use dragonwing_onnx::{load_model, compile_graph};
//!
//! let model = load_model("model.onnx")?;
//! let graph = compile_graph(&model, &backend)?;
//! let outputs = graph.run(inputs)?;
//! ```
//!
//! # Supported Ops
//!
//! - `Conv` (including depthwise via groups == C_in)
//! - `Relu`, `Relu6` (Clip with min=0, max=6)
//! - `Add` (same-shape only)
//! - `GlobalAveragePool`
//! - `Gemm` (alpha=1, beta=1 only)
//! - `Softmax` (last axis only)
//! - `Reshape`, `Flatten` (metadata only, no runtime op)
//! - `BatchNormalization` (folded into preceding Conv at load time)
//!
//! # INT8 Quantization (Task 006)
//!
//! The [`calibration`] module provides tools for post-training quantization:
//!
//! ```ignore
//! use dragonwing_onnx::{load_model, compile_model, Calibrator};
//! use dragonwing_core::Dtype;
//!
//! let model = load_model("model.onnx")?;
//! let graph = compile_model(&model, Dtype::F32)?;
//! let mut calibrator = Calibrator::new(graph)?;
//!
//! // Feed calibration images
//! for image in calibration_images {
//!     calibrator.feed("input", &image)?;
//! }
//!
//! // Compute quantization parameters
//! let params = calibrator.compute_params()?;
//! ```

#![warn(missing_docs)]

mod error;
mod proto;
mod model;
mod builder;
mod graph;
mod runtime;
mod postprocess;
mod fusion;
mod qdq;
#[cfg(feature = "cpu")]
mod calibration;
#[cfg(feature = "cpu")]
mod quantize;
#[cfg(feature = "vulkan")]
mod vulkan_runtime;

pub use error::{Error, Result};
pub use model::{Model, OnnxNode, OnnxTensor, OnnxAttribute, AttributeValue, DataType};
pub use builder::{OpBuilder, BuildContext, UnsupportedReason, ValidationReport, TensorShape, CompiledOp, OpParams};
pub use graph::{Graph, validate_model, compile_model, fold_batchnorm,
                convert_nchw_to_nhwc, transpose_nchw_to_nhwc, transpose_nhwc_to_nchw,
                fold_gemm_transpose};
pub use runtime::GraphRuntime;
pub use postprocess::{Detection, iou, decode_detections_v8, decode_detections_v8_alt, nms, nms_agnostic, postprocess_yolo};
pub use fusion::{apply_fusion_passes, count_fuseable_patterns, FusionStats};
pub use qdq::{fold_qdq_patterns, graph_has_qdq, QdqFoldStats};

/// CPU-optimized graph runtime (requires `cpu` feature).
#[cfg(feature = "cpu")]
pub use runtime::CpuGraphRuntime;

/// Calibration tools for INT8 quantization (requires `cpu` feature).
#[cfg(feature = "cpu")]
pub use calibration::{Calibrator, CalibrationStrategy};

/// Quantized graph compiler for INT8 inference (requires `cpu` feature).
#[cfg(feature = "cpu")]
pub use quantize::{QuantizedGraphCompiler, QuantizedGraphInfo};

/// Vulkan-accelerated graph runtime (requires `vulkan` feature).
#[cfg(feature = "vulkan")]
pub use vulkan_runtime::VulkanGraphRuntime;

use std::path::Path;

/// Load an ONNX model from a file path.
///
/// Parses the protobuf, extracts the graph structure, node attributes, and
/// initializers (constant tensors). Returns a `Model` struct ready for
/// compilation.
///
/// # Errors
///
/// - `Error::Io` if the file cannot be read.
/// - `Error::Parse` if the protobuf is malformed.
/// - `Error::UnsupportedOpset` if the opset version is not 12.
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Model> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| Error::Io(format!("failed to read {}: {}", path.as_ref().display(), e)))?;
    model::parse_model(&bytes)
}

/// Load an ONNX model from a byte slice.
///
/// Same as `load_model` but operates on in-memory bytes.
pub fn load_model_from_bytes(bytes: &[u8]) -> Result<Model> {
    model::parse_model(bytes)
}
