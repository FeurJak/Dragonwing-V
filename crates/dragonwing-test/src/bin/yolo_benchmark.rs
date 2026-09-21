//! YOLOv8 on Vulkan — end-to-end benchmark (Task 008 Phase 6).
//!
//! Usage:
//!     yolo-benchmark <path/to/yolov8.onnx>
//!
//! Loads an ONNX model, applies the dragonwing-onnx graph passes
//! (NCHW→NHWC, fold_gemm_transpose, BN-fold, fusion), constructs a
//! `VulkanGraphRuntime`, and runs N inferences while measuring latency.

use std::time::{Duration, Instant};

use dragonwing_core::Dtype;
use dragonwing_onnx::{
    apply_fusion_passes, compile_model, convert_nchw_to_nhwc, count_fuseable_patterns,
    fold_batchnorm, fold_gemm_transpose, fold_qdq_patterns, graph_has_qdq,
    transpose_nchw_to_nhwc, VulkanGraphRuntime,
};
use dragonwing_vulkan::{VulkanBackend, VulkanConfig};

#[derive(Debug, Clone, Copy)]
struct Stats {
    n: usize,
    min: Duration,
    median: Duration,
    p95: Duration,
    mean: Duration,
    max: Duration,
}

fn summarise(mut samples: Vec<Duration>) -> Stats {
    samples.sort();
    let n = samples.len();
    let sum: Duration = samples.iter().sum();
    let mean = sum / (n.max(1) as u32);
    let p95_idx = ((n as f64 * 0.95) as usize).min(n - 1);
    Stats {
        n,
        min: *samples.first().unwrap(),
        median: samples[n / 2],
        p95: samples[p95_idx],
        mean,
        max: *samples.last().unwrap(),
    }
}

fn print_stats(label: &str, s: Stats) {
    println!(
        "  {label}: n={}  min={:>8.2}ms  med={:>8.2}ms  p95={:>8.2}ms  mean={:>8.2}ms  max={:>8.2}ms",
        s.n,
        s.min.as_secs_f64() * 1e3,
        s.median.as_secs_f64() * 1e3,
        s.p95.as_secs_f64() * 1e3,
        s.mean.as_secs_f64() * 1e3,
        s.max.as_secs_f64() * 1e3,
    );
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: yolo-benchmark <path/to/model.onnx>");
        std::process::exit(2);
    });
    let warmup_n: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let iters_n: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let quantize = std::env::var("QUANTIZE").is_ok();
    let unbatched = std::env::var("UNBATCHED").is_ok();
    let max_ops: Option<usize> = std::env::var("MAX_OPS").ok().and_then(|v| v.parse().ok());

    println!("=== dragonwing-edge yolo-benchmark (Task 008 Phase 6) ===");
    println!("Loading {path}...");
    let mut model = match dragonwing_onnx::load_model(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR loading: {e}");
            std::process::exit(1);
        }
    };
    println!("  inputs : {} outputs : {}", model.inputs.len(), model.outputs.len());
    println!("  nodes  : {}", model.nodes.len());

    println!("\n[Pass 1] fold_batchnorm");
    if let Err(e) = fold_batchnorm(&mut model) {
        eprintln!("  fold_batchnorm warning: {e}");
    } else {
        println!("  ok (nodes after = {})", model.nodes.len());
    }

    println!("\n[Pass 2] compile_model (Dtype::F32)");
    let mut graph = match compile_model(&model, Dtype::F32) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ERROR compile: {e}");
            std::process::exit(1);
        }
    };
    println!("  ops    : {}", graph.ops.len());

    // ----- Task 009 — QDQ fold (before NCHW→NHWC so axis=0 weight
    // quantization sees OIHW layout). Cheap predicate skips when the
    // model is plain F32. -----
    if graph_has_qdq(&graph) {
        println!("\n[Pass 2b] fold_qdq_patterns");
        match fold_qdq_patterns(&mut graph) {
            Ok(stats) => {
                println!(
                    "  conv_folded={} dq_removed={} q_removed={} weights_quantized={} qdq_unfolded={}",
                    stats.conv_folded,
                    stats.dequantize_removed,
                    stats.quantize_removed,
                    stats.weights_quantized,
                    stats.qdq_unfolded,
                );
                println!("  ops after qdq fold: {}", graph.ops.len());
            }
            Err(e) => {
                eprintln!("ERROR fold_qdq_patterns: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("\n[Pass 2b] fold_qdq_patterns (skipped — no QDQ nodes detected)");
    }

    println!("\n[Pass 3] convert_nchw_to_nhwc");
    if let Err(e) = convert_nchw_to_nhwc(&mut graph) {
        eprintln!("  convert NHWC warning: {e}");
    } else {
        println!("  ok");
    }

    println!("\n[Pass 4] fold_gemm_transpose");
    if let Err(e) = fold_gemm_transpose(&mut graph) {
        eprintln!("  fold_gemm_transpose warning: {e}");
    } else {
        println!("  ok");
    }

    if quantize {
        println!("\n[Pass 5] (skipped — QUANTIZE env requested but no calib file path implemented yet)");
    }

    if let Some(limit) = max_ops {
        let total = graph.ops.len();
        if limit < total {
            graph.ops.truncate(limit);
            println!("\n[Truncate] keeping first {limit} of {total} ops for debug");
            if let Some(last_op) = graph.ops.last() {
                if let Some(out_name) = last_op.outputs.first() {
                    graph.outputs = vec![out_name.clone()];
                }
            }
        }
    }

    println!("\n[Pass 6] apply_fusion_passes");
    let stats_pre = count_fuseable_patterns(&graph);
    apply_fusion_passes(&mut graph);
    let stats_post = count_fuseable_patterns(&graph);
    println!(
        "  fuseable patterns before: conv_relu={} silu={} conv_requant_relu_i8={}",
        stats_pre.conv_relu, stats_pre.silu, stats_pre.conv_requant_relu_i8
    );
    println!(
        "  fuseable patterns after : conv_relu={} silu={} conv_requant_relu_i8={}",
        stats_post.conv_relu, stats_post.silu, stats_post.conv_requant_relu_i8
    );
    println!("  ops after fusion: {}", graph.ops.len());

    // ----- Vulkan setup ----------------------------------------------------
    println!("\n=== Vulkan ===");
    let backend = match VulkanBackend::new(VulkanConfig::default()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR Vulkan init: {e}");
            std::process::exit(2);
        }
    };
    println!("  device: {}", backend.device_name());
    println!("  driver: {}", backend.driver_info());

    // Capture the input shape (post-NCHW→NHWC).
    let (input_name, input_shape): (String, Vec<usize>) = {
        let (name, shape) = graph
            .input_info()
            .into_iter()
            .next()
            .expect("graph has no inputs");
        (name.to_string(), shape.dims.clone())
    };
    let (output_name, _output_shape) = {
        let (name, shape) = graph
            .output_info()
            .into_iter()
            .next()
            .expect("graph has no outputs");
        (name.to_string(), shape.dims.clone())
    };

    println!("\n[Init] Building VulkanGraphRuntime ({} ops)...", graph.ops.len());
    let init_start = Instant::now();
    let mut rt = match VulkanGraphRuntime::new(graph, backend.clone()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR runtime new: {e}");
            std::process::exit(1);
        }
    };
    let init_ms = init_start.elapsed().as_secs_f64() * 1e3;
    println!("  init time: {init_ms:.1} ms");
    println!("  slab count (vkAllocateMemory): {}", rt.slab_count());
    if let Some(s) = rt.slab_stats() {
        println!(
            "  slab bytes_in_use: {:.2} MiB / total: {:.2} MiB ({:.1}% util)",
            s.bytes_in_use as f64 / (1024.0 * 1024.0),
            s.total_allocated as f64 / (1024.0 * 1024.0),
            s.utilization * 100.0,
        );
    }

    // Synthetic input — random pattern in [0,1).
    let n_elem: usize = input_shape.iter().product();
    let input_data: Vec<f32> = (0..n_elem).map(|i| ((i % 251) as f32) / 251.0).collect();
    // If the input is 4D NCHW we need to NHWC-transpose it before upload.
    // After convert_nchw_to_nhwc the graph's input_shape is already NHWC, so
    // we feed NHWC-formatted data directly. (For an externally-shaped input
    // we'd compute the NCHW form first and call transpose_nchw_to_nhwc here.)
    let _ = transpose_nchw_to_nhwc; // silence unused warning

    let do_run = |rt: &mut VulkanGraphRuntime| -> Result<(), String> {
        if unbatched {
            rt.run_unbatched().map_err(|e| e.to_string())
        } else {
            rt.run().map_err(|e| e.to_string())
        }
    };
    if unbatched {
        println!("(running in UNBATCHED mode)");
    }

    println!("\n[Warmup] {warmup_n} iterations");
    for i in 0..warmup_n {
        rt.set_input_f32(&input_name, &input_data)
            .unwrap_or_else(|e| {
                eprintln!("warmup set_input failed: {e}");
                std::process::exit(1);
            });
        let t = Instant::now();
        do_run(&mut rt).unwrap_or_else(|e| {
            eprintln!("warmup run {i} failed: {e}");
            std::process::exit(1);
        });
        let dt = t.elapsed();
        println!("  warmup[{i}]: {:.2} ms", dt.as_secs_f64() * 1e3);
    }

    println!("\n[Bench] {iters_n} iterations");
    let mut samples = Vec::with_capacity(iters_n);
    for _ in 0..iters_n {
        rt.set_input_f32(&input_name, &input_data).expect("set");
        let t = Instant::now();
        do_run(&mut rt).expect("run");
        samples.push(t.elapsed());
    }
    print_stats("yolo_e2e", summarise(samples));

    // Read back the output to confirm the pipeline produced something.
    match rt.get_output_f32(&output_name) {
        Ok(out) => {
            let nonzero = out.iter().filter(|&&v| v.abs() > 1e-6).count();
            println!(
                "\n[Output] {output_name}: {} elements, {} nonzero ({:.1}%)",
                out.len(),
                nonzero,
                100.0 * nonzero as f64 / out.len() as f64,
            );
        }
        Err(e) => {
            eprintln!("get_output failed: {e}");
        }
    }

    println!("\n=== done ===");
}
