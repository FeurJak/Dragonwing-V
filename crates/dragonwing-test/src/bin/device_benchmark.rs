//! `device-benchmark` — measure VulkanGraphRuntime on the target device.
//!
//! Task 007 deliverable. Runs a battery of synthetic-graph benchmarks on
//! the local Vulkan device (Adreno A702 on Arduino UNO Q in production):
//!
//! 1. **Device info** — print Vulkan device name, driver, memory budget.
//! 2. **F32 SiLU chain** — N×Sigmoid+Mul vs N×fused-SiLU, to validate the
//!    Phase 4 fusion benefit.
//! 3. **INT8 GEMM** — 64×64×128, vs F32 GEMM at same size.
//! 4. **INT8 Conv2D** — small synthetic NHWC conv.
//! 5. **Fused INT8 Conv2D + Requant + ReLU** — same shape, end-to-end.
//!
//! For each benchmark, runs N warmup iterations, then M measured ones,
//! reports min/median/p95/mean in microseconds.
//!
//! Designed to run on the Arduino UNO Q via:
//!
//! ```sh
//! # Build on device (cross-compile from macOS is painful with Vulkan loader)
//! cd ~/Dragonwing-V
//! PATH=~/.cargo/bin:$PATH \
//!   cargo build -p dragonwing-test --bin device-benchmark --release
//! ./target/release/device-benchmark
//! ```

use dragonwing_core::{Backend, BufferKind, Dtype};
use dragonwing_onnx::{CompiledOp, Graph, OpParams, TensorShape, VulkanGraphRuntime};
use dragonwing_vulkan::{VulkanBackend, VulkanConfig};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------

/// Summary stats for a series of timed runs.
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
    assert!(!samples.is_empty(), "summarise: empty sample set");
    samples.sort();
    let n = samples.len();
    let min = samples[0];
    let max = samples[n - 1];
    let median = samples[n / 2];
    let p95_idx = ((n as f64) * 0.95).floor() as usize;
    let p95 = samples[p95_idx.min(n - 1)];
    let total_ns: u128 = samples.iter().map(|d| d.as_nanos()).sum();
    let mean = Duration::from_nanos((total_ns / n as u128) as u64);
    Stats {
        n,
        min,
        median,
        p95,
        mean,
        max,
    }
}

fn print_stats(label: &str, s: Stats) {
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    println!(
        "  {label:<40} n={n:>3}  min={min:>9.1}us  median={med:>9.1}us  p95={p95:>9.1}us  mean={mean:>9.1}us  max={max:>9.1}us",
        n = s.n,
        min = us(s.min),
        med = us(s.median),
        p95 = us(s.p95),
        mean = us(s.mean),
        max = us(s.max),
    );
}

/// Time `f()` for `warmup` warmup iterations then `iters` measured ones.
fn time<F: FnMut()>(label: &str, warmup: usize, iters: usize, mut f: F) -> Stats {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    let s = summarise(samples);
    print_stats(label, s);
    s
}

// ---------------------------------------------------------------------------
// Synthetic graph builders
// ---------------------------------------------------------------------------

/// F32 SiLU chain (sigmoid + mul) — the unfused baseline.
fn unfused_silu_graph(n_elements: usize) -> Graph {
    let mut shapes = HashMap::new();
    shapes.insert("x".into(), TensorShape::new(vec![n_elements], Dtype::F32));
    shapes.insert("s".into(), TensorShape::new(vec![n_elements], Dtype::F32));
    shapes.insert("z".into(), TensorShape::new(vec![n_elements], Dtype::F32));
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

/// F32 SiLU as one fused op — the Phase 4 deliverable.
fn fused_silu_graph(n_elements: usize) -> Graph {
    let mut shapes = HashMap::new();
    shapes.insert("x".into(), TensorShape::new(vec![n_elements], Dtype::F32));
    shapes.insert("z".into(), TensorShape::new(vec![n_elements], Dtype::F32));
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

/// F32 GEMM, M=N=64, K=128.
fn gemm_f32_graph(m: usize, n: usize, k: usize) -> Graph {
    let mut shapes = HashMap::new();
    shapes.insert("a".into(), TensorShape::new(vec![m, k], Dtype::F32));
    shapes.insert("b".into(), TensorShape::new(vec![k, n], Dtype::F32));
    shapes.insert("c".into(), TensorShape::new(vec![m, n], Dtype::F32));
    let ops = vec![CompiledOp {
        name: "g".into(),
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

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_silu_fusion(backend: &VulkanBackend) {
    println!("\n=== F32 SiLU: fused vs sigmoid+mul ===");

    for &n in &[1024usize, 16 * 1024, 256 * 1024] {
        println!("\n  elements = {n}");

        // Unfused
        let mut rt = VulkanGraphRuntime::new(unfused_silu_graph(n), backend.clone())
            .expect("unfused rt");
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001) - 1.0).collect();
        rt.set_input_f32("x", &x).expect("set_input");
        time("unfused (sigmoid + mul)", 5, 20, || {
            rt.run().expect("run");
        });

        // Fused
        let mut rt2 =
            VulkanGraphRuntime::new(fused_silu_graph(n), backend.clone()).expect("fused rt");
        rt2.set_input_f32("x", &x).expect("set_input");
        time("fused   (silu_f32)      ", 5, 20, || {
            rt2.run().expect("run");
        });

        // Correctness sanity check (one run, compare).
        rt.run().expect("run");
        rt2.run().expect("run");
        let z_unfused = rt.get_output_f32("z").expect("get");
        let z_fused = rt2.get_output_f32("z").expect("get");
        let max_diff = z_unfused
            .iter()
            .zip(z_fused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("  max |unfused - fused| = {max_diff:.6e}");
    }
}

fn bench_f32_gemm(backend: &VulkanBackend) {
    println!("\n=== F32 GEMM ===");
    for &(m, n, k) in &[(64, 64, 128), (128, 128, 256), (256, 256, 512)] {
        println!("\n  shape M={m} N={n} K={k}");
        let mut rt = VulkanGraphRuntime::new(gemm_f32_graph(m, n, k), backend.clone())
            .expect("gemm rt");
        // Fill with simple values to keep numerics tractable.
        let a: Vec<f32> = (0..m * k).map(|i| ((i & 31) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i & 15) as f32) * 0.01).collect();
        rt.set_input_f32("a", &a).expect("set a");
        rt.set_input_f32("b", &b).expect("set b");
        time("gemm_f32_naive", 5, 20, || {
            rt.run().expect("run");
        });
    }
}

fn bench_int8_gemm(backend: &VulkanBackend) {
    println!("\n=== INT8 GEMM (low-level, packed UINT32) ===");
    for &(m, n, k) in &[(64usize, 64, 128), (128, 128, 256), (256, 256, 512)] {
        println!("\n  shape M={m} N={n} K={k}");
        let a_bytes = m * k;
        let b_bytes = k * n;
        let c_bytes = m * n * 4;

        let a = backend.alloc(a_bytes, BufferKind::Storage).expect("a alloc");
        let b = backend.alloc(b_bytes, BufferKind::Storage).expect("b alloc");
        let mut c = backend.alloc(c_bytes, BufferKind::Storage).expect("c alloc");

        // Fill A and B with a deterministic pattern.
        let a_data: Vec<u8> = (0..a_bytes).map(|i| (i & 0x3f) as u8).collect();
        let b_data: Vec<u8> = (0..b_bytes).map(|i| (i & 0x1f) as u8).collect();
        let mut a = a;
        let mut b = b;
        backend.upload(&mut a, &a_data).expect("upload a");
        backend.upload(&mut b, &b_data).expect("upload b");

        time("gemm_i8_packed", 5, 20, || {
            dragonwing_vulkan::ops::gemm_i8_packed(backend, &a, &b, &mut c, m, n, k)
                .expect("gemm_i8");
            backend.synchronize().expect("sync");
        });
    }
}

fn bench_int8_conv2d(backend: &VulkanBackend) {
    println!("\n=== INT8 Conv2D NHWC (low-level, packed UINT32) ===");
    let configs = [
        // (n, h, w, c_in, c_out, k, stride, pad)
        (1usize, 32, 32, 16, 16, 3, 1, 1),
        (1, 32, 32, 32, 32, 3, 1, 1),
        (1, 16, 16, 64, 64, 3, 1, 1),
    ];
    for &(n, h, w, c_in, c_out, kk, stride, pad) in &configs {
        println!("\n  N={n} H={h} W={w} C_in={c_in} C_out={c_out} K={kk}x{kk} stride={stride} pad={pad}");
        let h_out = (h + 2 * pad - kk) / stride + 1;
        let w_out = (w + 2 * pad - kk) / stride + 1;

        let in_bytes = n * h * w * c_in;
        let kern_bytes = kk * kk * c_in * c_out;
        let out_bytes_i32 = n * h_out * w_out * c_out * 4;
        let out_bytes_packed = n * h_out * w_out * c_out;

        let mut input = backend
            .alloc(in_bytes, BufferKind::Storage)
            .expect("input alloc");
        let mut kernel = backend
            .alloc(kern_bytes, BufferKind::Storage)
            .expect("kernel alloc");
        let mut output_i32 = backend
            .alloc(out_bytes_i32, BufferKind::Storage)
            .expect("output i32 alloc");
        let mut output_packed = backend
            .alloc(out_bytes_packed, BufferKind::Storage)
            .expect("output packed alloc");

        let in_data: Vec<u8> = (0..in_bytes).map(|i| (i & 0x3f) as u8).collect();
        let kern_data: Vec<u8> = (0..kern_bytes).map(|i| (i & 0x1f) as u8).collect();
        backend.upload(&mut input, &in_data).expect("upload in");
        backend
            .upload(&mut kernel, &kern_data)
            .expect("upload kernel");

        // Unfused conv (output INT32, requantize separately)
        time("conv2d_i8 (unfused, INT32 out)", 5, 10, || {
            dragonwing_vulkan::ops::conv2d_i8_nhwc_packed(
                backend,
                &input,
                &kernel,
                &mut output_i32,
                n,
                h,
                w,
                c_in,
                c_out,
                kk,
                kk,
                stride,
                stride,
                pad,
                pad,
            )
            .expect("conv");
            backend.synchronize().expect("sync");
        });

        // Fused conv + requant + relu
        time("conv2d_requant_relu_i8 (fused) ", 5, 10, || {
            dragonwing_vulkan::ops::conv2d_requant_relu_i8_packed(
                backend,
                &input,
                &kernel,
                &mut output_packed,
                n,
                h,
                w,
                c_in,
                c_out,
                kk,
                kk,
                stride,
                stride,
                pad,
                pad,
                0.001,
                true,
            )
            .expect("conv_fused");
            backend.synchronize().expect("sync");
        });
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn print_device_info(backend: &VulkanBackend) {
    println!("=== Device info ===");
    println!("  device : {}", backend.device_name());
    println!("  driver : {}", backend.driver_info());
}

fn main() {
    println!("=== dragonwing-edge device-benchmark (Task 007) ===");

    let backend = match VulkanBackend::new(VulkanConfig::default()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: Vulkan init failed: {e}");
            eprintln!("       (this binary needs a real Vulkan device)");
            std::process::exit(2);
        }
    };

    print_device_info(&backend);

    bench_silu_fusion(&backend);
    bench_f32_gemm(&backend);
    bench_int8_gemm(&backend);
    bench_int8_conv2d(&backend);

    println!("\n=== done ===");
}
