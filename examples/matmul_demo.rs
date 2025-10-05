use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::{Graph, ReduceOps};
use std::time::Instant;

fn main() {
    println!("=== Matrix Multiplication Demo ===\n");

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Error: C compiler not available");
        return;
    }

    // デモサイズ: 512x512 × 512x512
    let m = 512isize;
    let k = 512isize;
    let n = 512isize;

    println!("Matrix dimensions:");
    println!("  A: {}x{}", m, k);
    println!("  B: {}x{}", k, n);
    println!("  C = A × B: {}x{}\n", m, n);

    // グラフ構築
    println!("Building computation graph...");
    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![m.into(), k.into()]);
    let b = graph.input(DType::F32, vec![k.into(), n.into()]);

    // Matrix multiplication: A[M,K] @ B[K,N] = C[M,N]
    // A: [M, K] -> [M, K, 1] -> [M, K, N]
    // B: [K, N] -> [1, K, N] -> [M, K, N]
    // Multiply: [M, K, N]
    // Sum over K: [M, N]
    let a_expanded = a.unsqueeze(2).expand(vec![m.into(), k.into(), n.into()]);
    let b_expanded = b.unsqueeze(0).expand(vec![m.into(), k.into(), n.into()]);
    let multiplied = a_expanded * b_expanded;
    let c = multiplied.sum(1); // Sum along K dimension

    graph.output(c);
    println!("Graph built successfully.\n");

    // 入力データ準備
    println!("Preparing input data ({} elements total)...", m * k + k * n);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 / 100.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 / 100.0).collect();

    println!("Input data prepared.\n");

    // 実行（最適化あり）
    println!("Executing with optimization enabled...");
    let start = Instant::now();
    let outputs = backend.execute(
        &graph,
        vec![
            harp::backend::c::CBuffer::from_slice(
                &a_data,
                &[m as usize, k as usize],
                DType::F32,
            ),
            harp::backend::c::CBuffer::from_slice(
                &b_data,
                &[k as usize, n as usize],
                DType::F32,
            ),
        ],
    );
    let duration = start.elapsed();
    println!("Execution completed in {:.3}s", duration.as_secs_f64());

    // 結果の検証
    let c_data = outputs[0].to_vec::<f32>();
    println!("Output shape: {:?}", outputs[0].shape());
    println!("Output size: {} elements\n", c_data.len());

    // サンプル結果を表示
    let n_usize = n as usize;
    println!("Sample output values:");
    println!("  C[0,0]     = {:.6}", c_data[0]);
    println!("  C[0,1]     = {:.6}", c_data[1]);
    println!("  C[1,0]     = {:.6}", c_data[n_usize]);
    let last = (m - 1) as usize;
    println!("  C[63,63]   = {:.6}", c_data[63 * n_usize + 63]);
    println!(
        "  C[{},{}]   = {:.6}",
        last,
        last,
        c_data[last * n_usize + last]
    );

    // 簡易検証: C[0,0]を手動計算と比較
    let mut expected_c00 = 0.0f32;
    for i in 0..k {
        expected_c00 += a_data[i as usize] * b_data[i as usize * n_usize];
    }
    println!("\nVerification:");
    println!("  Expected C[0,0] = {:.6}", expected_c00);
    println!("  Actual C[0,0]   = {:.6}", c_data[0]);
    let match_result = (c_data[0] - expected_c00).abs() < 1e-4;
    println!("  Match: {}", match_result);

    if match_result {
        println!("\n✓ Matrix multiplication successful!");
        println!("  Total elements computed: {}", c_data.len());
        println!("  Computation time: {:.3}s", duration.as_secs_f64());
        println!(
            "  Throughput: {:.2} MFLOPS",
            (2.0 * m as f64 * k as f64 * n as f64) / (duration.as_secs_f64() * 1e6)
        );
    }

    println!("\n=== Demo Complete ===");
}
