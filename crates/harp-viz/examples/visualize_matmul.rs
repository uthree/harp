use harp::ast::DType;
use harp::backend::Backend;
use harp::backend::generic::CBackend;
use harp::graph::Graph;
use harp::graph::ReduceOps;

fn main() {
    // VIZ環境変数をチェック
    if !harp::opt::graph::is_viz_enabled() {
        eprintln!("VIZ environment variable is not set to 1.");
        eprintln!("Run with: VIZ=1 cargo run --example visualize_matmul");
        return;
    }

    println!("=== Matrix Multiplication Visualization Demo ===\n");

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Error: C compiler not available");
        return;
    }

    // グラフ構築
    println!("Building computation graph...");
    let mut graph = Graph::new();
    let m = 64isize;
    let k = 64isize;
    let n = 64isize;

    let a = graph.input(DType::F32, vec![m.into(), k.into()]);
    let b = graph.input(DType::F32, vec![k.into(), n.into()]);

    // Matrix multiplication: A[M,K] @ B[K,N] = C[M,N]
    let a_expanded = a.unsqueeze(2).expand(vec![m.into(), k.into(), n.into()]);
    let b_expanded = b.unsqueeze(0).expand(vec![m.into(), k.into(), n.into()]);
    let multiplied = a_expanded * b_expanded;
    let c = multiplied.sum(1);

    graph.output(c);
    println!("Graph built successfully.\n");

    // 入力データ準備
    println!("Preparing input data...");
    let a_data: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 / 100.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 / 100.0).collect();

    // 最適化をビジュアライザー付きで実行
    println!("Setting up optimizer with visualizer...");
    use harp::opt::graph::{GraphFusionOptimizer, GraphOptimizer};

    let mut optimizer = GraphFusionOptimizer::new()
        .with_logging()
        .auto_visualize(|snapshots| {
            println!(
                "\nLaunching visualizer with {} snapshots...",
                snapshots.len()
            );
            let _ = harp_viz::launch_with_global_snapshots(snapshots);
        });

    optimizer.optimize(&mut graph);

    println!("Optimization completed.");

    // 実行
    println!("Executing optimized graph...");
    let _outputs = backend.execute(
        &graph,
        vec![
            harp::backend::c::CBuffer::from_slice(&a_data, &[m as usize, k as usize], DType::F32),
            harp::backend::c::CBuffer::from_slice(&b_data, &[k as usize, n as usize], DType::F32),
        ],
    );

    println!("Execution completed.");
}
