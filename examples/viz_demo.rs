//! Visualization demo for Eclat optimization history
//!
//! This example demonstrates the visualization of both graph-level and AST-level
//! optimization history using a matrix multiplication example with GPU backend.
//!
//! Run with:
//! ```bash
//! cargo run --example viz_demo
//! ```
//!
//! With debug logging:
//! ```bash
//! RUST_LOG=info cargo run --example viz_demo
//! ```

use eclat::ast::DType;
use eclat::backend::TargetBackend;
use eclat::graph::{Expr, GraphNode, input};
use eclat::lowerer::{ElementwiseReduceFusion, Lowerer, ViewFusion};
use eclat::opt::ast::optimizer::BeamSearchOptimizer as AstBeamSearchOptimizer;
use eclat::opt::ast::suggesters::{
    CompositeSuggester as AstCompositeSuggester, CseSuggester, GroupParallelizationSuggester,
    LocalParallelizationSuggester, LoopFusionSuggester, LoopInterchangeSuggester,
    LoopTilingSuggester, SharedMemorySuggester,
};
use eclat::opt::graph::{
    CompositeSuggester, FusionSuggester, GraphBeamSearchOptimizer, GraphOptimizer,
    MatMulDetectorSuggester,
};

fn main() {
    env_logger::init();

    // Determine target backend based on platform
    let target_backend = get_target_backend();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║        Eclat Optimization Visualizer Demo                      ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  This demo visualizes the optimization process for matrix      ║");
    println!("║  multiplication: C = A @ B                                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Target backend: {}", target_backend);
    println!();

    // Matrix dimensions
    let m: i64 = 64;
    let k: i64 = 128;
    let n: i64 = 64;

    println!("Matrix dimensions:");
    println!("  A: [{}, {}]", m, k);
    println!("  B: [{}, {}]", k, n);
    println!("  C = A @ B: [{}, {}]", m, n);
    println!();

    // Create input tensors
    let a = input(vec![Expr::Const(m), Expr::Const(k)], DType::F32).with_name("A");
    let b = input(vec![Expr::Const(k), Expr::Const(n)], DType::F32).with_name("B");

    // Build matrix multiplication graph using primitives
    println!("Building matmul graph using primitives...");
    let c = build_matmul_graph(&a, &b, m, k, n);
    println!("  Initial graph built");
    println!();

    // ========================================
    // Phase 1: Graph-level optimization
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 1: Graph-level optimization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Create graph suggesters
    let graph_suggester = CompositeSuggester::new(vec![
        Box::new(MatMulDetectorSuggester::new()),
        Box::new(FusionSuggester::new(ViewFusion, "view_fusion")),
        Box::new(FusionSuggester::new(
            ElementwiseReduceFusion,
            "elementwise_reduce_fusion",
        )),
    ]);

    // Run graph optimization with history recording
    let mut graph_optimizer = GraphBeamSearchOptimizer::new(graph_suggester)
        .without_progress()
        .with_max_steps(20)
        .with_beam_width(5)
        .with_target_backend(target_backend)
        .with_history();

    let optimized_graph = graph_optimizer.optimize(vec![c]);
    let graph_history = graph_optimizer.take_history().unwrap();

    println!("  Graph optimization recorded {} steps", graph_history.len());
    println!("  Target backend: {}", graph_history.target_backend());
    for (i, snapshot) in graph_history.iter().enumerate() {
        let suggester = snapshot
            .suggester_name
            .as_deref()
            .unwrap_or("(initial)");
        println!(
            "    Step {}: cost={:.2e}, suggester={}, alts={}",
            i,
            snapshot.cost,
            suggester,
            snapshot.alternatives.len()
        );
    }
    println!();

    // ========================================
    // Phase 2: Lower to AST
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 2: Lowering graph to AST");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut lowerer = Lowerer::new();
    let program = lowerer
        .lower(&optimized_graph)
        .expect("Lowering should succeed");
    println!("  AST generated successfully");
    println!();

    // ========================================
    // Phase 3: AST-level optimization (GPU)
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 3: AST-level optimization ({} backend)", target_backend);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Create AST suggesters with GPU parallelization
    let ast_suggester = AstCompositeSuggester::new(vec![
        // Loop transformations
        Box::new(LoopFusionSuggester::new()),
        Box::new(LoopTilingSuggester::new()),
        Box::new(LoopInterchangeSuggester::new()),
        // GPU parallelization
        Box::new(GroupParallelizationSuggester::new()),
        Box::new(LocalParallelizationSuggester::new()),
        Box::new(SharedMemorySuggester::new()),
        // General optimizations
        Box::new(CseSuggester::new()),
    ]);

    // Run AST optimization with history recording
    // Enable record_all_steps to record every step for visualization
    let mut ast_optimizer = AstBeamSearchOptimizer::new(ast_suggester)
        .without_progress()
        .with_max_steps(15)
        .with_beam_width(5)
        .with_target_backend(target_backend)
        .with_record_all_steps(true);

    let (_optimized_ast, ast_history) = ast_optimizer.optimize_with_history(program);

    println!("  AST optimization recorded {} steps", ast_history.len());
    println!("  Target backend: {}", ast_history.target_backend());
    for (i, snapshot) in ast_history.snapshots().iter().enumerate() {
        let suggester = snapshot
            .suggester_name
            .as_deref()
            .unwrap_or("(initial)");
        println!(
            "    Step {}: cost={:.2e}, suggester={}, alts={}",
            i,
            snapshot.cost,
            suggester,
            snapshot.alternatives.len()
        );
    }
    println!();

    // ========================================
    // Launch visualization
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Launching GUI visualization...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Controls:");
    println!("  ←/→ or H/L : Navigate steps");
    println!("  ↑/↓ or K/J : Select candidate");
    println!("  1/2        : AST view / Graph view");
    println!();

    // Launch visualization with appropriate renderer based on target backend from history
    launch_visualization(ast_history, graph_history);
}

/// Determine the target backend based on platform
fn get_target_backend() -> TargetBackend {
    #[cfg(target_os = "macos")]
    {
        TargetBackend::Metal
    }
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        TargetBackend::Cuda
    }
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        TargetBackend::Generic
    }
}

/// Launch visualization with the appropriate renderer based on target backend
fn launch_visualization(
    ast_history: eclat::opt::ast::history::OptimizationHistory,
    graph_history: eclat::opt::graph::GraphOptimizationHistory,
) {
    // Use the target backend from AST history to select renderer
    let target = ast_history.target_backend();

    match target {
        #[cfg(target_os = "macos")]
        TargetBackend::Metal => {
            use eclat_backend_metal::MetalRenderer;
            let renderer = MetalRenderer::new();
            if let Err(e) = eclat_viz::run_with_both(ast_history, graph_history, renderer) {
                eprintln!("Visualization error: {}", e);
            }
        }
        // TODO: Add CUDA and OpenCL renderer support when available
        // #[cfg(feature = "cuda")]
        // TargetBackend::Cuda => { ... }
        _ => {
            use eclat::backend::renderer::GenericRenderer;
            let renderer = GenericRenderer::new();
            if let Err(e) = eclat_viz::run_with_both(ast_history, graph_history, renderer) {
                eprintln!("Visualization error: {}", e);
            }
        }
    }
}

/// Build matrix multiplication graph using primitives
///
/// Matrix multiplication: C[i,j] = Σ_k A[i,k] * B[k,j]
///
/// Steps:
/// 1. A: [M, K] → unsqueeze(2) → [M, K, 1] → expand(2, N) → [M, K, N]
/// 2. B: [K, N] → permute → unsqueeze → permute → expand → [M, K, N]
/// 3. Elementwise multiply: A * B → [M, K, N]
/// 4. Sum over axis 1 (K): [M, 1, N]
/// 5. Squeeze axis 1: [M, N]
fn build_matmul_graph(a: &GraphNode, b: &GraphNode, m: i64, _k: i64, n: i64) -> GraphNode {
    // A: [M, K] → [M, K, 1] → [M, K, N]
    let a_expanded = a
        .unsqueeze(2)
        .expand(2, Expr::Const(n))
        .with_name("A_expanded");

    // B: [K, N] → [N, K] → [1, N, K] → [1, K, N] → [M, K, N]
    let b_expanded = b
        .permute(&[1, 0])
        .unsqueeze(0)
        .permute(&[0, 2, 1])
        .expand(0, Expr::Const(m))
        .with_name("B_expanded");

    // Elementwise multiply and sum over K dimension
    let product = (&a_expanded * &b_expanded).with_name("product");
    let summed = product.sum(1).with_name("summed");
    summed.squeeze(1).with_name("matmul_result")
}
