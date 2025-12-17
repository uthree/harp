//! SIMD最適化のデモ
//!
//! 要素ごとの演算でSIMD最適化が正しく行われることを確認します。
//! (a + b) * c + d という演算チェーンを最適化し、SIMD版が選択されることを示します。

use harp::backend::opencl::OpenCLRenderer;
use harp::backend::{create_multi_phase_optimizer, MultiPhaseConfig, Renderer};
use harp::graph::{DType, Graph};
use harp::lowerer::extract_program;
use harp::opt::ast::rules::all_algebraic_rules;
use harp::opt::ast::{
    AstOptimizer, BeamSearchOptimizer as AstBeamSearchOptimizer,
    CompositeSuggester as AstCompositeSuggester, FunctionInliningSuggester,
    GroupParallelizationSuggester, LocalParallelizationSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseOptimizer,
};
use harp::opt::graph::GraphOptimizer;
use harp_viz::{HarpVizApp, RendererType};

fn main() -> eframe::Result {
    harp::opt::log_capture::init_with_env_logger();

    // Elementwise演算グラフを作成: result = (a + b) * c + d
    let graph = create_elementwise_graph(1024, 1024);

    println!("=== SIMD Optimization Demo ===");
    println!("Graph: result = (a + b) * c + d");
    println!("Shape: [1024, 1024]");
    println!();

    // Phase 1: Graph optimization (SIMD幅4と8を候補として使用)
    let config = MultiPhaseConfig::new()
        .with_beam_width(4)
        .with_max_steps(5000)
        .with_progress(true)
        .with_collect_logs(true)
        .with_simd_widths(vec![4, 8]); // SIMD幅4と8の候補を生成

    let optimizer = create_multi_phase_optimizer(config);
    let (optimized_graph, graph_history) = optimizer.optimize_with_history(graph);

    // Lowering結果のログを表示
    println!("\n=== Graph Optimization History (Lowering phase) ===");
    for snapshot in graph_history.snapshots() {
        if snapshot.description.contains("Lowering") {
            println!(
                "Step {}: {} (suggester: {:?}, cost: {:.2})",
                snapshot.step, snapshot.description, snapshot.suggester_name, snapshot.cost
            );
        }
    }

    // Phase 2: Lower to AST
    let program = extract_program(optimized_graph);

    // Phase 3: AST optimization
    let (optimized_program, ast_history) = optimize_ast_with_history(program);

    // 生成コード表示
    println!("\n=== Generated OpenCL Code ===");
    let renderer = OpenCLRenderer::new();
    let code = renderer.render(&optimized_program);
    let code_str: String = code.into();

    // SIMD命令の使用状況を確認
    if code_str.contains("vload") || code_str.contains("vstore") {
        println!("SIMD optimization: ENABLED (using vector load/store)");
    } else {
        println!("SIMD optimization: DISABLED (scalar operations only)");
    }
    println!();
    println!("{}", code_str);

    // 可視化UI起動
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("Harp SIMD Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp SIMD Demo",
        options,
        Box::new(move |_| {
            let mut app = HarpVizApp::with_renderer_type(RendererType::OpenCL);
            app.load_graph_optimization_history(graph_history);
            app.load_ast_optimization_history(ast_history);
            app.load_optimized_ast(optimized_program);
            Ok(Box::new(app))
        }),
    )
}

/// AST最適化を履歴付きで実行
fn optimize_ast_with_history(
    program: harp::ast::AstNode,
) -> (harp::ast::AstNode, harp::opt::ast::OptimizationHistory) {
    // Phase 1: Rule-based optimization
    let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules());
    let rule_optimized = rule_optimizer.optimize(program);

    // Phase 2: Loop optimization with beam search
    let loop_suggester = AstCompositeSuggester::new(vec![
        Box::new(LoopTilingSuggester::new()),
        Box::new(LoopInliningSuggester::new()),
        Box::new(LoopInterchangeSuggester::new()),
        Box::new(LoopFusionSuggester::new()),
        Box::new(FunctionInliningSuggester::with_default_limit()),
        Box::new(GroupParallelizationSuggester::new()),
        Box::new(LocalParallelizationSuggester::new()),
    ]);

    let loop_optimizer = AstBeamSearchOptimizer::new(loop_suggester)
        .with_beam_width(4)
        .with_max_steps(5000)
        .with_progress(true);

    loop_optimizer.optimize_with_history(rule_optimized)
}

/// Elementwise演算グラフを作成: result = (a + b) * c + d
fn create_elementwise_graph(rows: usize, cols: usize) -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![rows, cols]);
    let b = graph.input("b", DType::F32, vec![rows, cols]);
    let c = graph.input("c", DType::F32, vec![rows, cols]);
    let d = graph.input("d", DType::F32, vec![rows, cols]);

    // result = (a + b) * c + d
    let sum = a + b;
    let mul = sum * c;
    let result = mul + d;

    graph.output("result", result);
    graph
}
