//! パイプラインを使った最適化と可視化のデモ
//!
//! 1024x1024の行列積を最適化し、その過程を可視化します。

use harp::backend::opencl::OpenCLRenderer;
use harp::backend::{create_multi_phase_optimizer, MultiPhaseConfig, Renderer};
use harp::graph::{DType, Graph};
use harp::lowerer::extract_program;
use harp::opt::ast::rules::all_algebraic_rules;
use harp::opt::ast::{
    BeamSearchOptimizer as AstBeamSearchOptimizer, CompositeSuggester as AstCompositeSuggester,
    FunctionInliningSuggester, GroupParallelizationSuggester, LocalParallelizationSuggester,
    LoopFusionSuggester, LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
    RuleBaseSuggester,
};
use harp::opt::graph::GraphOptimizer;
use harp_viz::{HarpVizApp, RendererType};

fn main() -> eframe::Result {
    harp::opt::log_capture::init_with_env_logger();

    // 1024x1024 行列積グラフを作成
    let graph = create_matmul_graph(1024);

    // Phase 1: Graph optimization
    // Note: unroll_factorsを使わず、AST最適化のタイル化+ループ展開で
    // アンロールと同等の効果を得る
    let config = MultiPhaseConfig::new()
        .with_beam_width(4)
        .with_max_steps(5000)
        .with_progress(true)
        .with_collect_logs(true);

    let optimizer = create_multi_phase_optimizer(config);
    let (optimized_graph, graph_history) = optimizer.optimize_with_history(graph);

    // Phase 2: Lower to AST
    let program = extract_program(optimized_graph);

    // Phase 3: AST optimization
    let (optimized_program, ast_history) = optimize_ast_with_history(program);

    // 生成コード表示
    println!("\n=== Generated OpenCL Code ===");
    let renderer = OpenCLRenderer::new();
    let code = renderer.render(&optimized_program);
    println!("{}", Into::<String>::into(code));

    // 可視化UI起動
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("Harp Pipeline Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Pipeline Demo",
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
    let loop_suggester = AstCompositeSuggester::new(vec![
        Box::new(LoopTilingSuggester::new()),
        Box::new(LoopInliningSuggester::new()),
        Box::new(LoopInterchangeSuggester::new()),
        Box::new(LoopFusionSuggester::new()),
        Box::new(FunctionInliningSuggester::with_default_limit()),
        Box::new(GroupParallelizationSuggester::new()),
        Box::new(LocalParallelizationSuggester::new()),
        Box::new(RuleBaseSuggester::new(all_algebraic_rules())),
    ]);

    let heuristic_optimizer = AstBeamSearchOptimizer::new(loop_suggester)
        .with_beam_width(4)
        .with_max_steps(5000)
        .with_progress(true);

    heuristic_optimizer.optimize_with_history(program)
}

/// NxN 行列積グラフを作成: C = A @ B
fn create_matmul_graph(n: usize) -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![n, n]);
    let b = graph.input("b", DType::F32, vec![n, n]);
    let c = a.matmul(b);
    graph.output("c", c);
    graph
}
