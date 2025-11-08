//! GenericPipelineを使った最適化と可視化のデモ
//!
//! 行列積を含む複雑な計算グラフを作成し、GenericPipelineで最適化を実行して
//! その過程を可視化します。

use harp::backend::openmp::{CCompiler, CRenderer};
use harp::backend::{GenericPipeline, Pipeline};
use harp::graph::{DType, Graph};
use harp::opt::ast::rules::all_algebraic_rules;
use harp::opt::ast::{
    BeamSearchOptimizer as AstBeamSearchOptimizer, RuleBaseSuggester,
    SimpleCostEstimator as AstSimpleCostEstimator,
};
use harp::opt::graph::{
    BeamSearchGraphOptimizer, CompositeSuggester, FusionSuggester, ParallelStrategyChanger,
    SimpleCostEstimator as GraphSimpleCostEstimator, ViewInsertionSuggester,
};
use harp_viz::HarpVizApp;

fn main() -> eframe::Result {
    env_logger::init();

    println!("=== Harp GenericPipeline 最適化デモ ===\n");
    println!("このデモでは、行列積を含む複雑な計算グラフを最適化します。");
    println!("最適化の各ステップがGenericPipelineに記録され、可視化されます。\n");

    // GenericPipelineを作成
    println!("【1/4】GenericPipelineを初期化中...");
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);
    println!("  ✓ Pipeline作成完了\n");

    // 複雑な計算グラフを作成（複数の演算を含む）
    println!("【2/4】複雑な計算グラフを構築中...");
    let graph = create_complex_computation_graph();
    println!("  ✓ グラフ作成完了");
    println!("    - 入力数: {}", graph.inputs().len());
    println!("    - 出力数: {}", graph.outputs().len());
    println!();

    // グラフ最適化を実行して履歴を記録
    println!("【3/4】グラフ最適化を実行中...");
    let graph_suggester = CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new().with_transpose(true)),
        Box::new(FusionSuggester::new()),
        Box::new(ParallelStrategyChanger::with_default_strategies()),
    ]);
    let graph_estimator = GraphSimpleCostEstimator::new();

    let graph_optimizer = BeamSearchGraphOptimizer::new(graph_suggester, graph_estimator)
        .with_beam_width(4)
        .with_max_depth(8)
        .with_progress(true);

    let (optimized_graph, graph_history) = graph_optimizer.optimize_with_history(graph);
    pipeline.set_graph_optimization_history(graph_history);

    println!("  ✓ グラフ最適化完了");
    println!(
        "    - 最適化ステップ数: {}",
        pipeline
            .last_graph_optimization_history()
            .map_or(0, |h| h.len())
    );
    println!();

    // グラフをProgramにlower
    let program = pipeline.lower_to_program(optimized_graph.clone());
    println!("【4/4】AST最適化を実行中...");

    // ASTの最適化を実行
    let ast_suggester = RuleBaseSuggester::new(all_algebraic_rules());
    let ast_estimator = AstSimpleCostEstimator::new();

    let ast_optimizer = AstBeamSearchOptimizer::new(ast_suggester, ast_estimator)
        .with_beam_width(5)
        .with_max_depth(10)
        .with_progress(true);

    // プログラムの各関数を最適化
    let mut total_ast_steps = 0;

    for (_name, func) in &program.functions {
        let body = &*func.body;
        let (_optimized_body, ast_history) = ast_optimizer.optimize_with_history(body.clone());
        total_ast_steps += ast_history.len();

        // 最初の関数の履歴を保存
        if pipeline.last_ast_optimization_history().is_none() {
            pipeline.set_ast_optimization_history(ast_history);
        }
    }

    println!("  ✓ AST最適化完了");
    println!("    - 最適化ステップ数: {}", total_ast_steps);
    println!();

    // 最適化の統計情報を表示
    print_optimization_stats(&pipeline);

    // 可視化アプリケーションを起動
    println!("\n可視化UIを起動中...");
    println!("  - Graph Viewerタブ: グラフ最適化の履歴を表示");
    println!("  - AST Viewerタブ: AST最適化の履歴を表示");
    println!("  - 矢印キーで前後のステップに移動できます");
    println!();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("Harp Pipeline Optimization Visualizer - Matrix Computation Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Pipeline Visualizer",
        options,
        Box::new(move |_cc| {
            let mut app = HarpVizApp::new();
            // Pipelineから履歴を読み込む（参照として）
            app.load_from_pipeline(&pipeline);
            Ok(Box::new(app))
        }),
    )
}

/// 複雑な計算グラフを作成
///
/// 以下の計算を実装:
/// ```
/// # 複数のテンソル演算を組み合わせ
/// x1 = a + b
/// x2 = x1 * c
/// x3 = x2 - d
/// y = reduce_sum(x3, axis=0)
/// z = reduce_sum(x3, axis=1)
/// w = y + (z expanded)
/// ```
fn create_complex_computation_graph() -> Graph {
    let mut graph = Graph::new();

    let rows = 128;
    let cols = 256;

    // 入力: a [rows, cols]
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![rows, cols])
        .build();

    // 入力: b [rows, cols]
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![rows, cols])
        .build();

    // 入力: c [rows, cols]
    let c = graph
        .input("c")
        .with_dtype(DType::F32)
        .with_shape(vec![rows, cols])
        .build();

    // 入力: d [rows, cols]
    let d = graph
        .input("d")
        .with_dtype(DType::F32)
        .with_shape(vec![rows, cols])
        .build();

    // 計算グラフを構築
    // x1 = a + b
    let x1 = a + b;

    // x2 = x1 * c
    let x2 = x1.clone() * c;

    // x3 = x2 - d
    let x3 = x2.clone() - d;

    // y = reduce_sum(x3, axis=0) -> [cols]
    let y = x3.clone().reduce_sum(0);

    // z = reduce_sum(x3, axis=1) -> [rows]
    let z = x3.clone().reduce_sum(1);

    // 複数の中間結果を出力
    graph.output("x1", x1);
    graph.output("x2", x2);
    graph.output("x3", x3);
    graph.output("y", y);
    graph.output("z", z);

    graph
}

/// 最適化の統計情報を表示
fn print_optimization_stats(pipeline: &GenericPipeline<CRenderer, CCompiler>) {
    println!("=== 最適化統計 ===");

    if let Some(graph_history) = pipeline.last_graph_optimization_history() {
        println!("\n【グラフ最適化】");
        println!("  ステップ数: {}", graph_history.len());

        if let (Some(first), Some(last)) = (
            graph_history.get(0),
            graph_history.get(graph_history.len() - 1),
        ) {
            let cost_reduction = first.cost - last.cost;
            let cost_reduction_percent = if first.cost > 0.0 {
                (cost_reduction / first.cost) * 100.0
            } else {
                0.0
            };

            println!("  初期コスト: {:.2}", first.cost);
            println!("  最終コスト: {:.2}", last.cost);
            println!(
                "  コスト削減: {:.2} ({:.1}%)",
                cost_reduction, cost_reduction_percent
            );
        }

        // コスト遷移を表示
        println!("\n  コスト遷移:");
        for i in 0..graph_history.len().min(5) {
            if let Some(snapshot) = graph_history.get(i) {
                println!("    Step {}: {:.2}", snapshot.step, snapshot.cost);
            }
        }
        if graph_history.len() > 5 {
            println!("    ... ({} steps total)", graph_history.len());
        }
    }

    if let Some(ast_history) = pipeline.last_ast_optimization_history() {
        println!("\n【AST最適化】");
        println!("  ステップ数: {}", ast_history.len());

        if let (Some(first), Some(last)) = (
            ast_history.snapshots().first(),
            ast_history.snapshots().last(),
        ) {
            let cost_reduction = first.cost - last.cost;
            let cost_reduction_percent = if first.cost > 0.0 {
                (cost_reduction / first.cost) * 100.0
            } else {
                0.0
            };

            println!("  初期コスト: {:.2}", first.cost);
            println!("  最終コスト: {:.2}", last.cost);
            println!(
                "  コスト削減: {:.2} ({:.1}%)",
                cost_reduction, cost_reduction_percent
            );
        }
    }
}
