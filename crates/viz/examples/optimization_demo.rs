//! グラフ最適化とAST最適化の両方を可視化する統合デモ

use harp::ast::helper::*;
use harp::ast::{AstNode, Literal};
use harp::graph::{DType, Graph};
use harp::opt::ast::rules::{add_commutative, all_algebraic_rules};
use harp::opt::ast::{
    BeamSearchOptimizer as AstBeamSearchOptimizer, RuleBaseSuggester,
    SimpleCostEstimator as AstSimpleCostEstimator,
};
use harp::opt::graph::{
    BeamSearchGraphOptimizer, CompositeSuggester, ParallelStrategyChanger,
    SimpleCostEstimator as GraphSimpleCostEstimator, ViewInsertionSuggester,
};
use harp_viz::HarpVizApp;

fn main() -> eframe::Result {
    env_logger::init();

    println!("=== Harp 最適化デモ ===\n");

    // 1. グラフ最適化を実行
    println!("【1/2】グラフ最適化を実行中...");
    let graph = create_sample_graph();
    log::info!(
        "Sample graph created with {} outputs",
        graph.outputs().len()
    );

    let graph_suggester = CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new().with_transpose(true)),
        Box::new(ParallelStrategyChanger::with_default_strategies()),
    ]);
    let graph_estimator = GraphSimpleCostEstimator::new();

    let graph_optimizer = BeamSearchGraphOptimizer::new(graph_suggester, graph_estimator)
        .with_beam_width(3)
        .with_max_depth(5)
        .with_progress(true);

    let (_optimized_graph, graph_history) = graph_optimizer.optimize_with_history(graph);
    log::info!(
        "Graph optimization complete. {} steps recorded",
        graph_history.len()
    );
    println!("  ✓ グラフ最適化完了: {} ステップ\n", graph_history.len());

    // 2. AST最適化を実行
    println!("【2/2】AST最適化を実行中...");
    let ast = create_sample_ast();
    log::info!("Sample AST created");

    let mut rules = all_algebraic_rules();
    rules.push(add_commutative());

    let ast_suggester = RuleBaseSuggester::new(rules);
    let ast_estimator = AstSimpleCostEstimator::new();

    let ast_optimizer = AstBeamSearchOptimizer::new(ast_suggester, ast_estimator)
        .with_beam_width(5)
        .with_max_depth(10)
        .with_progress(true);

    let (_optimized_ast, ast_history) = ast_optimizer.optimize_with_history(ast);
    log::info!(
        "AST optimization complete. {} steps recorded",
        ast_history.len()
    );
    println!("  ✓ AST最適化完了: {} ステップ\n", ast_history.len());

    // 3. 可視化アプリケーションを起動
    println!("可視化UIを起動中...");
    println!("  - Graph Viewerタブ: グラフ最適化の履歴");
    println!("  - AST Viewerタブ: AST最適化の履歴");
    println!();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Harp Optimization Visualizer - Integrated Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Optimization Visualizer",
        options,
        Box::new(move |_cc| {
            let mut app = HarpVizApp::new();
            // 両方の履歴を読み込む
            app.load_graph_optimization_history(graph_history);
            app.load_ast_optimization_history(ast_history);
            Ok(Box::new(app))
        }),
    )
}

/// デモ用のサンプルグラフを作成
///
/// 以下の計算を行うグラフを作成:
/// ```
/// y = ((a + b) * c) - d
/// z = reduce_sum(y, axis=0)
/// ```
fn create_sample_graph() -> Graph {
    let mut graph = Graph::new();

    // 入力ノードを作成 (形状: [10, 20])
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![10, 20])
        .build();

    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![10, 20])
        .build();

    let c = graph
        .input("c")
        .with_dtype(DType::F32)
        .with_shape(vec![10, 20])
        .build();

    let d = graph
        .input("d")
        .with_dtype(DType::F32)
        .with_shape(vec![10, 20])
        .build();

    // 計算グラフを構築
    // temp1 = a + b
    let temp1 = a + b;

    // temp2 = temp1 * c
    let temp2 = temp1 * c;

    // y = temp2 - d
    let y = temp2 - d;

    // z = reduce_sum(y, axis=0) -> 形状: [20]
    let z = y.reduce_sum(0);

    // 出力ノードを登録
    graph.output("y", y);
    graph.output("z", z);

    graph
}

/// デモ用のサンプルASTを作成
///
/// 以下の式を表現:
/// ```
/// ((2 + 3) * 1) + ((a + 0) * (b + c))
/// ```
fn create_sample_ast() -> AstNode {
    // 左辺: (2 + 3) * 1
    let left = (AstNode::Const(Literal::Isize(2)) + AstNode::Const(Literal::Isize(3)))
        * AstNode::Const(Literal::Isize(1));

    // 右辺: (a + 0) * (b + c)
    let a_plus_zero = var("a") + AstNode::Const(Literal::Isize(0));
    let b_plus_c = var("b") + var("c");
    let right = a_plus_zero * b_plus_c;

    // 全体: left + right
    left + right
}
