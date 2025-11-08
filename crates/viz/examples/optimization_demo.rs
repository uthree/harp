//! グラフ最適化の各ステップを可視化するデモ

use harp::graph::{DType, Graph};
use harp::opt::graph::{
    BeamSearchGraphOptimizer, CompositeSuggester, ParallelStrategyChanger, SimpleCostEstimator,
    ViewInsertionSuggester,
};
use harp_viz::HarpVizApp;

fn main() -> eframe::Result {
    env_logger::init();

    // デモ用のサンプルグラフを作成
    let graph = create_sample_graph();
    log::info!(
        "Sample graph created with {} outputs",
        graph.outputs().len()
    );

    // 最適化器を設定
    let suggester = CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new().with_transpose(true)),
        Box::new(ParallelStrategyChanger::with_default_strategies()),
    ]);
    let estimator = SimpleCostEstimator::new();

    let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
        .with_beam_width(3)
        .with_max_depth(5)
        .with_progress(true);

    // 最適化を実行して履歴を取得
    log::info!("Running optimization...");
    let (_optimized_graph, history) = optimizer.optimize_with_history(graph);
    log::info!("Optimization complete. {} steps recorded", history.len());

    // 可視化アプリケーションを起動
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Harp Optimization Visualizer - Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Optimization Visualizer",
        options,
        Box::new(move |_cc| {
            let mut app = HarpVizApp::new();
            app.load_optimization_history(history);
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
