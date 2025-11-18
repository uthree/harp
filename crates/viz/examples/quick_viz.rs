//! Pipelineから直接visualizerを起動する最も簡単な例
//!
//! この例では、GenericPipelineで最適化を有効化し、
//! HarpVizApp::run_from_pipeline()を使ってワンライナーで可視化を起動します。

use harp::backend::c::{CCompiler, CRenderer};
use harp::backend::{GenericPipeline, Pipeline};
use harp::graph::{DType, Graph};
use harp_viz::HarpVizApp;

fn main() -> eframe::Result {
    env_logger::init();

    // OpenMPバックエンド（CRenderer + CCompiler）でPipelineを作成
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 最適化を有効化
    pipeline.enable_graph_optimization = true;
    pipeline.enable_ast_optimization = true;
    pipeline.collect_histories = true; // 履歴を記録

    // カスタマイズ（オプション）
    pipeline.graph_config.beam_width = 4;
    pipeline.graph_config.max_steps = 20;
    pipeline.ast_config.beam_width = 4;
    pipeline.ast_config.max_steps = 20;

    // サンプルグラフを作成
    let graph = create_sample_graph();

    // グラフをコンパイル（最適化履歴が自動的に記録される）
    println!("Compiling graph with optimizations enabled...");
    match pipeline.compile_graph(graph) {
        Ok(_kernel) => {
            println!("Compilation successful!");
            println!("Launching visualizer...");

            // ワンライナーで可視化ウィンドウを起動
            HarpVizApp::run_from_pipeline(&pipeline)
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// サンプルの計算グラフを作成
///
/// グラフ: y = (a + b) * c - d
///         z = reduce_sum(y, axis=0)
fn create_sample_graph() -> Graph {
    let mut graph = Graph::new();

    // 入力ノードを作成
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

    // 計算: y = (a + b) * c - d
    let sum = a + b;
    let product = sum * c;
    let y = product - d;

    // reduce_sum: z = sum(y, axis=0)
    let z = y.reduce_sum(0);

    // 出力ノードを登録
    graph.output("y", y);
    graph.output("z", z);

    graph
}
