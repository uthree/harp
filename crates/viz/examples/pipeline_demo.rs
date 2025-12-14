//! GenericPipelineを使った最適化と可視化のデモ
//!
//! 1024x1024の行列積を最適化し、その過程を可視化します。
//! RuntimeSelectorによる実測値ベース最適化を使用します。

use harp::backend::opencl::{OpenCLCompiler, OpenCLRenderer};
use harp::backend::GenericPipeline;
use harp::graph::{DType, Graph};
use harp_viz::{HarpVizApp, RendererType};

fn main() -> eframe::Result {
    harp::opt::log_capture::init_with_env_logger();

    // Pipeline初期化
    let mut pipeline = GenericPipeline::new(OpenCLRenderer::new(), OpenCLCompiler::new());
    pipeline.collect_histories = true;
    pipeline.graph_config.show_progress = true;
    pipeline.ast_config.show_progress = true;

    // 実測値ベース最適化を有効化(するとめっちゃ重くなるので注意)
    // pipeline.enable_runtime_selector();

    // 1024x1024 行列積グラフを作成
    let graph = create_matmul_graph(1024);

    // 最適化実行
    let (program, ast_histories) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Optimization failed");

    // 生成コード表示
    println!("\n=== Generated OpenCL Code ===");
    println!("{}", OpenCLRenderer::new().render_program(&program));

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
            if let Some(h) = pipeline.histories.combined_graph_history() {
                app.load_graph_optimization_history(h);
            }
            if let Some(h) = ast_histories.get("program") {
                app.load_ast_optimization_history(h.clone());
            }
            app.load_optimized_ast(program);
            Ok(Box::new(app))
        }),
    )
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
