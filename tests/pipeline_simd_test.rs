use harp::backend::c::{CCompiler, CRenderer};
use harp::backend::{GenericPipeline, OptimizationConfig, Pipeline};
use harp::graph::{DType, Graph};

#[test]
fn test_pipeline_selects_simd() {
    // GenericPipelineがSIMD化を選択するか確認
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![100]);
    let b = graph.input("b", DType::F32, vec![100]);
    let c = a + b;
    graph.output("c", c);

    let renderer = CRenderer::new();
    let compiler = CCompiler::new();

    let mut pipeline = GenericPipeline::new(renderer, compiler);
    pipeline.enable_graph_optimization = true;
    pipeline.graph_config = OptimizationConfig {
        beam_width: 10,
        max_steps: 20,
        show_progress: false,
    };

    // グラフ最適化を実行（コンパイルはスキップ）
    let optimized = pipeline.optimize_graph(graph);

    // 出力ノードのelementwise_strategiesを確認
    let output = optimized.outputs().get("c").unwrap();
    println!("Pipeline optimization result:");
    println!(
        "  elementwise_strategies: {:?}",
        output.elementwise_strategies
    );

    // SIMD化されているか確認
    if !output.elementwise_strategies.is_empty() {
        let simd_width = output.elementwise_strategies[0].simd_width();
        println!("  SIMD width: {}", simd_width);
        assert!(simd_width > 1, "SIMD化が選択されるべき");
    } else {
        panic!("elementwise_strategiesが空です");
    }
}
