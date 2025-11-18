use harp::backend::c::{CCompiler, CPipeline, CRenderer};
use harp::backend::Pipeline;
use harp::graph::{DType, Graph, GraphNode};

fn main() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![10, 10])
        .build();

    // スカラー定数
    let const1: GraphNode = 2.0f32.into();
    let const2: GraphNode = 3.0f32.into();
    let scale = const1 * const2;
    
    println!("scale shape: {:?}", scale.view.shape());
    println!("scale strides: {:?}", scale.view.strides());

    let result = a + scale;
    graph.output("result", result);

    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = CPipeline::new(renderer, compiler);
    
    // グラフ最適化を有効化
    pipeline.enable_graph_optimization = true;
    pipeline.enable_ast_optimization = false;
    
    let optimized_graph = pipeline.optimize_graph(graph);
    
    // 最適化後のグラフを確認
    println!("Optimized graph nodes: {}", optimized_graph.nodes.len());
    for (i, node) in optimized_graph.nodes.iter().enumerate() {
        println!("Node {}: op={:?}, shape={:?}, strides={:?}", 
                 i, node.op, node.view.shape(), node.view.strides());
    }
}
