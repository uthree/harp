use harp::backend::c::{CCompiler, CPipeline, CRenderer};
use harp::backend::{Pipeline, Renderer};
use harp::graph::{DType, Graph};

fn main() {
    // test_complex_graph_optimizationと同じグラフを作成
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![6, 6])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![6, 6])
        .build();
    let c = graph
        .input("c")
        .with_dtype(DType::F32)
        .with_shape(vec![6, 6])
        .build();
    let d = graph
        .input("d")
        .with_dtype(DType::F32)
        .with_shape(vec![6, 6])
        .build();

    let temp1 = a + b;
    let temp2 = temp1 * c;
    let result = temp2 + d;
    graph.output("result", result);

    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = CPipeline::new(renderer, compiler);

    pipeline.enable_graph_optimization = true;
    pipeline.enable_ast_optimization = true;

    let optimized_graph = pipeline.optimize_graph(graph.clone());
    let program = pipeline.lower_to_program(optimized_graph);
    let optimized_program = pipeline.optimize_program(program);
    let code = pipeline.renderer().render(&optimized_program);

    println!("=== Optimized Code ===");
    println!("{}", code.as_str());
}
