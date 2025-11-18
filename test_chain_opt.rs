use harp::backend::c::{CCompiler, CPipeline, CRenderer};
use harp::backend::{Pipeline, Renderer};
use harp::graph::{DType, Graph};

fn main() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![8, 8])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![8, 8])
        .build();
    let c = graph
        .input("c")
        .with_dtype(DType::F32)
        .with_shape(vec![8, 8])
        .build();

    let temp = a + b;
    let result = temp * c;
    graph.output("result", result);

    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = CPipeline::new(renderer, compiler);
    
    pipeline.enable_graph_optimization = true;
    pipeline.enable_ast_optimization = false;
    
    let optimized_graph = pipeline.optimize_graph(graph);
    let program = pipeline.lower_to_program(optimized_graph);
    let code = pipeline.renderer().render(&program);
    
    println!("=== With Graph Optimization ===");
    println!("{}", code.as_str());
}
