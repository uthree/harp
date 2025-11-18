use harp::backend::c::{CCompiler, CPipeline, CRenderer};
use harp::backend::{Pipeline, Renderer};
use harp::graph::{DType, Graph};

fn main() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![8, 16])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![8, 16])
        .build();

    let mul_result = a * b;
    let result = mul_result.reduce_sum(1);
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
    
    println!("{}", code.as_str());
}
