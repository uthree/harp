use harp::backend::openmp::{CCompiler, CRenderer};
use harp::backend::GenericPipeline;
use harp::graph::{DType, Graph, GraphNode};

fn main() {
    harp::opt::log_capture::init_with_env_logger();
    
    let mut graph = Graph::new();
    
    // 簡単なテスト: input + const
    let input = graph
        .input("input")
        .with_dtype(DType::F32)
        .with_shape(vec![256, 64])
        .build();
    
    let scale = GraphNode::constant(6.0);
    let scale_unsqueezed = scale.view(scale.view.clone().unsqueeze(0).unsqueeze(0));
    let scale_expanded = scale_unsqueezed.view(
        scale_unsqueezed.view.clone().expand(vec![256.into(), 64.into()])
    );
    
    let result = input + scale_expanded;
    graph.output("result", result);
    
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);
    pipeline.enable_graph_optimization = true;
    pipeline.enable_ast_optimization = true;
    
    let (program, _) = pipeline.optimize_graph_with_all_histories(graph).unwrap();
    
    let code = pipeline.renderer.render_program(&program);
    println!("Generated C code:\n{}", code);
}
