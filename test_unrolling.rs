use harp::backend::openmp::{CCompiler, CRenderer};
use harp::backend::GenericPipeline;
use harp::graph::{DType, Graph};

fn main() {
    harp::opt::log_capture::init_with_env_logger();
    
    let mut graph = Graph::new();
    
    // 簡単なテスト: input + const
    let input = graph
        .input("input")
        .with_dtype(DType::F32)
        .with_shape(vec![256, 64])
        .build();
    
    // スカラー定数は自動的にブロードキャストされる
    let result = input + 6.0f32;
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
