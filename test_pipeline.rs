use harp::backend::openmp::{CRenderer, CCompiler};
use harp::backend::{GenericPipeline, GraphOptimizationConfig, AstOptimizationConfig};
use harp::graph::{DType, Graph, GraphNode};
use harp::ast::renderer::render_ast_with;

fn main() {
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    
    let graph_config = GraphOptimizationConfig {
        beam_width: 4,
        max_steps: 100,
        show_progress: false,
    };
    
    let ast_config = AstOptimizationConfig {
        beam_width: 4,
        max_steps: 10000,
        show_progress: false,
    };
    
    let mut pipeline = GenericPipeline::new(renderer.clone(), compiler)
        .with_graph_optimization_config(graph_config)
        .with_ast_optimization_config(ast_config);
    
    // 定数演算を含むグラフを作成（pipeline_demoと同じ）
    let mut graph = Graph::new();
    
    let m = 64;
    let k = 32;
    let n = 64;
    
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![m, k])
        .build();
    
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![k, n])
        .build();
    
    // 定数演算（これが問題の原因かも）
    let const1 = GraphNode::constant(2.0);
    let const2 = GraphNode::constant(3.0);
    let scale = const1 * const2;
    
    let result = a + b;
    
    graph.output("result", result);
    
    // 最適化を実行
    let (program, _) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize");
    
    // コードを生成
    let code = render_ast_with(&program, &renderer);
    println!("{}", code);
}
