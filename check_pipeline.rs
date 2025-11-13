use harp::backend::openmp::CRenderer;
use harp::graph::{DType, Graph, GraphNode};
use harp::ast::renderer::render_ast_with;

fn matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    use harp::graph::shape::Expr;

    let a_shape = a.view.shape();
    let b_shape = b.view.shape();

    let m = a_shape[0].clone();
    let k_a = a_shape[1].clone();
    let n = b_shape[1].clone();

    let a_unsqueezed = a.view(a.view.clone().unsqueeze(2));
    let b_unsqueezed = b.view(b.view.clone().unsqueeze(0));

    let expanded_shape = vec![m.clone(), k_a.clone(), n.clone()];
    let a_expanded = a_unsqueezed.view(a_unsqueezed.view.clone().expand(expanded_shape.clone()));
    let b_expanded = b_unsqueezed.view(b_unsqueezed.view.clone().expand(expanded_shape));

    let product = a_expanded * b_expanded;
    product.reduce_sum(1)
}

fn main() {
    use harp::backend::{GenericPipeline, GraphOptimizationConfig, AstOptimizationConfig};
    
    let renderer = CRenderer::new();
    let compiler = harp::backend::openmp::CCompiler::new();
    
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
    
    let mut graph = Graph::new();
    
    let m = 64;
    let k = 32;
    let n = 64;
    let p = 128;
    
    let a = graph.input("a").with_dtype(DType::F32).with_shape(vec![m, k]).build();
    let b = graph.input("b").with_dtype(DType::F32).with_shape(vec![k, n]).build();
    let c = graph.input("c").with_dtype(DType::F32).with_shape(vec![m, n]).build();
    let d = graph.input("d").with_dtype(DType::F32).with_shape(vec![n, p]).build();
    
    let const1 = GraphNode::constant(2.0);
    let const2 = GraphNode::constant(3.0);
    let scale_scalar = const1 * const2;
    
    let scale_unsqueezed = scale_scalar.view(scale_scalar.view.clone().unsqueeze(0).unsqueeze(0));
    let scale = scale_unsqueezed.view(scale_unsqueezed.view.clone().expand(vec![m.into(), n.into()]));
    
    let temp1 = matmul(a, b);
    let temp2 = temp1 + c;
    let temp3 = temp2 * scale;
    let result = matmul(temp3, d);
    
    graph.output("result", result);
    
    let (program, _) = pipeline.optimize_graph_with_all_histories(graph).expect("Failed to optimize");
    
    let code = render_ast_with(&program, &renderer);
    
    // kernel_2 を探して表示
    for line in code.lines() {
        if line.contains("kernel_2") {
            let start = code.lines().position(|l| l.contains("void kernel_2")).unwrap();
            for (i, line) in code.lines().enumerate() {
                if i >= start && i < start + 20 {
                    println!("{}", line);
                }
                if i >= start && line.contains("}") && line.trim() == "}" {
                    break;
                }
            }
            break;
        }
    }
}
