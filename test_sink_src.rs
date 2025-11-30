use harp::graph::{DType, Graph, GraphOp};
use harp::backend::pipeline::{SuggesterFlags, optimize_graph_with_history};
use harp::opt::graph::SimpleCostEstimator;

fn main() {
    env_logger::init();
    
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);
    let c = a + b;
    graph.output("c", c);
    
    println!("=== Initial Graph ===");
    println!("Input metas: {:?}", graph.input_metas().iter().map(|m| &m.name).collect::<Vec<_>>());
    println!("Output metas: {:?}", graph.output_metas().iter().map(|m| &m.name).collect::<Vec<_>>());
    if let Some(sink) = graph.sink() {
        println!("Sink src count: {}", sink.src.len());
        for (i, src) in sink.src.iter().enumerate() {
            let op_name = match &src.op {
                GraphOp::Buffer { name } => format!("Buffer({})", name),
                GraphOp::Custom { .. } => "Custom".to_string(),
                GraphOp::Elementwise { .. } => "Elementwise".to_string(),
                _ => format!("{:?}", src.op),
            };
            println!("  src[{}]: {}", i, op_name);
        }
    }
    
    // Optimize
    let flags = SuggesterFlags::single_stage();
    let estimator = SimpleCostEstimator::new();
    let (optimized, _) = optimize_graph_with_history(
        graph,
        flags,
        estimator,
        4,
        50,
        false,
    );
    
    println!("\n=== After Optimization ===");
    println!("Input metas: {:?}", optimized.input_metas().iter().map(|m| &m.name).collect::<Vec<_>>());
    println!("Output metas: {:?}", optimized.output_metas().iter().map(|m| &m.name).collect::<Vec<_>>());
    if let Some(sink) = optimized.sink() {
        println!("Sink src count: {}", sink.src.len());
        for (i, src) in sink.src.iter().enumerate() {
            let op_name = match &src.op {
                GraphOp::Buffer { name } => format!("Buffer({})", name),
                GraphOp::Custom { .. } => "Custom".to_string(),
                GraphOp::Elementwise { .. } => "Elementwise".to_string(),
                _ => format!("{:?}", src.op),
            };
            println!("  src[{}]: {}", i, op_name);
        }
        
        if let GraphOp::Sink { ast, outputs } = &sink.op {
            println!("Outputs: {:?}", outputs);
            if let harp::ast::AstNode::Program { functions, .. } = ast {
                println!("Program functions: {}", functions.len());
                for f in functions {
                    match f {
                        harp::ast::AstNode::Function { name, params, .. } => {
                            println!("  Function {:?}: {} params", name, params.len());
                            for p in params {
                                println!("    - {} ({:?})", p.name, p.dtype);
                            }
                        }
                        harp::ast::AstNode::Kernel { name, params, .. } => {
                            println!("  Kernel {:?}: {} params", name, params.len());
                            for p in params {
                                println!("    - {} ({:?})", p.name, p.dtype);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
