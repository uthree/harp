use harp::graph::{DType, Graph};
use harp::opt::graph::{
    BeamSearchGraphOptimizer, CompositeSuggester, LoweringSuggester,
    BufferAbsorptionSuggester, SinkAbsorptionSuggester, SimpleCostEstimator,
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);
    let c = a + b;
    graph.output("c", c);
    
    println!("=== Initial Graph ===");
    print_graph(&graph);
    
    // Lowering
    let lowering = LoweringSuggester::new();
    let suggestions = lowering.suggest(&graph);
    let lowered = &suggestions[0];
    
    println!("\n=== After Lowering ===");
    print_graph(lowered);
    
    // BufferAbsorption
    let buffer_absorber = BufferAbsorptionSuggester::new();
    let suggestions = buffer_absorber.suggest(lowered);
    let absorbed = &suggestions[0];
    
    println!("\n=== After BufferAbsorption ===");
    print_graph(absorbed);
    
    // SinkAbsorption
    let sink_absorber = SinkAbsorptionSuggester::new();
    let suggestions = sink_absorber.suggest(absorbed);
    if suggestions.is_empty() {
        println!("\n=== No SinkAbsorption suggestions ===");
    } else {
        let final_graph = &suggestions[0];
        println!("\n=== After SinkAbsorption ===");
        print_graph(final_graph);
    }
}

fn print_graph(graph: &Graph) {
    println!("Inputs: {:?}", graph.input_metas().iter().map(|m| &m.name).collect::<Vec<_>>());
    println!("Outputs: {:?}", graph.outputs().keys().collect::<Vec<_>>());
    
    if let Some(sink) = graph.sink() {
        println!("Sink exists, src count: {}", sink.src.len());
        for (i, src) in sink.src.iter().enumerate() {
            println!("  src[{}]: {:?}", i, std::mem::discriminant(&src.op));
            match &src.op {
                harp::graph::GraphOp::Buffer { name } => println!("    Buffer: {}", name),
                harp::graph::GraphOp::Custom { input_buffers, .. } => {
                    println!("    Custom input_buffers: {:?}", input_buffers.as_ref().map(|v| v.iter().map(|b| &b.name).collect::<Vec<_>>()));
                    println!("    Custom src count: {}", src.src.len());
                    for (j, s) in src.src.iter().enumerate() {
                        match &s.op {
                            harp::graph::GraphOp::Buffer { name } => println!("      src[{}]: Buffer({})", j, name),
                            _ => println!("      src[{}]: {:?}", j, std::mem::discriminant(&s.op)),
                        }
                    }
                }
                _ => {}
            }
        }
    } else {
        for (name, node) in graph.outputs() {
            println!("Output '{}': {:?}", name, std::mem::discriminant(&node.op));
        }
    }
}
