// デバッグスクリプトを一時的に作成
use harp_core::graph::{Graph, DType};
use harp_core::lowerer::{extract_program, create_lowering_optimizer};
use harp_core::opt::graph::GraphOptimizer;
use harp_core::ast::AstNode;

fn main() {
    // test_transpose_contiguous_add と同じグラフを作成
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);
    let b = graph.input("b", DType::F32, vec![3, 2]);
    
    // Transpose a: [2, 3] -> [3, 2]
    let a_transposed_view = a.view.clone().permute(vec![1, 0]);
    let a_transposed = a.view(a_transposed_view);
    
    // Make contiguous
    let a_contiguous = a_transposed.contiguous();
    
    // Add
    let c = a_contiguous + b;
    graph.output("c", c);

    // Optimize and extract program
    let optimizer = create_lowering_optimizer(4, 5000);
    let (optimized_graph, _) = optimizer.optimize_with_history(graph);
    let program = extract_program(optimized_graph);
    
    // Print execution waves
    if let AstNode::Program { execution_waves, functions, .. } = &program {
        println!("Number of functions: {}", functions.len());
        for (i, func) in functions.iter().enumerate() {
            match func {
                AstNode::Kernel { name, .. } | AstNode::Function { name, .. } => {
                    println!("Function {}: {:?}", i, name);
                }
                _ => {}
            }
        }
        
        println!("\nExecution waves:");
        for (i, wave) in execution_waves.iter().enumerate() {
            println!("Wave {}:", i);
            for call in wave {
                println!("  kernel: {}, inputs: {:?}, outputs: {:?}", 
                         call.kernel_name, call.inputs, call.outputs);
            }
        }
    }
}
