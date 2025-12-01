// シンプルなテスト: View -> Custom -> Sink パターンのテスト
use harp::graph::{DType, Graph, GraphOp};
use harp::opt::graph::{GraphSuggester, ViewMergeSuggester, ProgramRootAbsorptionSuggester};
use harp::opt::graph::suggesters::LoweringSuggester;

fn main() {
    // シンプルなグラフ: a + b
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let sum = a + b;
    
    // Viewを適用
    let permuted = sum.view(sum.view.clone().permute(vec![1, 0]));
    graph.output("result", permuted);
    
    println!("=== Initial Graph ===");
    print_sink(&graph);
    
    // 1. Lowering
    let lowering = LoweringSuggester::new();
    let lowered_graphs = lowering.suggest(&graph);
    println!("\n=== After Lowering ({} suggestions) ===", lowered_graphs.len());
    
    if !lowered_graphs.is_empty() {
        let lowered = &lowered_graphs[0];
        print_sink(lowered);
        
        // 2. ViewMerge
        let view_merge = ViewMergeSuggester::new();
        let merged_graphs = view_merge.suggest(lowered);
        println!("\n=== After ViewMerge ({} suggestions) ===", merged_graphs.len());
        
        if !merged_graphs.is_empty() {
            let merged = &merged_graphs[0];
            print_sink(merged);
            
            // 3. ProgramRootAbsorption
            let absorption = ProgramRootAbsorptionSuggester::new();
            let absorbed_graphs = absorption.suggest(merged);
            println!("\n=== After ProgramRootAbsorption ({} suggestions) ===", absorbed_graphs.len());
            
            if !absorbed_graphs.is_empty() {
                print_sink(&absorbed_graphs[0]);
            }
        } else {
            println!("ViewMerge produced no suggestions!");
            // ViewMergeなしでProgramRootAbsorptionを試す
            let absorption = ProgramRootAbsorptionSuggester::new();
            let absorbed_graphs = absorption.suggest(lowered);
            println!("\n=== After ProgramRootAbsorption without ViewMerge ({} suggestions) ===", absorbed_graphs.len());
        }
    }
}

fn print_sink(graph: &Graph) {
    if let Some(sink) = graph.sink() {
        println!("Sink src count: {}", sink.src.len());
        for (i, src) in sink.src.iter().enumerate() {
            print_node(src, 1, &format!("src[{}]", i));
        }
    }
}

fn print_node(node: &harp::graph::GraphNode, depth: usize, prefix: &str) {
    let indent = "  ".repeat(depth);
    let op_name = match &node.op {
        GraphOp::Buffer { name } => format!("Buffer({})", name),
        GraphOp::Kernel { .. } => "Custom".to_string(),
        GraphOp::View(_) => "View".to_string(),
        GraphOp::Elementwise { op, .. } => format!("Elementwise({:?})", op),
        _ => format!("{:?}", std::mem::discriminant(&node.op)),
    };
    println!("{}{}: {} (src_count={})", indent, prefix, op_name, node.src.len());
    for (i, src) in node.src.iter().enumerate() {
        print_node(src, depth + 1, &format!("src[{}]", i));
    }
}
