use harp::graph::{DType, ElementwiseOp, Graph, GraphOp};
use harp::opt::graph::{
    BeamSearchGraphOptimizer, CompositeSuggester, FusionSuggester, GraphOptimizer, GraphSuggester,
    KernelMergeSuggester, LoweringSuggester, SimpleCostEstimator,
};
use std::collections::HashSet;

fn main() {
    env_logger::init();

    // 複数の Custom ノードが生成されるグラフ
    // Reduce を含むため、Fusion で統合されず、別々の Custom になる
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = graph.input("c", DType::F32, vec![10]);

    // (a + b).reduce_sum(1) -> [10] の Custom
    let sum = &a + &b;
    let reduced = sum.reduce_sum(1); // [10, 20] -> [10]

    // reduced + c -> [10] の別の Custom
    let result = &reduced + &c;
    graph.output("result", result);

    println!("期待: reduce と elementwise が別々の Custom になり、マージ可能");

    println!("=== 初期グラフ ===");
    for (name, output) in graph.outputs() {
        println!(
            "  Output '{}': {:?}",
            name,
            std::mem::discriminant(&output.op)
        );
    }

    // Step 1: Fusion + Lowering で Custom に変換
    // ログ収集を有効化してコスト遷移を確認
    let suggesters: Vec<Box<dyn GraphSuggester>> = vec![
        Box::new(FusionSuggester::new()),
        Box::new(LoweringSuggester::new()),
    ];
    let composite = CompositeSuggester::new(suggesters);
    let estimator = SimpleCostEstimator::new();
    let optimizer = BeamSearchGraphOptimizer::new(composite, estimator)
        .with_beam_width(4)
        .with_max_steps(50)
        .with_collect_logs(true);
    let (lowered_graph, history) = optimizer.optimize_with_history(graph);

    // 各ステップのコストとノード構成を表示
    println!("\n=== 最適化履歴 ===");
    for (i, snapshot) in history.snapshots().iter().enumerate() {
        let (fn_count, prog_count) = count_custom_nodes(&snapshot.graph);
        let output = snapshot.graph.outputs().get("result").unwrap();
        let output_op_name = match &output.op {
            GraphOp::Custom { .. } => "Custom",
            GraphOp::Elementwise { op, .. } => match op {
                ElementwiseOp::Add => "Elementwise::Add",
                _ => "Elementwise::Other",
            },
            _ => "Other",
        };
        println!(
            "  Step {}: cost={:.2}, Custom(Fn)={}, Custom(Prog)={}, output={}",
            i, snapshot.cost, fn_count, prog_count, output_op_name
        );
    }

    println!("\n=== Lowering後のグラフ ===");
    let (fn_count, prog_count) = count_custom_nodes(&lowered_graph);
    println!("  Custom(Function): {}", fn_count);
    println!("  Custom(Program): {}", prog_count);

    for (name, output) in lowered_graph.outputs() {
        println!(
            "  Output '{}': {:?}",
            name,
            std::mem::discriminant(&output.op)
        );
        println!("    view is_contiguous: {}", output.view.is_contiguous());
        println!("    src count: {}", output.src.len());
        for (i, src) in output.src.iter().enumerate() {
            println!("      src[{}]: {:?}", i, std::mem::discriminant(&src.op));
            println!("        view is_contiguous: {}", src.view.is_contiguous());
            if let GraphOp::Buffer { name } = &src.op {
                println!("        -> Buffer name: {}", name);
            }
            if matches!(&src.op, GraphOp::Custom { .. }) {
                println!("        -> Custom node (can be merged)");
            }
        }
    }

    // Step 2: KernelMergeSuggester でマージ
    let merge_suggester = KernelMergeSuggester::new();
    let suggestions = merge_suggester.suggest(&lowered_graph);
    println!("\n=== KernelMergeSuggester ===");
    println!("  Suggestions count: {}", suggestions.len());

    if suggestions.is_empty() {
        println!("  警告: マージ候補が見つかりません！");

        // 詳細デバッグ
        println!("\n=== 詳細デバッグ: ノード構造 ===");
        let mut visited = HashSet::new();
        for output in lowered_graph.outputs().values() {
            debug_nodes(output, &mut visited, 0);
        }
    } else {
        println!("  マージ候補が見つかりました。マージを実行...");

        let merge_optimizer = BeamSearchGraphOptimizer::new(
            CompositeSuggester::new(vec![Box::new(KernelMergeSuggester::new())]),
            SimpleCostEstimator::new(),
        )
        .with_beam_width(4)
        .with_max_steps(20);
        let merged_graph = merge_optimizer.optimize(lowered_graph);

        let (fn_count, prog_count) = count_custom_nodes(&merged_graph);
        println!("\n=== マージ後のグラフ ===");
        println!("  Custom(Function): {}", fn_count);
        println!("  Custom(Program): {}", prog_count);
    }
}

fn count_custom_nodes(graph: &Graph) -> (usize, usize) {
    let mut visited = HashSet::new();
    let mut function_count = 0;
    let mut program_count = 0;

    fn visit(
        node: &harp::graph::GraphNode,
        visited: &mut HashSet<*const harp::graph::GraphNodeData>,
        fn_count: &mut usize,
        prog_count: &mut usize,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        for src in &node.src {
            visit(src, visited, fn_count, prog_count);
        }

        if let GraphOp::Custom { ast } = &node.op {
            match ast {
                harp::ast::AstNode::Function { .. } => *fn_count += 1,
                harp::ast::AstNode::Program { .. } => *prog_count += 1,
                _ => {}
            }
        }
    }

    for output in graph.outputs().values() {
        visit(
            output,
            &mut visited,
            &mut function_count,
            &mut program_count,
        );
    }

    (function_count, program_count)
}

fn debug_nodes(
    node: &harp::graph::GraphNode,
    visited: &mut HashSet<*const harp::graph::GraphNodeData>,
    depth: usize,
) {
    let ptr = node.as_ptr();
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    let indent = "  ".repeat(depth);
    println!("{}Node: {:?}", indent, std::mem::discriminant(&node.op));
    println!("{}  src count: {}", indent, node.src.len());

    for (i, src) in node.src.iter().enumerate() {
        print!("{}  src[{}]: ", indent, i);
        match &src.op {
            GraphOp::Custom { .. } => println!("Custom (mergeable!)"),
            GraphOp::Buffer { name } => println!("Buffer({})", name),
            _ => println!("{:?}", std::mem::discriminant(&src.op)),
        }
    }

    for src in &node.src {
        debug_nodes(src, visited, depth + 1);
    }
}
