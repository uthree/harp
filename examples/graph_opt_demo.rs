/// Graph optimization demo
/// このプログラムは、Graph レベルの最適化（特にベクトル化）が正しく動作することを確認します。
use harp::ast::DType;
use harp::graph::Graph;
use harp::lowerer::LoweringOptimizer;
use harp::opt::graph::optimizer::BeamSearchOptimizer;
use harp::opt::graph::suggester::VectorizationSuggester;

fn main() {
    println!("=== Graph Optimization Demo ===\n");

    // 1. グラフを作成
    let mut graph = Graph::new();

    // 入力: [1024] の F32 配列 2つ
    let a = graph.input(DType::F32, vec![1024.into()]);
    let b = graph.input(DType::F32, vec![1024.into()]);

    // 要素ごとの演算: c = a + b
    let c = a + b;

    // 出力として登録
    graph.output(c.clone());

    println!("Original Graph:");
    println!("  Input A: [1024] F32");
    println!("  Input B: [1024] F32");
    println!("  Output C = A + B");
    println!("  Node strategy: {:?}\n", c.strategy);

    // 2. ベクトル化の提案を取得
    println!("Generating vectorization suggestions...");
    let suggestions = VectorizationSuggester::suggest(&graph);
    println!(
        "  Generated {} vectorization suggestions\n",
        suggestions.len()
    );

    for (i, suggested_graph) in suggestions.iter().enumerate() {
        println!("Suggestion {}:", i + 1);
        if let Some(output_node) = suggested_graph.outputs.first() {
            if let Some(strategy) = &output_node.strategy {
                println!("  Vectorize: {:?}", strategy.vectorize);
                println!("  Unroll: {:?}", strategy.unroll);
                println!("  Parallelize: {:?}", strategy.parallelize);
            }
        }
    }
    println!();

    // 3. BeamSearchOptimizer を使って最適化
    println!("Running BeamSearchOptimizer...");
    let optimizer = BeamSearchOptimizer::new();
    let optimized_graph = optimizer.optimize(&graph);

    println!("Optimized Graph:");
    if let Some(output_node) = optimized_graph.outputs.first() {
        if let Some(strategy) = &output_node.strategy {
            println!("  Applied strategy:");
            println!("    Vectorize: {:?}", strategy.vectorize);
            println!("    Unroll: {:?}", strategy.unroll);
            println!("    Parallelize: {:?}", strategy.parallelize);
        } else {
            println!("  No strategy applied");
        }
    }
    println!();

    // 4. Lowering してコード生成
    println!("Lowering to AST...");
    let optimizer = LoweringOptimizer::with_default_config();
    let original_ast = optimizer.optimize_and_lower(&graph);
    let optimized_ast = optimizer.optimize_and_lower(&optimized_graph);

    println!(
        "Original AST children count: {}",
        original_ast.children().len()
    );
    println!(
        "Optimized AST children count: {}",
        optimized_ast.children().len()
    );
    println!();

    println!("=== Demo completed successfully ===");
}
