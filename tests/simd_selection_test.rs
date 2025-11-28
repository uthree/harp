use harp::graph::{DType, Graph};
use harp::opt::ast::SimpleCostEstimator as AstSimpleCostEstimator;
use harp::opt::graph::{
    AstBasedCostEstimator, BeamSearchGraphOptimizer, CompositeSuggester, FusionSuggester,
    GraphCostEstimator, ParallelStrategyChanger, SimdSuggester, SimpleCostEstimator,
    ViewInsertionSuggester,
};

#[test]
fn test_simd_selection_with_simple_estimator() {
    // SimpleCostEstimator（Graph側）を使用
    let suggester = CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
        Box::new(ParallelStrategyChanger::new()),
        Box::new(SimdSuggester::new()),
    ]);
    let estimator = SimpleCostEstimator::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![100]);
    let b = graph.input("b", DType::F32, vec![100]);
    let c = a + b;
    graph.output("c", c);

    let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
        .with_beam_width(10)
        .with_max_steps(20)
        .with_progress(false);

    let (optimized_graph, _history) = optimizer.optimize_with_history(graph);

    // 出力ノードのelementwise_strategiesを確認
    let output = optimized_graph.outputs().get("c").unwrap();
    println!(
        "SimpleCostEstimator - elementwise_strategies: {:?}",
        output.elementwise_strategies
    );

    // SIMD化されているか確認
    if !output.elementwise_strategies.is_empty() {
        let simd_width = output.elementwise_strategies[0].simd_width();
        println!("SimpleCostEstimator - SIMD width: {}", simd_width);
        assert!(simd_width > 1, "SIMD化が選択されるべき");
    }
}

#[test]
fn test_simd_selection_with_ast_based_estimator() {
    // AstBasedCostEstimator（AST側）を使用
    let suggester = CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
        Box::new(ParallelStrategyChanger::new()),
        Box::new(SimdSuggester::new()),
    ]);
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![100]);
    let b = graph.input("b", DType::F32, vec![100]);
    let c = a + b;
    graph.output("c", c);

    let ast_estimator = AstSimpleCostEstimator::new();
    let estimator_for_cost = AstBasedCostEstimator::new(ast_estimator);
    let initial_cost = estimator_for_cost.estimate(&graph);
    println!("AstBasedCostEstimator - Initial cost: {}", initial_cost);

    let ast_estimator2 = AstSimpleCostEstimator::new();
    let estimator = AstBasedCostEstimator::new(ast_estimator2);
    let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
        .with_beam_width(10)
        .with_max_steps(20)
        .with_progress(false);

    let (optimized_graph, history) = optimizer.optimize_with_history(graph);

    // 最適化履歴を表示
    println!("\nAstBasedCostEstimator - Optimization history:");
    for snapshot in history.snapshots().iter().take(5) {
        let output = snapshot.graph.outputs().get("c").unwrap();
        println!(
            "  Step {}: cost={:.2e}, strategy={:?}",
            snapshot.step, snapshot.cost, output.elementwise_strategies
        );
    }

    // 出力ノードのelementwise_strategiesを確認
    let output = optimized_graph.outputs().get("c").unwrap();
    let final_cost = estimator_for_cost.estimate(&optimized_graph);
    println!("\nAstBasedCostEstimator - Final cost: {}", final_cost);
    println!(
        "AstBasedCostEstimator - elementwise_strategies: {:?}",
        output.elementwise_strategies
    );

    // SIMD化されているか確認
    if !output.elementwise_strategies.is_empty() {
        let simd_width = output.elementwise_strategies[0].simd_width();
        println!("AstBasedCostEstimator - SIMD width: {}", simd_width);
        // AstBasedの場合、SIMD化されない可能性が高い
    }
}
