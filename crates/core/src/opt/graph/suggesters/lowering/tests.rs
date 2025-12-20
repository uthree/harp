//! LoweringSuggesterのテスト

use super::*;
use crate::ast::{AstNode, DType as AstDType, Scope, helper::*};
use crate::graph::shape::View;
use crate::graph::{DType, ReduceOp};

/// テスト用: FusedElementwiseReduceノードを作成
fn test_fused_elementwise_reduce(
    inputs: Vec<GraphNode>,
    expr: AstNode,
    reduce_op: ReduceOp,
    axes: Vec<usize>,
) -> GraphNode {
    let dtype = inputs[0].dtype.clone();
    let view = inputs[0].view.clone();
    let mut new_shape = view.shape().to_vec();

    let mut sorted_axes = axes.clone();
    sorted_axes.sort();
    for &axis in sorted_axes.iter().rev() {
        new_shape.remove(axis);
    }
    let reduced_view = View::contiguous(new_shape);

    GraphNode::new(
        dtype,
        GraphOp::FusedElementwiseReduce {
            expr,
            reduce_op,
            axes: sorted_axes,
            reduce_strategy: None,
        },
        inputs,
        reduced_view,
    )
}

#[test]
fn test_lower_elementwise_add() {
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = a + b;
    graph.output("c", c);

    let suggestions = suggester.suggest(&graph);

    // Sequential専用モードでは1つの候補が生成される
    assert_eq!(
        suggestions.len(),
        1,
        "Sequential-only mode should generate exactly 1 candidate"
    );

    // 候補のグラフでKernelノードが使われていることを確認
    let new_graph = &suggestions[0].graph;
    let outputs = new_graph.outputs();
    let output = outputs.get("c").unwrap();
    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Candidate should use Kernel node"
    );
}

#[test]
fn test_lower_reduce_sum() {
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = a.reduce_sum(1);
    graph.output("b", b);

    let suggestions = suggester.suggest(&graph);

    // Sequential専用モードでは1つの候補が生成される
    assert_eq!(
        suggestions.len(),
        1,
        "Sequential-only mode should generate exactly 1 candidate"
    );

    // 候補のグラフでKernelノードが使われていることを確認
    let new_graph = &suggestions[0].graph;
    let outputs = new_graph.outputs();
    let output = outputs.get("b").unwrap();
    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Candidate should use Kernel node"
    );
}

#[test]
fn test_skip_already_custom() {
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);

    // 既にKernelノードを使用
    let custom_func = function(
        None::<String>,
        vec![],
        AstDType::Tuple(vec![]),
        block(vec![], Scope::new()),
    );
    let b = a.custom_function(custom_func);
    graph.output("b", b);

    let suggestions = suggester.suggest(&graph);

    // Kernelノードはスキップされるので候補なし
    assert_eq!(suggestions.len(), 0);
}

#[test]
fn test_beam_search_with_lowering() {
    use crate::opt::graph::{
        BeamSearchGraphOptimizer, CompositeSuggester, GraphCostEstimator, SimpleCostEstimator,
    };

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = a + b;
    graph.output("c", c);

    // 初期コストを確認
    let estimator = SimpleCostEstimator::new();
    let initial_cost = estimator.estimate(&graph);
    println!("Initial cost: {}", initial_cost);

    // LoweringSuggesterのみでBeamSearch
    let composite = CompositeSuggester::new(vec![Box::new(LoweringSuggester::new())]);

    let optimizer = BeamSearchGraphOptimizer::new(composite)
        .with_beam_width(4)
        .with_max_steps(10);

    let (optimized, history) = optimizer.optimize_with_history(graph);

    println!("Optimization steps: {}", history.len());
    for (i, snapshot) in history.snapshots().iter().enumerate() {
        println!("  Step {}: cost = {}", i, snapshot.cost);
    }

    // 最適化後のグラフを確認
    let outputs = optimized.outputs();
    let output = outputs.get("c").unwrap();
    println!("Final output op: {:?}", std::mem::discriminant(&output.op));

    // Kernelノードに変換されているはず
    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Output should be Kernel node, but got {:?}",
        output.op
    );
}

#[test]
fn test_beam_search_with_fusion_and_lowering() {
    use crate::opt::graph::{BeamSearchGraphOptimizer, CompositeSuggester, FusionSuggester};

    // (a + b) * c + d というElementwiseチェーンを作成
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = graph.input("c", DType::F32, vec![10, 20]);
    let d = graph.input("d", DType::F32, vec![10, 20]);

    let sum = a + b;
    let mul = sum * c;
    let result = mul + d;
    graph.output("result", result);

    // FusionとLoweringの両方を含むSuggester
    let suggesters: Vec<Box<dyn crate::opt::graph::GraphSuggester>> = vec![
        Box::new(FusionSuggester::new()),
        Box::new(LoweringSuggester::new()),
    ];
    let composite = CompositeSuggester::new(suggesters);

    let optimizer = BeamSearchGraphOptimizer::new(composite)
        .with_beam_width(4)
        .with_max_steps(50);

    let (optimized, history) = optimizer.optimize_with_history(graph);

    println!("Optimization steps: {}", history.len());
    for (i, snapshot) in history.snapshots().iter().enumerate() {
        println!("  Step {}: cost = {}", i, snapshot.cost);
    }

    // 最適化後のグラフを確認
    let outputs = optimized.outputs();
    let output = outputs.get("result").unwrap();
    println!("Final output op: {:?}", std::mem::discriminant(&output.op));
    println!("Final output src count: {}", output.src.len());

    // Kernelノードに変換されているはず
    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Output should be Kernel node, but got {:?}",
        output.op
    );

    // 全ての入力が単一のKernelノードに融合されているはず (4入力 + 1出力Buffer = 5)
    assert!(
        output.src.len() <= 5,
        "Kernel node should have at most 5 src nodes (4 inputs + 1 output buffer), but got {}",
        output.src.len()
    );
}

#[test]
fn test_lower_fused_elementwise_reduce_multiple_axes() {
    let suggester = LoweringSuggester::new();

    // 複数軸縮約: [M, N, K1, K2] で K1, K2軸を縮約
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![3, 4, 5, 6]); // [M, N, K1, K2]
    let b = graph.input("b", DType::F32, vec![3, 4, 5, 6]);

    // FusedElementwiseReduce: multiply + sum over axes 2, 3
    let expr = wildcard("0") * wildcard("1");
    let c = test_fused_elementwise_reduce(vec![a, b], expr, ReduceOp::Sum, vec![2, 3]);
    graph.output("c", c);

    let suggestions = suggester.suggest(&graph);

    // Sequential専用モードでは1つの候補が生成される
    assert_eq!(
        suggestions.len(),
        1,
        "Sequential-only mode should generate exactly 1 candidate"
    );

    // 候補のグラフでKernelノードが使われていることを確認
    let new_graph = &suggestions[0].graph;
    let outputs = new_graph.outputs();
    let output = outputs.get("c").unwrap();
    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Candidate should use Kernel node"
    );

    // 出力形状が正しいことを確認: [3, 4] (K1, K2軸が縮約された)
    let output_shape = output.view.shape();
    assert_eq!(output_shape.len(), 2, "Output should be 2D");
    assert_eq!(output_shape[0], 3.into(), "Output dim 0 should be 3");
    assert_eq!(output_shape[1], 4.into(), "Output dim 1 should be 4");
}
