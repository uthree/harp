//! LoweringSuggesterのテスト

use super::*;
use crate::ast::{AstNode, DType as AstDType, Scope, helper::*};
use crate::graph::shape::View;
use crate::graph::{DType, ReduceOp};
use crate::opt::graph::suggesters::CanonicalFormSuggester;

/// グラフを正規形に変換するヘルパー関数
/// Elementwise/Reduce/FusedElementwiseをFusedElementwiseReduceに変換
fn canonicalize_graph(graph: &Graph) -> Graph {
    let canonical = CanonicalFormSuggester::new();
    let mut current = graph.clone();

    // すべてのノードが正規化されるまで繰り返す
    loop {
        let suggestions = canonical.suggest(&current);
        if suggestions.is_empty() {
            break;
        }
        current = suggestions[0].graph.clone();
    }
    current
}

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

    // 正規化してからlowering
    let canonical_graph = canonicalize_graph(&graph);
    let suggestions = suggester.suggest(&canonical_graph);

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

    // 正規化してからlowering
    let canonical_graph = canonicalize_graph(&graph);
    let suggestions = suggester.suggest(&canonical_graph);

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

    // CanonicalFormSuggesterとLoweringSuggesterでBeamSearch
    let composite = CompositeSuggester::new(vec![
        Box::new(CanonicalFormSuggester::new()),
        Box::new(LoweringSuggester::new()),
    ]);

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

    // Fusion、CanonicalForm、Loweringを含むSuggester
    let suggesters: Vec<Box<dyn crate::opt::graph::GraphSuggester>> = vec![
        Box::new(FusionSuggester::new()),
        Box::new(CanonicalFormSuggester::new()),
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

// ============================================================================
// 非連続View（Strided View）対応のテスト
// ============================================================================

#[test]
fn test_lower_elementwise_with_transposed_input() {
    use crate::opt::graph::BeamSearchGraphOptimizer;

    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![20, 10]);

    // aを転置 [10, 20] -> [20, 10]
    let a_t = a.view(a.view.clone().permute(vec![1, 0]));

    // 転置した入力と通常の入力を加算
    let c = a_t + b;
    graph.output("c", c);

    // 正規化してからLowering候補を生成
    let canonical_graph = canonicalize_graph(&graph);
    let suggestions = suggester.suggest(&canonical_graph);

    // 転置したViewを持つ入力もloweringできることを確認
    assert!(
        !suggestions.is_empty(),
        "Should generate suggestions for transposed view"
    );

    // 最終的にKernelノードに変換されることを確認
    let optimizer =
        BeamSearchGraphOptimizer::new(crate::opt::graph::CompositeSuggester::new(vec![
            Box::new(CanonicalFormSuggester::new()),
            Box::new(LoweringSuggester::new()),
        ]))
        .with_beam_width(4)
        .with_max_steps(20);

    let (optimized, _) = optimizer.optimize_with_history(graph);
    let output = optimized.outputs().get("c").unwrap();

    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Transposed elementwise should be lowered to Kernel node"
    );
}

#[test]
fn test_lower_reduce_with_transposed_input() {
    use crate::opt::graph::BeamSearchGraphOptimizer;

    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]); // [10, 20]

    // 転置して [20, 10] に
    let a_t = a.view(a.view.clone().permute(vec![1, 0]));

    // 軸0でreduce
    let b = a_t.reduce_sum(0); // [10]
    graph.output("b", b);

    // 正規化してからLowering候補を生成
    let canonical_graph = canonicalize_graph(&graph);
    let suggestions = suggester.suggest(&canonical_graph);

    // 転置したViewを持つ入力もloweringできることを確認
    assert!(
        !suggestions.is_empty(),
        "Should generate suggestions for transposed view reduce"
    );

    // 最終的にKernelノードに変換されることを確認
    let optimizer =
        BeamSearchGraphOptimizer::new(crate::opt::graph::CompositeSuggester::new(vec![
            Box::new(CanonicalFormSuggester::new()),
            Box::new(LoweringSuggester::new()),
        ]))
        .with_beam_width(4)
        .with_max_steps(20);

    let (optimized, _) = optimizer.optimize_with_history(graph);
    let output = optimized.outputs().get("b").unwrap();

    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Transposed reduce should be lowered to Kernel node"
    );
}

#[test]
fn test_lower_cast_with_transposed_input() {
    use crate::opt::graph::BeamSearchGraphOptimizer;

    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);

    // 転置してからキャスト
    let a_t = a.view(a.view.clone().permute(vec![1, 0]));
    let b = a_t.cast(DType::I32);
    graph.output("b", b);

    // Lowering候補を生成
    let suggestions = suggester.suggest(&graph);

    // 転置したViewを持つ入力もloweringできることを確認
    assert!(
        !suggestions.is_empty(),
        "Should generate suggestions for transposed view cast"
    );

    // 最終的にKernelノードに変換されることを確認
    let optimizer = BeamSearchGraphOptimizer::new(crate::opt::graph::CompositeSuggester::new(
        vec![Box::new(LoweringSuggester::new())],
    ))
    .with_beam_width(4)
    .with_max_steps(20);

    let (optimized, _) = optimizer.optimize_with_history(graph);
    let output = optimized.outputs().get("b").unwrap();

    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Transposed cast should be lowered to Kernel node"
    );
}

#[test]
fn test_lower_elementwise_with_broadcast_input() {
    use crate::graph::shape::Expr;
    use crate::opt::graph::BeamSearchGraphOptimizer;

    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 1]); // [10, 1]
    let b = graph.input("b", DType::F32, vec![10, 20]); // [10, 20]

    // aをブロードキャスト [10, 1] -> [10, 20]
    let a_broadcast = a.broadcast_to(vec![Expr::from(10), Expr::from(20)]);

    // ブロードキャストした入力と通常の入力を加算
    let c = a_broadcast + b;
    graph.output("c", c);

    // 正規化してからLowering候補を生成
    let canonical_graph = canonicalize_graph(&graph);
    let suggestions = suggester.suggest(&canonical_graph);

    // ブロードキャストしたViewを持つ入力もloweringできることを確認
    assert!(
        !suggestions.is_empty(),
        "Should generate suggestions for broadcast view"
    );

    // 最終的にKernelノードに変換されることを確認
    let optimizer =
        BeamSearchGraphOptimizer::new(crate::opt::graph::CompositeSuggester::new(vec![
            Box::new(CanonicalFormSuggester::new()),
            Box::new(LoweringSuggester::new()),
        ]))
        .with_beam_width(4)
        .with_max_steps(20);

    let (optimized, _) = optimizer.optimize_with_history(graph);
    let output = optimized.outputs().get("c").unwrap();

    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Broadcast elementwise should be lowered to Kernel node"
    );
}

// ============================================================================
// FusedElementwiseReduce統一テスト（axes=[]のケース）
// ============================================================================

#[test]
fn test_fused_elementwise_reduce_with_empty_axes() {
    // axes=[]のケース: Elementwiseとして処理される
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);

    // FusedElementwiseReduce with axes=[] は Elementwise と同等
    let expr = wildcard("0") + wildcard("1");
    let c = test_fused_elementwise_reduce(vec![a.clone(), b.clone()], expr, ReduceOp::Sum, vec![]);
    graph.output("c", c);

    let suggestions = suggester.suggest(&graph);

    // axes=[]でも正しくloweringされることを確認
    assert_eq!(
        suggestions.len(),
        1,
        "FusedElementwiseReduce with axes=[] should generate 1 candidate"
    );

    // Kernelノードに変換されることを確認
    let new_graph = &suggestions[0].graph;
    let outputs = new_graph.outputs();
    let output = outputs.get("c").unwrap();
    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "FusedElementwiseReduce with axes=[] should be lowered to Kernel node"
    );

    // 出力形状が入力と同じことを確認（縮約なし）
    let output_shape = output.view.shape();
    assert_eq!(output_shape.len(), 2, "Output should be 2D");
    assert_eq!(output_shape[0], 10.into(), "Output dim 0 should be 10");
    assert_eq!(output_shape[1], 20.into(), "Output dim 1 should be 20");
}

#[test]
fn test_elementwise_via_unified_lowering() {
    // Elementwiseが正規化→lowering経由でKernelに変換されることを確認
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![5, 10, 15]);
    let b = graph.input("b", DType::F32, vec![5, 10, 15]);
    let c = a * b; // Elementwise Mul
    graph.output("c", c);

    // 正規化してからLowering
    let canonical_graph = canonicalize_graph(&graph);
    let suggestions = suggester.suggest(&canonical_graph);
    assert_eq!(suggestions.len(), 1, "Should generate 1 candidate");

    // Kernelノードに変換されることを確認
    let new_graph = &suggestions[0].graph;
    let output = new_graph.outputs().get("c").unwrap();
    assert!(matches!(output.op, GraphOp::Kernel { .. }));

    // 出力形状が正しいことを確認
    let output_shape = output.view.shape();
    assert_eq!(output_shape.len(), 3);
    assert_eq!(output_shape[0], 5.into());
    assert_eq!(output_shape[1], 10.into());
    assert_eq!(output_shape[2], 15.into());
}

#[test]
fn test_reduce_via_unified_lowering() {
    // Reduceが正規化→lowering経由でKernelに変換されることを確認
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![5, 10, 15]);
    let b = a.reduce(ReduceOp::Prod, 1); // Reduce Prod over axis 1
    graph.output("b", b);

    // 正規化してからLowering
    let canonical_graph = canonicalize_graph(&graph);
    let suggestions = suggester.suggest(&canonical_graph);
    assert_eq!(suggestions.len(), 1, "Should generate 1 candidate");

    // Kernelノードに変換されることを確認
    let new_graph = &suggestions[0].graph;
    let output = new_graph.outputs().get("b").unwrap();
    assert!(matches!(output.op, GraphOp::Kernel { .. }));

    // 出力形状が正しいことを確認（axis 1が縮約された）
    let output_shape = output.view.shape();
    assert_eq!(output_shape.len(), 2);
    assert_eq!(output_shape[0], 5.into());
    assert_eq!(output_shape[1], 15.into());
}
