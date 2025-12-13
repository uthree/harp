//! LoweringSuggesterのテスト

use super::*;
use crate::ast::{DType as AstDType, Scope, helper::*};
use crate::graph::DType;

#[test]
fn test_lower_elementwise_add() {
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = a + b;
    graph.output("c", c);

    let suggestions = suggester.suggest(&graph);

    // Elementwise Add (2D) に対して複数の並列化戦略が生成される:
    // - Sequential, FlatParallel, MultiDimParallel{1}, MultiDimParallel{2}
    assert!(
        suggestions.len() >= 1,
        "At least one candidate should be generated"
    );

    // 全ての候補のグラフでKernelノードが使われていることを確認
    for (i, new_graph) in suggestions.iter().enumerate() {
        let outputs = new_graph.outputs();
        let output = outputs.get("c").unwrap();
        assert!(
            matches!(output.op, GraphOp::Kernel { .. }),
            "Candidate {} should use Kernel node",
            i
        );
    }
}

#[test]
fn test_lower_reduce_sum() {
    let suggester = LoweringSuggester::new();

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = a.reduce_sum(1);
    graph.output("b", b);

    let suggestions = suggester.suggest(&graph);

    // Reduce Sum に対して複数の並列化戦略が生成される
    assert!(
        suggestions.len() >= 1,
        "At least one candidate should be generated"
    );

    // 全ての候補のグラフでKernelノードが使われていることを確認
    for (i, new_graph) in suggestions.iter().enumerate() {
        let outputs = new_graph.outputs();
        let output = outputs.get("b").unwrap();
        assert!(
            matches!(output.op, GraphOp::Kernel { .. }),
            "Candidate {} should use Kernel node",
            i
        );
    }
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

    let optimizer = BeamSearchGraphOptimizer::new(composite, SimpleCostEstimator::new())
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
    use crate::opt::graph::{
        BeamSearchGraphOptimizer, CompositeSuggester, FusionSuggester, SimpleCostEstimator,
    };

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

    let optimizer = BeamSearchGraphOptimizer::new(composite, SimpleCostEstimator::new())
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

// Note: 複数出力のテストは現在サポートされていないため削除されました。
// 詳細は spec/TODO.md を参照してください。

#[test]
fn test_sequential_only_mode() {
    let suggester = LoweringSuggester::sequential_only();
    assert!(suggester.is_sequential_only());

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = a + b;
    graph.output("c", c);

    let suggestions = suggester.suggest(&graph);

    // Sequential専用モードでは1つの候補のみ生成される
    assert_eq!(
        suggestions.len(),
        1,
        "Sequential-only mode should generate exactly 1 candidate, got {}",
        suggestions.len()
    );

    // 候補のグラフでKernelノードが使われていることを確認
    let new_graph = &suggestions[0];
    let outputs = new_graph.outputs();
    let output = outputs.get("c").unwrap();
    assert!(
        matches!(output.op, GraphOp::Kernel { .. }),
        "Candidate should use Kernel node"
    );
}

#[test]
fn test_sequential_only_vs_normal() {
    let normal = LoweringSuggester::new();
    let sequential = LoweringSuggester::sequential_only();

    assert!(!normal.is_sequential_only());
    assert!(sequential.is_sequential_only());

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = a + b;
    graph.output("c", c);

    let normal_suggestions = normal.suggest(&graph);
    let sequential_suggestions = sequential.suggest(&graph);

    // 通常モードでは複数の候補が生成される
    assert!(
        normal_suggestions.len() > 1,
        "Normal mode should generate multiple candidates, got {}",
        normal_suggestions.len()
    );

    // Sequential専用モードでは1つの候補のみ
    assert_eq!(
        sequential_suggestions.len(),
        1,
        "Sequential-only mode should generate exactly 1 candidate"
    );
}
