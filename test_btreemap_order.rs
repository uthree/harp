use harp::graph::{DType, Graph};
use harp::opt::graph::{BeamSearchGraphOptimizer, FusionSuggester, SimpleCostEstimator};

fn main() {
    // 2つのグラフを異なる出力順序で作成
    let mut graph1 = Graph::new();
    let a1 = graph1.input("x").with_dtype(DType::F32).with_shape(vec![10]).build();
    let b1 = graph1.input("y").with_dtype(DType::F32).with_shape(vec![10]).build();
    graph1.output("alpha", a1.clone() + b1.clone());
    graph1.output("beta", a1.clone() * b1.clone());
    graph1.output("gamma", a1.clone() - b1.clone());

    let mut graph2 = Graph::new();
    let a2 = graph2.input("x").with_dtype(DType::F32).with_shape(vec![10]).build();
    let b2 = graph2.input("y").with_dtype(DType::F32).with_shape(vec![10]).build();
    // 逆順で追加
    graph2.output("gamma", a2.clone() - b2.clone());
    graph2.output("beta", a2.clone() * b2.clone());
    graph2.output("alpha", a2.clone() + b2.clone());

    println!("=== 出力順序の確認 ===");
    println!("\nグラフ1の出力順序:");
    for (i, name) in graph1.outputs().keys().enumerate() {
        println!("  {}. {}", i, name);
    }

    println!("\nグラフ2の出力順序:");
    for (i, name) in graph2.outputs().keys().enumerate() {
        println!("  {}. {}", i, name);
    }

    let dot1 = graph1.to_dot();
    let dot2 = graph2.to_dot();

    println!("\nDOT文字列が同一: {}", dot1 == dot2);
    println!("DOT文字列の長さ: graph1={}, graph2={}", dot1.len(), dot2.len());

    // 最適化を実行して、候補が重複しないことを確認
    println!("\n=== 最適化の実行 ===");
    let suggester = FusionSuggester::new();
    let estimator = SimpleCostEstimator::new();
    let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
        .with_beam_width(10)
        .with_max_steps(5)
        .with_progress(false);

    let (result1, history1) = optimizer.optimize_with_history(graph1.clone());
    let (result2, history2) = optimizer.optimize_with_history(graph2.clone());

    println!("\nグラフ1の最適化ステップ数: {}", history1.snapshots().len());
    println!("グラフ2の最適化ステップ数: {}", history2.snapshots().len());

    // 最終結果のDOT文字列が同じか確認
    let final_dot1 = result1.to_dot();
    let final_dot2 = result2.to_dot();

    println!("\n最適化後のDOT文字列が同一: {}", final_dot1 == final_dot2);

    if final_dot1 == final_dot2 && dot1 == dot2 {
        println!("\n✓ 出力順序の問題が完全に修正されました！");
    } else {
        println!("\n✗ まだ問題が残っています");
        if dot1 != dot2 {
            println!("  - 初期グラフのDOT文字列が異なる");
        }
        if final_dot1 != final_dot2 {
            println!("  - 最適化後のグラフのDOT文字列が異なる");
        }
    }
}
