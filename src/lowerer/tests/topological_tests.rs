use super::super::*;
use crate::graph::DType as GraphDType;

#[test]
fn test_topological_sort_simple() {
    // a + b のグラフ
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let result = a + b;
    graph.output("result", result);

    let order = Lowerer::topological_sort(&graph);

    // 2世代に分かれる：
    // Generation 0: result (+)
    // Generation 1: a, b (並列実行可能な入力ノード)
    assert_eq!(order.len(), 2);
    assert_eq!(order[0].len(), 1); // result
    assert_eq!(order[1].len(), 2); // a, b
}

#[test]
fn test_topological_sort_complex() {
    // (a + b) * (c + d) のグラフ
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let c = graph
        .input("c")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let d = graph
        .input("d")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();

    let sum1 = a + b;
    let sum2 = c + d;
    let result = sum1 * sum2;
    graph.output("result", result);

    let order = Lowerer::topological_sort(&graph);

    // 世代構造を確認
    // Generation 0: result (*)
    // Generation 1: sum1 (+), sum2 (+) - 並列実行可能
    // Generation 2: a, b, c, d - 並列実行可能（入力ノード）
    assert_eq!(order.len(), 3);
    assert_eq!(order[0].len(), 1); // result
    assert_eq!(order[1].len(), 2); // sum1, sum2
    assert_eq!(order[2].len(), 4); // a, b, c, d
}
