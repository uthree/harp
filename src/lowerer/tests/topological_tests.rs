use super::super::*;
use crate::ast::renderer::render_ast_with;
use crate::backend::c::CRenderer;
use crate::graph::DType as GraphDType;

#[test]
fn test_topological_sort_simple() {
    // a + b のグラフ
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![10]);
    let b = graph.input("b", GraphDType::F32, vec![10]);
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
    let a = graph.input("a", GraphDType::F32, vec![10]);
    let b = graph.input("b", GraphDType::F32, vec![10]);
    let c = graph.input("c", GraphDType::F32, vec![10]);
    let d = graph.input("d", GraphDType::F32, vec![10]);

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

#[test]
fn test_multiple_kernels_variable_naming() {
    // グラフ最適化により連続した演算は1つのカーネルに融合される
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![10]);
    let b = graph.input("b", GraphDType::F32, vec![10]);

    // ((a + b) * a) + b - グラフ最適化により1つのCustomノードに融合
    let sum1 = a.clone() + b.clone();
    let sum2 = sum1.clone() * a.clone();
    let result = sum2 + b;
    graph.output("result", result);

    // Lower to AST
    let program = lower(graph);

    // Render to code
    let renderer = CRenderer::new();
    let code = render_ast_with(&program, &renderer);

    // デバッグ出力
    println!("Generated code:\n{}", code);

    // 中間変数（alu0 など）は生成されない
    assert!(
        !code.contains("alu0"),
        "alu0 should not be defined (intermediate variables eliminated)"
    );

    // 出力への直接代入が行われていることを確認
    // グラフ最適化後は output[ridx0] = ... の形式
    assert!(
        code.contains("output["),
        "output should be directly assigned"
    );

    // グラフ最適化により1つのカーネルに融合される
    // カーネル名は "kernel_0" または LoweringSuggester による "E_10" などの形式
    assert!(
        code.contains("kernel_0") || code.contains("E_10") || code.contains("E_"),
        "A kernel function should be defined"
    );

    // 演算が含まれていることを確認（入力バッファへのアクセス）
    assert!(
        code.contains("input0[") && code.contains("input1["),
        "inputs should be directly loaded"
    );
}
