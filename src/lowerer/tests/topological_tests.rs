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
    // 複数のカーネル関数が生成される場合、各カーネル内の変数名が独立していることを確認
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![10]);
    let b = graph.input("b", GraphDType::F32, vec![10]);

    // 3つの演算（3つのカーネル関数が生成される）
    let sum1 = a.clone() + b.clone();
    let sum2 = sum1.clone() * a.clone();
    let result = sum2 + b;
    graph.output("result", result);

    // Lower to AST
    let program = lower(graph);

    // Render to code
    let renderer = CRenderer::new();
    let code = render_ast_with(&program, &renderer);

    // 最適化により中間変数を排除し、直接演算結果をストアする形式になった
    // kernel_0: output[...] = (input0[...] + input1[...]);
    // kernel_1: output[...] = (input0[...] * input1[...]);
    // kernel_2: output[...] = (input0[...] + input1[...]);

    // デバッグ出力
    println!("Generated code:\n{}", code);

    // 各カーネル関数が直接ストアを行っていることを確認
    // 中間変数（alu0 など）は生成されない
    assert!(
        !code.contains("alu0"),
        "alu0 should not be defined (intermediate variables eliminated)"
    );

    // 代わりに、直接演算結果をストアしていることを確認
    // output[...] = (...); の形式
    assert!(
        code.contains("output[("),
        "output should be directly assigned"
    );

    // 各カーネルが正しく生成されていることを確認
    assert!(code.contains("kernel_0"), "kernel_0 should be defined");
    assert!(code.contains("kernel_1"), "kernel_1 should be defined");
    assert!(code.contains("kernel_2"), "kernel_2 should be defined");

    // 演算が含まれていることを確認
    assert!(
        code.contains("input0[") && code.contains("input1["),
        "inputs should be directly loaded"
    );
}
