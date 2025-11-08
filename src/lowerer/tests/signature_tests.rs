use super::super::*;
use crate::graph::DType as GraphDType;

#[test]
fn test_create_signature_simple() {
    use crate::graph::shape::Expr;

    // 単純なグラフ: a (shape=[10, 20]) → output
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10, 20])
        .build();
    graph.output("result", a);

    let signature = Lowerer::create_signature(&graph);

    // 入力が1つ
    assert_eq!(signature.inputs.len(), 1);
    assert_eq!(signature.inputs[0].name, "a");
    assert_eq!(signature.inputs[0].shape.len(), 2);
    assert_eq!(signature.inputs[0].shape[0], Expr::from(10));
    assert_eq!(signature.inputs[0].shape[1], Expr::from(20));

    // 出力が1つ
    assert_eq!(signature.outputs.len(), 1);
    assert_eq!(signature.outputs[0].name, "result");
    assert_eq!(signature.outputs[0].shape.len(), 2);

    // 動的なshape変数はなし
    assert_eq!(signature.shape_vars.len(), 0);
}

#[test]
fn test_create_signature_with_dynamic_shape() {
    use crate::graph::shape::Expr;

    // 動的なshapeを持つグラフ: a (shape=[N, M])
    let mut graph = Graph::new();
    let n = Expr::Var("N".to_string());
    let m = Expr::Var("M".to_string());
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![n.clone(), m.clone()])
        .build();
    graph.output("result", a);

    let signature = Lowerer::create_signature(&graph);

    // 入力が1つ
    assert_eq!(signature.inputs.len(), 1);
    assert_eq!(signature.inputs[0].name, "a");
    assert_eq!(signature.inputs[0].shape.len(), 2);
    assert_eq!(signature.inputs[0].shape[0], n);
    assert_eq!(signature.inputs[0].shape[1], m);

    // 動的なshape変数が2つ（ソートされている）
    assert_eq!(signature.shape_vars.len(), 2);
    assert!(signature.shape_vars.contains(&"M".to_string()));
    assert!(signature.shape_vars.contains(&"N".to_string()));
}

#[test]
fn test_create_signature_multiple_inputs_outputs() {
    use crate::graph::shape::Expr;

    // 複数入出力のグラフ
    let mut graph = Graph::new();
    let a = graph
        .input("input_a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let b = graph
        .input("input_b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();

    let sum = a.clone() + b.clone();
    let prod = a * b;

    graph.output("sum", sum);
    graph.output("product", prod);

    let signature = Lowerer::create_signature(&graph);

    // 入力が2つ
    assert_eq!(signature.inputs.len(), 2);
    assert!(signature.inputs.iter().any(|i| i.name == "input_a"));
    assert!(signature.inputs.iter().any(|i| i.name == "input_b"));

    // 出力が2つ
    assert_eq!(signature.outputs.len(), 2);
    assert!(signature.outputs.iter().any(|o| o.name == "sum"));
    assert!(signature.outputs.iter().any(|o| o.name == "product"));

    // 全て同じshape [10]
    for input in &signature.inputs {
        assert_eq!(input.shape.len(), 1);
        assert_eq!(input.shape[0], Expr::from(10));
    }
}
