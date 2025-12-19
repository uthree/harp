use super::super::*;
use crate::graph::DType as GraphDType;

#[test]
fn test_create_signature_simple() {
    use crate::graph::shape::Expr;

    // 単純なグラフ: a (shape=[10, 20]) → output
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![10, 20]);
    graph.output("result", a);

    let signature = create_signature(&graph);

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

    // shape変数のデフォルト値を設定（必須）
    graph.set_shape_var_default("N", 100);
    graph.set_shape_var_default("M", 200);

    let a = graph.input("a", GraphDType::F32, vec![n.clone(), m.clone()]);
    graph.output("result", a);

    let signature = create_signature(&graph);

    // 入力が1つ
    assert_eq!(signature.inputs.len(), 1);
    assert_eq!(signature.inputs[0].name, "a");
    assert_eq!(signature.inputs[0].shape.len(), 2);
    assert_eq!(signature.inputs[0].shape[0], n);
    assert_eq!(signature.inputs[0].shape[1], m);

    // 動的なshape変数が2つ、デフォルト値付き
    assert_eq!(signature.shape_vars.len(), 2);
    assert_eq!(signature.shape_vars.get("M"), Some(&200));
    assert_eq!(signature.shape_vars.get("N"), Some(&100));
}

// Note: test_create_signature_multiple_inputs_outputs は複数出力が
// 現在サポートされていないため削除されました。
