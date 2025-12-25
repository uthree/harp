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

// Note: Dynamic shape tests were removed as Expr::Var was removed
// in favor of static shapes for simplicity.
