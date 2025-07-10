mod common;
use common::eval_rules;
use harp::node::{constant, cos, tan};

#[test]
fn test_eval_cos() {
    let rewriter = eval_rules();
    let graph = cos(constant(0.0f32));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::node::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_eval_tan() {
    let rewriter = eval_rules();
    let graph = tan(constant(std::f32::consts::PI / 4.0));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::node::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}
