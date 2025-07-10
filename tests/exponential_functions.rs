mod common;
use common::eval_rules;
use harp::node::{constant, exp, pow};

#[test]
fn test_eval_exp() {
    let rewriter = eval_rules();
    let graph = exp(constant(1.0f32));
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
    assert!((result - std::f32::consts::E).abs() < 1e-6);
}

#[test]
fn test_eval_pow() {
    let rewriter = eval_rules();
    let graph = pow(constant(2.0f32), constant(3.0f32));
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
    assert!((result - 8.0).abs() < 1e-6);
}
