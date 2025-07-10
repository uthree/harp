use harp::node::{constant, exp2, log2, sin, sqrt};
use harp::simplify::default_rewriter;

#[test]
fn test_eval_sin() {
    let rewriter = default_rewriter();
    let graph = sin(constant(std::f32::consts::PI / 2.0));
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
fn test_eval_exp2() {
    let rewriter = default_rewriter();
    let graph = exp2(constant(2.0f32));
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
    assert!((result - 4.0).abs() < 1e-6);
}

#[test]
fn test_eval_log2() {
    let rewriter = default_rewriter();
    let graph = log2(constant(8.0f32));
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
    assert!((result - 3.0).abs() < 1e-6);
}

#[test]
fn test_eval_sqrt() {
    let rewriter = default_rewriter();
    let graph = sqrt(constant(16.0f32));
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
    assert!((result - 4.0).abs() < 1e-6);
}
