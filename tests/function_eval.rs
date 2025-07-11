use harp::node::{constant, exp, exp2, log2, pow, sqrt};
use harp::simplify::default_rewriter;
use rstest::rstest;

#[rstest]
#[case(exp(constant(1.0f32)), std::f32::consts::E)]
#[case(exp2(constant(2.0f32)), 4.0)]
#[case(log2(constant(4.0f32)), 2.0)]
#[case(sqrt(constant(9.0f32)), 3.0)]
#[ignore]
fn test_eval_unary_functions(#[case] graph: harp::node::Node, #[case] expected: f32) {
    let rewriter = default_rewriter();
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::op::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - expected).abs() < 1e-6);
}

#[rstest]
#[case(pow(constant(2.0f32), constant(3.0f32)), 8.0)]
fn test_eval_binary_functions(#[case] graph: harp::node::Node, #[case] expected: f32) {
    let rewriter = default_rewriter();
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::op::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - expected).abs() < 1e-6);
}
