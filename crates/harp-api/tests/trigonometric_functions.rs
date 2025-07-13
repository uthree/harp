use harp_api::prelude::*;
use harp_ir::node::{constant, cos, sin, tan};
use harp_ir::simplify::default_rewriter;
use rstest::rstest;

#[rstest]
#[case(cos(constant(0.0f32)), 1.0)]
#[case(tan(constant(std::f32::consts::PI / 4.0)), 1.0)]
#[case(sin(constant(std::f32::consts::PI / 2.0)), 1.0)]
fn test_eval_trigonometric_functions(#[case] graph: harp_ir::node::Node, #[case] expected: f32) {
    let rewriter = default_rewriter();
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp_ir::op::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - expected).abs() < 1e-6);
}
