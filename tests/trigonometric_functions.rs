use harp::node::{constant, cos, tan};
use harp::simplify::default_rewriter;

#[test]
fn test_eval_cos() {
    let rewriter = default_rewriter();
    let graph = cos(constant(0.0f32));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph.op().as_any().downcast_ref::<harp::op::Const>().unwrap().0.as_any().downcast_ref::<f32>().unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_eval_tan() {
    let rewriter = default_rewriter();
    let graph = tan(constant(std::f32::consts::PI / 4.0));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph.op().as_any().downcast_ref::<harp::op::Const>().unwrap().0.as_any().downcast_ref::<f32>().unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}
