use harp::node::constant;
use harp::pattern::Rewriter;

#[test]
fn test_rewriter_log_crate() {
    let rule = harp::rewrite_rule!(let x = capture("x"); x.clone() + constant(0.0f32) => x);
    let rewriter = Rewriter::new("test_rewriter", vec![rule]);
    let graph = constant(1.0f32) + constant(0.0f32);
    let _ = rewriter.rewrite(graph);
}
