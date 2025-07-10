use harp::node::{constant, Node};
use harp::simplify::default_rewriter;

#[test]
fn test_identity_add() {
    let rewriter = default_rewriter();
    let a = || Node::from(constant(2.0f32));
    let graph = a() + constant(0.0f32);
    let rewritten = rewriter.rewrite(graph);
    assert_eq!(rewritten, a());
}

#[test]
fn test_identity_mul() {
    let rewriter = default_rewriter();
    let a = || Node::from(constant(2.0f32));
    let graph = a() * constant(1.0f32);
    let rewritten = rewriter.rewrite(graph);
    assert_eq!(rewritten, a());
}

#[test]
fn test_annihilator_mul() {
    let rewriter = default_rewriter();
    let a = || Node::from(constant(2.0f32));
    let graph = a() * constant(0.0f32);
    let rewritten = rewriter.rewrite(graph);
    assert_eq!(rewritten, constant(0.0f32));
}

#[test]
fn test_complex_simplification() {
    let rewriter = default_rewriter();
    let a = || Node::from(constant(2.0f32));
    // (a * 1.0) + (a * 0.0) -> a + 0.0 -> a
    let graph = (a() * constant(1.0f32)) + (a() * constant(0.0f32));
    let rewritten = rewriter.rewrite(graph);
    assert_eq!(rewritten, a());
}
