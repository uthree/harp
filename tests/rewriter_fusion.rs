use harp::node::constant;
use harp::pattern::Rewriter;
use harp::rewrite_rule;

#[test]
#[allow(deprecated)]
fn test_rewriter_fusion() {
    // 1. Define two separate rewrite rules.
    let rule1 = rewrite_rule!(let x = capture("x"); x.clone() + constant(0.0f32) => x);
    let rewriter1 = Rewriter::new("identity_add", vec![rule1]);

    let rule2 = rewrite_rule!(let x = capture("x"); x.clone() * constant(1.0f32) => x);
    let rewriter2 = Rewriter::new("identity_mul", vec![rule2]);

    // 2. Fuse the rewriters.
    let fused_rewriter = rewriter1 + rewriter2;

    // 3. Define a graph that can be simplified by both rules.
    let graph = (constant(42.0f32) * constant(1.0f32)) + constant(0.0f32);

    // 4. Apply the fused rewriter.
    let rewritten_graph = fused_rewriter.rewrite(graph);

    // 5. Assert that the graph is fully simplified.
    assert_eq!(rewritten_graph, constant(42.0f32));
}
