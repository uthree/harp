use harp::node::{self, Node};
use harp::rewriter;

#[test]
fn test_rewriter_fusion() {
    // Rewriter 1: Simplifies "x + 0"
    let rewriter1 = rewriter!([
        (
            let x = capture("x")
            => x + Node::from(0.0f32)
            => |x| Some(x)
        )
    ]);

    // Rewriter 2: Simplifies "y * 1"
    let rewriter2 = rewriter!([
        (
            let y = capture("y")
            => y * Node::from(1.0f32)
            => |y| Some(y)
        )
    ]);

    // Fuse the two rewriters using the `+` operator
    let fused_rewriter = rewriter1 + rewriter2;

    // Build a graph that can be simplified by both rules: (a + 0) * 1
    let a = node::constant(10.0f32);
    let graph = (a.clone() + 0.0f32) * 1.0f32;

    // Apply the fused rewriter
    let rewritten_graph = fused_rewriter.rewrite(graph);

    // The result should be simplified down to "a"
    assert_eq!(rewritten_graph, a);
}
