use harp::node::{self, Node};
use harp::pattern::Rewriter;
use harp::{capture, rewriter};

#[test]
fn test_rewriter_macro() {
    let rewriter: Rewriter = rewriter!("test", [
        (
            let x = capture("x")
            => x * Node::from(1.0f32)
            => |x| Some(x)
        )
    ]);
    let graph = Node::from(2.0f32) * Node::from(1.0f32);
    let rewritten = rewriter.rewrite(graph);
    assert_eq!(rewritten, Node::from(2.0f32));
}
