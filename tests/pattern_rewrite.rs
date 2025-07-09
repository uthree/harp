use harp::node::{self, Capture, Node, NodeRef, Recip};
use harp::pattern::{RewriteRule, Rewriter};
use std::sync::Arc;

/// Helper function to create a capture node.
fn capture(name: &str) -> NodeRef {
    NodeRef::from(Arc::new(Node {
        op: Box::new(Capture(name.to_string())),
        src: vec![],
    }))
}

#[test]
fn test_double_recip_rewrite_with_node_pattern() {
    // 1. Define the graph to be rewritten: recip(recip(a))
    let a = node::constant(1.0f32);
    let recip_a = NodeRef::from(Arc::new(Node {
        op: Box::new(Recip),
        src: vec![a.clone()],
    }));
    let graph = NodeRef::from(Arc::new(Node {
        op: Box::new(Recip),
        src: vec![recip_a],
    }));

    // 2. Define the rewrite rule using Node patterns: recip(recip(x)) => x
    let x = capture("x");
    let searcher = NodeRef::from(Arc::new(Node {
        op: Box::new(Recip),
        src: vec![NodeRef::from(Arc::new(Node {
            op: Box::new(Recip),
            src: vec![x.clone()],
        }))],
    }));
    let rewriter = x;
    let rule = RewriteRule::new(searcher, rewriter);

    // 3. Apply the rule
    let rewriter = Rewriter::new(vec![rule]);
    let rewritten_graph = rewriter.rewrite(graph);

    // 4. Assert that the rewritten graph is `a`
    assert_eq!(*rewritten_graph, *a);
}
