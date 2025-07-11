use harp::dot::ToDot;
use harp::node;
use harp::tensor::Tensor;

#[test]
fn test_node_to_dot() {
    let a = node::variable("a");
    let b = node::constant(2.0f32);
    let graph = a + b;

    let dot_string = graph.to_dot();

    // Check for key elements in the DOT string
    assert!(dot_string.contains("digraph G"));
    assert!(dot_string.contains("rankdir=TB"));
    // Check for node definitions (content and shape)
    assert!(dot_string.contains("[label=\"Variable\", shape=\"box\"]"));
    assert!(dot_string.contains("[label=\"Const\n(2.0)\n<f32>\", shape=\"ellipse\"]"));
    assert!(dot_string.contains("[label=\"OpAdd\", shape=\"box\"]"));
    // Check that there are two edges pointing to some node.
    // This is less brittle than checking for a specific node ID.
    assert_eq!(dot_string.matches("-> node").count(), 2);
}

#[test]
fn test_tensor_to_dot() {
    let shape = vec![2, 3];
    let a = Tensor::new_load(shape.clone());
    let b = Tensor::new_load(shape.clone());
    let c = a + b;

    let dot_string = c.to_dot();

    // Check for key elements
    assert!(dot_string.contains("digraph G"));
    // Check for tensor node definitions (op name and shape)
    assert!(
        dot_string
            .matches("[label=\"Load\nshape: [2, 3]\", shape=\"box\"]")
            .count()
            == 2
    );
    assert!(dot_string.contains("[label=\"OpAdd\nshape: [2, 3]\", shape=\"box\"]"));
    // Check that there are two edges pointing to some node.
    assert_eq!(dot_string.matches("-> node").count(), 2);
}
