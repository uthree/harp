use harp::node::{self, Node};

#[test]
fn test_operator_overloads_with_primitives() {
    let a = node::constant(1.0f32);
    let b = node::constant(2.0f32);

    // Test expression: a - b / 3.0
    let result_node = a.clone() - b.clone() / 3.0f32;

    // Manually construct the expected graph structure.
    let expected_node = a - (b / Node::from(3.0f32));

    assert_eq!(result_node, expected_node);
}
