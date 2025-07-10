use harp::node::{self, Node};

#[test]
fn test_from_f32() {
    let node: Node = 42.0f32.into();
    let expected = node::constant(42.0f32);
    assert_eq!(node, expected);
}

#[test]
fn test_from_i32() {
    let node: Node = 42.into();
    let expected = node::constant(42);
    assert_eq!(node, expected);
}

#[test]
fn test_expression_with_primitive() {
    let a = node::constant(1.0f32);
    let c = a + 2.0f32;

    let expected = node::constant(1.0f32) + node::constant(2.0f32);
    assert_eq!(c, expected);
}
