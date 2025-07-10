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
fn test_expression_with_from() {
    let a = node::constant(1.0f32);
    let b: Node = 2.0f32.into();
    let c = a + b;

    let expected = node::constant(1.0f32) + node::constant(2.0f32);
    assert_eq!(c, expected);
}
