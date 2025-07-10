use harp::node::{self};

#[test]
fn test_rem_operator() {
    let a = node::constant(10.0f32);
    let b = node::constant(3.0f32);

    // Test Node % Node
    let result1 = a.clone() % b.clone();
    assert_eq!(result1.op().name(), "OpRem");
    assert_eq!(result1.src().len(), 2);
    assert_eq!(result1.src()[0], a);
    assert_eq!(result1.src()[1], b);

    // Test Node % primitive
    let result2 = a.clone() % 3.0f32;
    assert_eq!(result2.op().name(), "OpRem");
    assert_eq!(result2.src().len(), 2);
    assert_eq!(result2.src()[0], a);
    assert_eq!(result2.src()[1], node::constant(3.0f32));
}

#[test]
fn test_rem_assign_operator() {
    let mut a = node::constant(10.0f32);
    let b = node::constant(3.0f32);

    let expected = a.clone() % b.clone();
    a %= b;
    assert_eq!(a, expected);
}

#[test]
fn test_rem_assign_primitive() {
    let mut a = node::constant(10.0f32);
    let expected = a.clone() % 3.0f32;
    a %= 3.0f32;
    assert_eq!(a, expected);
}
