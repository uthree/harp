use harp::node;

#[test]
fn test_add_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = a.clone() + b.clone();
    a += b;
    assert_eq!(a, expected);
}

#[test]
fn test_sub_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = a.clone() - b.clone();
    a -= b;
    assert_eq!(a, expected);
}

#[test]
fn test_mul_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = a.clone() * b.clone();
    a *= b;
    assert_eq!(a, expected);
}

#[test]
fn test_div_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = a.clone() / b.clone();
    a /= b;
    assert_eq!(a, expected);
}
