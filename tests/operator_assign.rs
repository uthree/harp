use harp::node;

#[test]
fn test_add_assign_primitive() {
    let mut a = node::constant(1.0f32);
    let expected = a.clone() + 2.0f32;
    a += 2.0f32;
    assert_eq!(a, expected);
}

#[test]
fn test_sub_assign_primitive() {
    let mut a = node::constant(1.0f32);
    let expected = a.clone() - 2.0f32;
    a -= 2.0f32;
    assert_eq!(a, expected);
}

#[test]
fn test_mul_assign_primitive() {
    let mut a = node::constant(1.0f32);
    let expected = a.clone() * 2.0f32;
    a *= 2.0f32;
    assert_eq!(a, expected);
}

#[test]
fn test_div_assign_primitive() {
    let mut a = node::constant(1.0f32);
    let expected = a.clone() / 2.0f32;
    a /= 2.0f32;
    assert_eq!(a, expected);
}
