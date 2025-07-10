use harp::node;
use std::ops::Neg;

#[test]
fn test_add_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = node::add(a.clone(), b.clone());
    a += b;
    assert_eq!(a, expected);
}

#[test]
fn test_sub_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = node::add(a.clone(), b.clone().neg());
    a -= b;
    assert_eq!(a, expected);
}

#[test]
fn test_mul_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = node::mul(a.clone(), b.clone());
    a *= b;
    assert_eq!(a, expected);
}

#[test]
fn test_div_assign() {
    let mut a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let expected = node::mul(a.clone(), node::recip(b.clone()));
    a /= b;
    assert_eq!(a, expected);
}
