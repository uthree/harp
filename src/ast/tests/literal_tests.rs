use super::super::*;
use crate::ast::helper::*;

#[test]
fn test_literal_from_f32() {
    let lit: Literal = 3.14f32.into();
    match lit {
        Literal::F32(v) => assert_eq!(v, 3.14),
        _ => panic!("Expected F32 literal"),
    }

    let lit = Literal::from(2.5f32);
    match lit {
        Literal::F32(v) => assert_eq!(v, 2.5),
        _ => panic!("Expected F32 literal"),
    }
}

#[test]
fn test_literal_from_isize() {
    let lit: Literal = 42isize.into();
    match lit {
        Literal::Isize(v) => assert_eq!(v, 42),
        _ => panic!("Expected Isize literal"),
    }

    let lit = Literal::from(-10isize);
    match lit {
        Literal::Isize(v) => assert_eq!(v, -10),
        _ => panic!("Expected Isize literal"),
    }
}

#[test]
fn test_literal_from_usize() {
    let lit: Literal = 100usize.into();
    match lit {
        Literal::Usize(v) => assert_eq!(v, 100),
        _ => panic!("Expected Usize literal"),
    }

    let lit = Literal::from(256usize);
    match lit {
        Literal::Usize(v) => assert_eq!(v, 256),
        _ => panic!("Expected Usize literal"),
    }
}

#[test]
fn test_literal_dtype() {
    let f32_lit = Literal::F32(3.14);
    assert_eq!(f32_lit.dtype(), DType::F32);

    let isize_lit = Literal::Isize(42);
    assert_eq!(isize_lit.dtype(), DType::Isize);

    let usize_lit = Literal::Usize(100);
    assert_eq!(usize_lit.dtype(), DType::Usize);
}
