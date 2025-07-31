// tests/uop.rs

use harp::ast::{AstNode, DType, Op};

#[test]
fn test_uop_creation() {
    let uop = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    assert_eq!(uop.op, Op::Var("a".to_string()));
    assert!(uop.src.is_empty());
    assert_eq!(uop.dtype, DType::F32);
}

#[test]
fn test_from_literal() {
    let uop_f32: AstNode = 1.0f32.into();
    assert_eq!(uop_f32.dtype, DType::F32);
    if let Op::Const(c) = uop_f32.op {
        assert_eq!(c.dtype(), DType::F32);
    } else {
        panic!("Expected Op::Const");
    }

    let uop_i32: AstNode = 10i32.into();
    assert_eq!(uop_i32.dtype, DType::I32);
    if let Op::Const(c) = uop_i32.op {
        assert_eq!(c.dtype(), DType::I32);
    } else {
        panic!("Expected Op::Const");
    }
}

#[test]
fn test_unary_ops() {
    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);

    let neg_a = -a.clone();
    assert_eq!(neg_a.op, Op::Neg);
    assert_eq!(neg_a.src.len(), 1);
    assert_eq!(*neg_a.src[0], a);

    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let sqrt_a = a.clone().sqrt();
    assert_eq!(sqrt_a.op, Op::Sqrt);
    assert_eq!(sqrt_a.src.len(), 1);
    assert_eq!(*sqrt_a.src[0], a);

    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let sin_a = a.clone().sin();
    assert_eq!(sin_a.op, Op::Sin);
    assert_eq!(sin_a.src.len(), 1);
    assert_eq!(*sin_a.src[0], a);
}

#[test]
fn test_binary_ops() {
    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::F32);

    let add_ab = a.clone() + b.clone();
    assert_eq!(add_ab.op, Op::Add);
    assert_eq!(add_ab.src.len(), 2);
    assert_eq!(*add_ab.src[0], a);
    assert_eq!(*add_ab.src[1], b);

    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::F32);
    let sub_ab = a.clone() - b.clone();
    assert_eq!(sub_ab.op, Op::Add); // sub is implemented as a + (-b)
    assert_eq!(sub_ab.src.len(), 2);
    assert_eq!(*sub_ab.src[0], a);
    assert_eq!(sub_ab.src[1].op, Op::Neg);
    assert_eq!(*sub_ab.src[1].src[0], b);

    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::F32);
    let mul_ab = a.clone() * b.clone();
    assert_eq!(mul_ab.op, Op::Mul);
    assert_eq!(mul_ab.src.len(), 2);
    assert_eq!(*mul_ab.src[0], a);
    assert_eq!(*mul_ab.src[1], b);

    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::F32);
    let div_ab = a.clone() / b.clone();
    assert_eq!(div_ab.op, Op::Mul); // div is implemented as a * (1/b)
    assert_eq!(div_ab.src.len(), 2);
    assert_eq!(*div_ab.src[0], a);
    assert_eq!(div_ab.src[1].op, Op::Recip);
    assert_eq!(*div_ab.src[1].src[0], b);

    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::F32);
    let rem_ab = a.clone() % b.clone();
    assert_eq!(rem_ab.op, Op::Rem);
    assert_eq!(rem_ab.src.len(), 2);
    assert_eq!(*rem_ab.src[0], a);
    assert_eq!(*rem_ab.src[1], b);

    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::F32);
    let max_ab = a.clone().max(b.clone());
    assert_eq!(max_ab.op, Op::Max);
    assert_eq!(max_ab.src.len(), 2);
    assert_eq!(*max_ab.src[0], a);
    assert_eq!(*max_ab.src[1], b);
}

#[test]
#[should_panic(expected = "type mismatch")]
fn test_binary_op_type_mismatch() {
    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::I32);
    let _ = a + b; // This should panic
}

#[test]
fn test_complex_expression() {
    let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::F64);
    let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::F64);
    let c: AstNode = 2.0f64.into();

    // (a + b) * c
    let expr = (a.clone() + b.clone()) * c.clone();

    assert_eq!(expr.op, Op::Mul);
    assert_eq!(expr.src.len(), 2);
    assert_eq!(*expr.src[1], c);

    let add_expr = &expr.src[0];
    assert_eq!(add_expr.op, Op::Add);
    assert_eq!(add_expr.src.len(), 2);
    assert_eq!(*add_expr.src[0], a);
    assert_eq!(*add_expr.src[1], b);
}

#[test]
fn test_partial_eq_ignores_id() {
    let node1 = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);
    let node2 = AstNode::new(Op::Var("a".to_string()), vec![], DType::F32);

    // IDs should be different
    assert_ne!(node1.id, node2.id);
    // But the nodes should be considered equal
    assert_eq!(node1, node2);

    let node3 = AstNode::new(Op::Var("b".to_string()), vec![], DType::F32);
    assert_ne!(node1, node3);
}
