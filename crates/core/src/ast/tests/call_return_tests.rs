#![allow(clippy::approx_constant)]

use super::*;
use crate::ast::helper::{const_f32, const_int, var};

// Call tests
#[test]
fn test_call_children() {
    let call = AstNode::Call {
        name: "add".to_string(),
        args: vec![const_int(1), const_int(2)],
    };

    let children = call.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_call_map_children() {
    let call = AstNode::Call {
        name: "mul".to_string(),
        args: vec![const_int(3), const_int(4)],
    };

    let mapped = call.map_children(&|node| match node {
        AstNode::Const(Literal::Int(n)) => const_int(n * 2),
        _ => node.clone(),
    });

    if let AstNode::Call { name, args } = mapped {
        assert_eq!(name, "mul");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], const_int(6));
        assert_eq!(args[1], const_int(8));
    } else {
        panic!("Expected Call node");
    }
}

#[test]
fn test_call_check_scope() {
    let mut scope = Scope::new();
    scope
        .declare("x".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();
    scope
        .declare("y".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let call = AstNode::Call {
        name: "add".to_string(),
        args: vec![var("x"), var("y")],
    };

    assert!(call.check_scope(&scope).is_ok());
}

// Return tests
#[test]
fn test_return_children() {
    let ret = AstNode::Return {
        value: Box::new(const_int(42)),
    };

    let children = ret.children();
    assert_eq!(children.len(), 1);
}

#[test]
fn test_return_infer_type() {
    let ret = AstNode::Return {
        value: Box::new(const_f32(3.14)),
    };

    assert_eq!(ret.infer_type(), DType::F32);
}

#[test]
fn test_return_check_scope() {
    let mut scope = Scope::new();
    scope
        .declare("result".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    let ret = AstNode::Return {
        value: Box::new(var("result")),
    };

    assert!(ret.check_scope(&scope).is_ok());
}
