#![allow(clippy::approx_constant)]

use super::*;

// Call tests
#[test]
fn test_call_children() {
    let call = AstNode::Call {
        name: "add".to_string(),
        args: vec![AstNode::Const(1isize.into()), AstNode::Const(2isize.into())],
    };

    let children = call.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_call_map_children() {
    let call = AstNode::Call {
        name: "mul".to_string(),
        args: vec![AstNode::Const(3isize.into()), AstNode::Const(4isize.into())],
    };

    let mapped = call.map_children(&|node| match node {
        AstNode::Const(Literal::Int(n)) => AstNode::Const(Literal::Int(n * 2)),
        _ => node.clone(),
    });

    if let AstNode::Call { name, args } = mapped {
        assert_eq!(name, "mul");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], AstNode::Const(Literal::Int(6)));
        assert_eq!(args[1], AstNode::Const(Literal::Int(8)));
    } else {
        panic!("Expected Call node");
    }
}

#[test]
fn test_call_check_scope() {
    let mut scope = Scope::new();
    scope
        .declare("x".to_string(), DType::F32, Mutability::Immutable, None)
        .unwrap();
    scope
        .declare("y".to_string(), DType::F32, Mutability::Immutable, None)
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
        value: Box::new(AstNode::Const(42isize.into())),
    };

    let children = ret.children();
    assert_eq!(children.len(), 1);
}

#[test]
fn test_return_infer_type() {
    let ret = AstNode::Return {
        value: Box::new(AstNode::Const(3.14f32.into())),
    };

    assert_eq!(ret.infer_type(), DType::F32);
}

#[test]
fn test_return_check_scope() {
    let mut scope = Scope::new();
    scope
        .declare(
            "result".to_string(),
            DType::Isize,
            Mutability::Immutable,
            None,
        )
        .unwrap();

    let ret = AstNode::Return {
        value: Box::new(var("result")),
    };

    assert!(ret.check_scope(&scope).is_ok());
}
