use super::super::*;
use crate::ast::helper::*;

#[test]
fn test_children_const() {
    let node = AstNode::Const(3.14f32.into());
    let children = node.children();
    assert_eq!(children.len(), 0);
}

#[test]
fn test_children_binary_ops() {
    let a = AstNode::Const(1.0f32.into());
    let b = AstNode::Const(2.0f32.into());
    let node = a + b;
    let children = node.children();
    assert_eq!(children.len(), 2);

    let node = AstNode::Const(3isize.into()) * AstNode::Const(4isize.into());
    let children = node.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_children_unary_ops() {
    let node = sqrt(AstNode::Const(4.0f32.into()));
    let children = node.children();
    assert_eq!(children.len(), 1);

    let node = sin(AstNode::Const(1.0f32.into()));
    let children = node.children();
    assert_eq!(children.len(), 1);

    let node = recip(AstNode::Const(2.0f32.into()));
    let children = node.children();
    assert_eq!(children.len(), 1);
}

#[test]
fn test_children_cast() {
    let node = cast(AstNode::Const(3.14f32.into()), DType::Isize);
    let children = node.children();
    assert_eq!(children.len(), 1);
}

#[test]
fn test_children_composite() {
    // (a + b) * c
    let a = AstNode::Const(1.0f32.into());
    let b = AstNode::Const(2.0f32.into());
    let c = AstNode::Const(3.0f32.into());
    let product = (a + b) * c;

    let children = product.children();
    assert_eq!(children.len(), 2);

    // The first child should be the sum node
    let sum_children = children[0].children();
    assert_eq!(sum_children.len(), 2);
}
