use super::super::*;
use crate::ast::helper::*;

#[test]
fn test_scope_declare() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
        )
        .unwrap();

    assert!(scope.get("x").is_some());
    assert_eq!(scope.get("x").unwrap().dtype, DType::F32);
}

#[test]
fn test_scope_duplicate_declare() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
        )
        .unwrap();

    let result = scope.declare(
        "x".to_string(),
        DType::Isize,
        Mutability::Mutable,
    );

    assert!(result.is_err());
}

#[test]
fn test_scope_check_read() {
    let mut scope = Scope::new();

    scope
        .declare(
            "input".to_string(),
            DType::F32.to_ptr(),
            Mutability::Immutable,
        )
        .unwrap();

    assert!(scope.check_read("input").is_ok());
    assert!(scope.check_read("undefined").is_err());
}

#[test]
fn test_scope_check_write_immutable() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
        )
        .unwrap();

    let result = scope.check_write("x", &DType::F32);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("immutable"));
}

#[test]
fn test_scope_check_write_mutable() {
    let mut scope = Scope::new();

    scope
        .declare(
            "output".to_string(),
            DType::F32,
            Mutability::Mutable,
        )
        .unwrap();

    assert!(scope.check_write("output", &DType::F32).is_ok());
}

#[test]
fn test_scope_check_write_type_mismatch() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Mutable,
        )
        .unwrap();

    let result = scope.check_write("x", &DType::Isize);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Type mismatch"));
}

#[test]
fn test_scope_parent_lookup() {
    let mut parent = Scope::new();
    parent
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
        )
        .unwrap();

    let child = Scope::with_parent(parent);

    // 親スコープの変数にアクセスできる
    assert!(child.check_read("x").is_ok());
}

#[test]
fn test_check_scope_var() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
        )
        .unwrap();

    let var_node = AstNode::Var("x".to_string());
    assert!(var_node.check_scope(&scope).is_ok());

    let undefined_var = AstNode::Var("undefined".to_string());
    assert!(undefined_var.check_scope(&scope).is_err());
}

#[test]
fn test_check_scope_assign() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Mutable,
        )
        .unwrap();

    let assign_node = AstNode::Assign {
        var: "x".to_string(),
        value: Box::new(AstNode::Const(3.14f32.into())),
    };

    assert!(assign_node.check_scope(&scope).is_ok());
}

#[test]
fn test_check_scope_assign_immutable() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
        )
        .unwrap();

    let assign_node = AstNode::Assign {
        var: "x".to_string(),
        value: Box::new(AstNode::Const(3.14f32.into())),
    };

    let result = assign_node.check_scope(&scope);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("immutable"));
}

#[test]
fn test_check_scope_complex_expression() {
    let mut scope = Scope::new();

    scope
        .declare(
            "input".to_string(),
            DType::F32.to_ptr(),
            Mutability::Immutable,
        )
        .unwrap();

    scope
        .declare(
            "output".to_string(),
            DType::F32.to_ptr(),
            Mutability::Mutable,
        )
        .unwrap();

    scope
        .declare(
            "i".to_string(),
            DType::Usize,
            Mutability::Immutable,
        )
        .unwrap();

    // output[i] = input[i] * 2.0
    let expr = AstNode::Store {
        ptr: Box::new(AstNode::Var("output".to_string())),
        offset: Box::new(AstNode::Var("i".to_string())),
        value: Box::new(AstNode::Mul(
            Box::new(AstNode::Load {
                ptr: Box::new(AstNode::Var("input".to_string())),
                offset: Box::new(AstNode::Var("i".to_string())),
                count: 1,
                dtype: DType::F32,
            }),
            Box::new(AstNode::Const(2.0f32.into())),
        )),
    };

    assert!(expr.check_scope(&scope).is_ok());
}
