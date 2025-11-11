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
            AccessRegion::ThreadLocal,
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
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    let result = scope.declare(
        "x".to_string(),
        DType::Isize,
        Mutability::Mutable,
        AccessRegion::ThreadLocal,
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
            AccessRegion::Shared,
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
            AccessRegion::ThreadLocal,
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
            AccessRegion::ThreadLocal,
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
            AccessRegion::ThreadLocal,
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
            AccessRegion::Shared,
        )
        .unwrap();

    let child = Scope::with_parent(parent);

    // 親スコープの変数にアクセスできる
    assert!(child.check_read("x").is_ok());
}

#[test]
fn test_scope_can_access_parallel_immutable() {
    let mut scope = Scope::new();

    scope
        .declare(
            "input1".to_string(),
            DType::F32.to_ptr(),
            Mutability::Immutable,
            AccessRegion::Shared,
        )
        .unwrap();

    scope
        .declare(
            "input2".to_string(),
            DType::F32.to_ptr(),
            Mutability::Immutable,
            AccessRegion::Shared,
        )
        .unwrap();

    // 両方immutableなので並列OK
    assert!(scope.can_access_parallel("input1", "input2"));
}

#[test]
fn test_scope_can_access_parallel_thread_local() {
    let mut scope = Scope::new();

    scope
        .declare(
            "temp1".to_string(),
            DType::F32,
            Mutability::Mutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    scope
        .declare(
            "temp2".to_string(),
            DType::F32,
            Mutability::Mutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    // 両方ThreadLocalなので並列OK
    assert!(scope.can_access_parallel("temp1", "temp2"));
}

#[test]
fn test_scope_can_access_parallel_sharded() {
    let mut scope = Scope::new();

    scope
        .declare(
            "output1".to_string(),
            DType::F32.to_ptr(),
            Mutability::Mutable,
            AccessRegion::ShardedBy(vec![0]),
        )
        .unwrap();

    scope
        .declare(
            "output2".to_string(),
            DType::F32.to_ptr(),
            Mutability::Mutable,
            AccessRegion::ShardedBy(vec![1]),
        )
        .unwrap();

    // 異なる軸でシャーディングされているので並列OK
    assert!(scope.can_access_parallel("output1", "output2"));
}

#[test]
fn test_scope_cannot_access_parallel_mutable_shared() {
    let mut scope = Scope::new();

    scope
        .declare(
            "output".to_string(),
            DType::F32.to_ptr(),
            Mutability::Mutable,
            AccessRegion::Shared,
        )
        .unwrap();

    scope
        .declare(
            "input".to_string(),
            DType::F32.to_ptr(),
            Mutability::Immutable,
            AccessRegion::Shared,
        )
        .unwrap();

    // 片方がMutableでSharedなので並列NG
    assert!(!scope.can_access_parallel("output", "input"));
}

#[test]
fn test_check_scope_var() {
    let mut scope = Scope::new();

    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
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
            AccessRegion::ThreadLocal,
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
            AccessRegion::ThreadLocal,
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
            AccessRegion::Shared,
        )
        .unwrap();

    scope
        .declare(
            "output".to_string(),
            DType::F32.to_ptr(),
            Mutability::Mutable,
            AccessRegion::ShardedBy(vec![0]),
        )
        .unwrap();

    scope
        .declare(
            "i".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
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
