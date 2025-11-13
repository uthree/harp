use super::*;

// Range tests
#[test]
fn test_range_basic() {
    let mut scope = Scope::new();
    scope
        .declare("i".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0isize.into())),
        step: Box::new(AstNode::Const(1isize.into())),
        stop: Box::new(AstNode::Const(10isize.into())),
        body: Box::new(AstNode::Block {
            statements: vec![],
            scope: Box::new(scope),
        }),
    };

    // Rangeはunit型を返す
    assert_eq!(range.infer_type(), DType::Tuple(vec![]));
}

#[test]
fn test_range_children() {
    let mut scope = Scope::new();
    scope
        .declare("i".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0isize.into())),
        step: Box::new(AstNode::Const(1isize.into())),
        stop: Box::new(AstNode::Const(10isize.into())),
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Const(1.0f32.into()), AstNode::Const(2.0f32.into())],
            scope: Box::new(scope),
        }),
    };

    let children = range.children();
    // start, step, stop, body = 4個
    assert_eq!(children.len(), 4);
}

#[test]
fn test_range_with_scope() {
    let mut outer_scope = Scope::new();
    outer_scope
        .declare("N".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    outer_scope
        .declare(
            "input".to_string(),
            DType::F32.to_ptr(),
            Mutability::Immutable,
        )
        .unwrap();

    outer_scope
        .declare(
            "output".to_string(),
            DType::F32.to_ptr(),
            Mutability::Mutable,
        )
        .unwrap();

    // ループ内のスコープ
    let mut loop_scope = Scope::with_parent(outer_scope.clone());
    loop_scope
        .declare("i".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    // for i in 0..N: output[i] = input[i] * 2
    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0isize.into())),
        step: Box::new(AstNode::Const(1isize.into())),
        stop: Box::new(AstNode::Var("N".to_string())),
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Store {
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
            }],
            scope: Box::new(loop_scope),
        }),
    };

    // スコープチェック
    assert!(range.check_scope(&outer_scope).is_ok());
}

#[test]
fn test_range_scope_check_undefined_loop_var() {
    let mut outer_scope = Scope::new();
    outer_scope
        .declare("N".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    // ループスコープにループ変数を宣言しない
    let loop_scope = Scope::with_parent(outer_scope.clone());

    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0isize.into())),
        step: Box::new(AstNode::Const(1isize.into())),
        stop: Box::new(AstNode::Var("N".to_string())),
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Var("i".to_string())],
            scope: Box::new(loop_scope),
        }),
    };

    // ループ変数が宣言されていないのでエラー
    let result = range.check_scope(&outer_scope);
    assert!(result.is_err());
}

#[test]
fn test_range_nested() {
    let mut outer_scope = Scope::new();
    outer_scope
        .declare("N".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    // 外側のループスコープ
    let mut outer_loop_scope = Scope::with_parent(outer_scope.clone());
    outer_loop_scope
        .declare("i".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    // 内側のループスコープ
    let mut inner_loop_scope = Scope::with_parent(outer_loop_scope.clone());
    inner_loop_scope
        .declare("j".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    // for j in 0..N: use i and j
    let inner_range = AstNode::Range {
        var: "j".to_string(),
        start: Box::new(AstNode::Const(0isize.into())),
        step: Box::new(AstNode::Const(1isize.into())),
        stop: Box::new(AstNode::Var("N".to_string())),
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Var("i".to_string()), AstNode::Var("j".to_string())],
            scope: Box::new(inner_loop_scope),
        }),
    };

    // for i in 0..N: ...
    let outer_range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0isize.into())),
        step: Box::new(AstNode::Const(1isize.into())),
        stop: Box::new(AstNode::Var("N".to_string())),
        body: Box::new(AstNode::Block {
            statements: vec![inner_range],
            scope: Box::new(outer_loop_scope),
        }),
    };

    // ネストしたループのスコープチェック
    assert!(outer_range.check_scope(&outer_scope).is_ok());
}

// Block tests
#[test]
fn test_block_basic() {
    let mut scope = Scope::new();
    scope
        .declare("x".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    let block = AstNode::Block {
        statements: vec![
            AstNode::Var("x".to_string()),
            AstNode::Const(42isize.into()),
        ],
        scope: Box::new(scope),
    };

    // Blockは最後の文の型を返す
    assert_eq!(block.infer_type(), DType::Int);
}

#[test]
fn test_block_children() {
    let block = AstNode::Block {
        statements: vec![AstNode::Const(1isize.into()), AstNode::Const(2isize.into())],
        scope: Box::new(Scope::new()),
    };

    let children = block.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_block_check_scope() {
    let mut scope = Scope::new();
    scope
        .declare("a".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let block = AstNode::Block {
        statements: vec![var("a"), AstNode::Const(1.0f32.into())],
        scope: Box::new(scope),
    };

    // スコープチェックが成功するはず
    let outer_scope = Scope::new();
    assert!(block.check_scope(&outer_scope).is_ok());
}
