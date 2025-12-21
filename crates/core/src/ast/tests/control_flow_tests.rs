use super::*;

// Range tests
#[test]
fn test_range_basic() {
    let mut scope = Scope::new();
    scope
        .declare("i".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    let range_node = range(
        "i",
        const_int(0),
        const_int(1),
        const_int(10),
        block(vec![], scope),
    );

    // Rangeはunit型を返す
    assert_eq!(range_node.infer_type(), DType::Tuple(vec![]));
}

#[test]
fn test_range_children() {
    let mut scope = Scope::new();
    scope
        .declare("i".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    let range_node = range(
        "i",
        const_int(0),
        const_int(1),
        const_int(10),
        block(vec![const_f32(1.0), const_f32(2.0)], scope),
    );

    let children = range_node.children();
    // start, step, stop, body = 4個
    assert_eq!(children.len(), 4);
}

#[test]
fn test_range_with_scope() {
    let mut outer_scope = Scope::new();
    outer_scope
        .declare("N".to_string(), DType::I64, Mutability::Immutable)
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
        .declare("i".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    // for i in 0..N: output[i] = input[i] * 2
    let range_node = range(
        "i",
        const_int(0),
        const_int(1),
        var("N"),
        block(
            vec![store(
                var("output"),
                var("i"),
                load(var("input"), var("i"), DType::F32) * const_f32(2.0),
            )],
            loop_scope,
        ),
    );

    // スコープチェック
    assert!(range_node.check_scope(&outer_scope).is_ok());
}

#[test]
fn test_range_scope_check_undefined_loop_var() {
    let mut outer_scope = Scope::new();
    outer_scope
        .declare("N".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    // ループスコープにループ変数を宣言しない
    let loop_scope = Scope::with_parent(outer_scope.clone());

    let range_node = range(
        "i",
        const_int(0),
        const_int(1),
        var("N"),
        block(vec![var("i")], loop_scope),
    );

    // ループ変数が宣言されていないのでエラー
    let result = range_node.check_scope(&outer_scope);
    assert!(result.is_err());
}

#[test]
fn test_range_nested() {
    let mut outer_scope = Scope::new();
    outer_scope
        .declare("N".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    // 外側のループスコープ
    let mut outer_loop_scope = Scope::with_parent(outer_scope.clone());
    outer_loop_scope
        .declare("i".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    // 内側のループスコープ
    let mut inner_loop_scope = Scope::with_parent(outer_loop_scope.clone());
    inner_loop_scope
        .declare("j".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    // for j in 0..N: use i and j
    let inner_range = range(
        "j",
        const_int(0),
        const_int(1),
        var("N"),
        block(vec![var("i"), var("j")], inner_loop_scope),
    );

    // for i in 0..N: ...
    let outer_range = range(
        "i",
        const_int(0),
        const_int(1),
        var("N"),
        block(vec![inner_range], outer_loop_scope),
    );

    // ネストしたループのスコープチェック
    assert!(outer_range.check_scope(&outer_scope).is_ok());
}

// Block tests
#[test]
fn test_block_basic() {
    let mut scope = Scope::new();
    scope
        .declare("x".to_string(), DType::I64, Mutability::Immutable)
        .unwrap();

    let block_node = block(vec![var("x"), const_int(42)], scope);

    // Blockは最後の文の型を返す
    assert_eq!(block_node.infer_type(), DType::I64);
}

#[test]
fn test_block_children() {
    let block_node = block(vec![const_int(1), const_int(2)], Scope::new());

    let children = block_node.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_block_check_scope() {
    let mut scope = Scope::new();
    scope
        .declare("a".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let block_node = block(vec![var("a"), const_f32(1.0)], scope);

    // スコープチェックが成功するはず
    let outer_scope = Scope::new();
    assert!(block_node.check_scope(&outer_scope).is_ok());
}
