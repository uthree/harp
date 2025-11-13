use super::*;

// Function tests
#[test]
fn test_function_new() {
    let params = vec![
        VarDecl {
            name: "a".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        },
        VarDecl {
            name: "b".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        },
    ];
    let return_type = DType::F32;
    let body = vec![AstNode::Return {
        value: Box::new(var("a") + var("b")),
    }];

    let func = Function::new(FunctionKind::Normal, params, return_type, body);
    assert!(func.is_ok());

    let func = func.unwrap();
    assert_eq!(func.params.len(), 2);
    assert_eq!(func.return_type, DType::F32);
    assert_eq!(func.kind, FunctionKind::Normal);

    // bodyはBlock nodeになっている
    match &*func.body {
        AstNode::Block { statements, .. } => {
            assert_eq!(statements.len(), 1);
        }
        _ => panic!("Expected Block node for function body"),
    }
}

#[test]
fn test_function_check_body() {
    let params = vec![VarDecl {
        name: "x".to_string(),
        dtype: DType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::Normal,
    }];
    let return_type = DType::Int;
    let body = vec![AstNode::Return {
        value: Box::new(var("x") * AstNode::Const(2isize.into())),
    }];

    let func = Function::new(FunctionKind::Normal, params, return_type, body).unwrap();
    assert!(func.check_body().is_ok());
}

#[test]
fn test_function_infer_return_type() {
    let params = vec![];
    let return_type = DType::F32;
    let body = vec![AstNode::Return {
        value: Box::new(AstNode::Const(1.0f32.into())),
    }];

    let func = Function::new(FunctionKind::Normal, params, return_type, body).unwrap();
    assert_eq!(func.infer_return_type(), DType::F32);
}

// Program tests
#[test]
fn test_program_new() {
    let program = Program::new("main".to_string());
    assert_eq!(program.entry_point, "main");
    assert_eq!(program.functions.len(), 0);
}

#[test]
fn test_program_add_function() {
    let mut program = Program::new("main".to_string());

    let func = Function::new(FunctionKind::Normal, vec![], DType::Tuple(vec![]), vec![]).unwrap();
    assert!(program.add_function("main".to_string(), func).is_ok());
    assert_eq!(program.functions.len(), 1);
}

#[test]
fn test_program_get_function() {
    let mut program = Program::new("main".to_string());
    let func = Function::new(FunctionKind::Normal, vec![], DType::Tuple(vec![]), vec![]).unwrap();
    program.add_function("main".to_string(), func).unwrap();

    assert!(program.get_function("main").is_some());
    assert!(program.get_function("nonexistent").is_none());
}

#[test]
fn test_program_validate() {
    let mut program = Program::new("main".to_string());

    // エントリーポイントがない場合はエラー
    assert!(program.validate().is_err());

    // エントリーポイントを追加
    let func = Function::new(FunctionKind::Normal, vec![], DType::Tuple(vec![]), vec![]).unwrap();
    program.add_function("main".to_string(), func).unwrap();

    // 成功するはず
    assert!(program.validate().is_ok());
}

#[test]
fn test_program_with_function_call() {
    let mut program = Program::new("main".to_string());

    // helper関数: double(x) = x * 2
    let double_func = Function::new(
        FunctionKind::Normal,
        vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::Int,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }],
        DType::Int,
        vec![AstNode::Return {
            value: Box::new(var("x") * AstNode::Const(2isize.into())),
        }],
    )
    .unwrap();
    program
        .add_function("double".to_string(), double_func)
        .unwrap();

    // main関数: Call double(5)
    let main_func = Function::new(
        FunctionKind::Normal,
        vec![],
        DType::Int,
        vec![AstNode::Call {
            name: "double".to_string(),
            args: vec![AstNode::Const(5isize.into())],
        }],
    )
    .unwrap();
    program.add_function("main".to_string(), main_func).unwrap();

    // プログラムの検証
    assert!(program.validate().is_ok());
    assert!(program.has_function("double"));
    assert!(program.has_function("main"));
}
