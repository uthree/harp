use super::*;
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

#[test]
fn test_infer_type_const() {
    let node = AstNode::Const(3.14f32.into());
    assert_eq!(node.infer_type(), DType::F32);

    let node = AstNode::Const(42isize.into());
    assert_eq!(node.infer_type(), DType::Isize);

    let node = AstNode::Const(100usize.into());
    assert_eq!(node.infer_type(), DType::Usize);
}

#[test]
fn test_infer_type_binary_ops() {
    // Same types should return that type
    let node = AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into());
    assert_eq!(node.infer_type(), DType::F32);

    let node = AstNode::Const(3isize.into()) * AstNode::Const(4isize.into());
    assert_eq!(node.infer_type(), DType::Isize);

    // Mixed types should return Unknown
    let node = AstNode::Const(1.0f32.into()) + AstNode::Const(2isize.into());
    assert_eq!(node.infer_type(), DType::Unknown);
}

#[test]
fn test_infer_type_unary_ops() {
    // Recip preserves type
    let node = recip(AstNode::Const(2.0f32.into()));
    assert_eq!(node.infer_type(), DType::F32);

    // Math operations return F32
    let node = sqrt(AstNode::Const(4.0f32.into()));
    assert_eq!(node.infer_type(), DType::F32);

    let node = sin(AstNode::Const(1.0f32.into()));
    assert_eq!(node.infer_type(), DType::F32);

    let node = log2(AstNode::Const(8.0f32.into()));
    assert_eq!(node.infer_type(), DType::F32);

    let node = exp2(AstNode::Const(3.0f32.into()));
    assert_eq!(node.infer_type(), DType::F32);
}

#[test]
fn test_infer_type_cast() {
    let node = cast(AstNode::Const(3.14f32.into()), DType::Isize);
    assert_eq!(node.infer_type(), DType::Isize);

    let node = cast(AstNode::Const(42isize.into()), DType::F32);
    assert_eq!(node.infer_type(), DType::F32);
}

#[test]
fn test_infer_type_composite() {
    // (a + b) * c where all are F32
    let a = AstNode::Const(1.0f32.into());
    let b = AstNode::Const(2.0f32.into());
    let c = AstNode::Const(3.0f32.into());
    let expr = (a + b) * c;
    assert_eq!(expr.infer_type(), DType::F32);

    // sqrt(a + b) where a, b are F32
    let a = AstNode::Const(4.0f32.into());
    let b = AstNode::Const(5.0f32.into());
    let expr = sqrt(a + b);
    assert_eq!(expr.infer_type(), DType::F32);

    // Complex expression with cast
    let a = AstNode::Const(10isize.into());
    let b = AstNode::Const(20isize.into());
    let casted = cast(a + b, DType::F32);
    let result = sqrt(casted);
    assert_eq!(result.infer_type(), DType::F32);
}

#[test]
fn test_dtype_to_vec() {
    let base_type = DType::F32;
    let vec_type = base_type.to_vec(4);

    match vec_type {
        DType::Vec(elem_type, size) => {
            assert_eq!(*elem_type, DType::F32);
            assert_eq!(size, 4);
        }
        _ => panic!("Expected Vec type"),
    }
}

#[test]
fn test_dtype_from_vec() {
    let vec_type = DType::F32.to_vec(8);

    let result = vec_type.from_vec();
    assert!(result.is_some());

    let (elem_type, size) = result.unwrap();
    assert_eq!(elem_type, &DType::F32);
    assert_eq!(size, 8);

    // Non-vec type should return None
    let scalar = DType::F32;
    assert!(scalar.from_vec().is_none());
}

#[test]
fn test_dtype_to_ptr() {
    let base_type = DType::F32;
    let ptr_type = base_type.to_ptr();

    match ptr_type {
        DType::Ptr(pointee) => {
            assert_eq!(*pointee, DType::F32);
        }
        _ => panic!("Expected Ptr type"),
    }
}

#[test]
fn test_dtype_from_ptr() {
    let ptr_type = DType::F32.to_ptr();

    let result = ptr_type.from_ptr();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), &DType::F32);

    // Non-ptr type should return None
    let scalar = DType::F32;
    assert!(scalar.from_ptr().is_none());
}

#[test]
fn test_dtype_element_type() {
    // Vec should return element type
    let vec_type = DType::F32.to_vec(4);
    assert_eq!(vec_type.element_type(), &DType::F32);

    // Non-vec should return self
    let scalar = DType::Isize;
    assert_eq!(scalar.element_type(), &DType::Isize);
}

#[test]
fn test_dtype_deref_type() {
    // Ptr should return pointee type
    let ptr_type = DType::F32.to_ptr();
    assert_eq!(ptr_type.deref_type(), &DType::F32);

    // Non-ptr should return self
    let scalar = DType::Isize;
    assert_eq!(scalar.deref_type(), &DType::Isize);
}

#[test]
fn test_dtype_is_vec() {
    let vec_type = DType::F32.to_vec(4);
    assert!(vec_type.is_vec());

    let scalar = DType::F32;
    assert!(!scalar.is_vec());
}

#[test]
fn test_dtype_is_ptr() {
    let ptr_type = DType::F32.to_ptr();
    assert!(ptr_type.is_ptr());

    let scalar = DType::F32;
    assert!(!scalar.is_ptr());
}

#[test]
fn test_dtype_nested_types() {
    // Vec of Ptr
    let ptr_type = DType::F32.to_ptr();
    let vec_of_ptr = ptr_type.to_vec(4);

    assert!(vec_of_ptr.is_vec());
    let (elem, size) = vec_of_ptr.from_vec().unwrap();
    assert_eq!(size, 4);
    assert!(elem.is_ptr());

    // Ptr to Vec
    let vec_type = DType::F32.to_vec(8);
    let ptr_to_vec = vec_type.to_ptr();

    assert!(ptr_to_vec.is_ptr());
    let pointee = ptr_to_vec.from_ptr().unwrap();
    assert!(pointee.is_vec());
}

#[test]
fn test_var_node() {
    let var = AstNode::Var("x".to_string());
    assert_eq!(var.children().len(), 0);
    assert_eq!(var.infer_type(), DType::Unknown);
}

#[test]
fn test_load_scalar() {
    let load = AstNode::Load {
        ptr: Box::new(AstNode::Var("input0".to_string())),
        offset: Box::new(AstNode::Const(0usize.into())),
        count: 1,
    };

    // children should include ptr and offset
    let children = load.children();
    assert_eq!(children.len(), 2);

    // Type inference: Var returns Unknown, so deref_type returns Unknown
    // This test demonstrates the structure, actual type depends on context
    let inferred = load.infer_type();
    assert_eq!(inferred, DType::Unknown);
}

#[test]
fn test_load_vector() {
    // Create a proper pointer type for testing
    let ptr_node = AstNode::Cast(
        Box::new(AstNode::Var("buffer".to_string())),
        DType::F32.to_ptr(),
    );

    let load = AstNode::Load {
        ptr: Box::new(ptr_node),
        offset: Box::new(AstNode::Const(0usize.into())),
        count: 4,
    };

    // Type should be Vec<F32, 4>
    let inferred = load.infer_type();
    assert_eq!(inferred, DType::F32.to_vec(4));

    // children should include ptr and offset
    let children = load.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_store() {
    let store = AstNode::Store {
        ptr: Box::new(AstNode::Var("output0".to_string())),
        offset: Box::new(AstNode::Const(0usize.into())),
        value: Box::new(AstNode::Const(3.14f32.into())),
    };

    // children should include ptr, offset, and value
    let children = store.children();
    assert_eq!(children.len(), 3);

    // Store returns unit type (empty tuple)
    let inferred = store.infer_type();
    assert_eq!(inferred, DType::Tuple(vec![]));
}

#[test]
fn test_assign() {
    let assign = AstNode::Assign {
        var: "alu0".to_string(),
        value: Box::new(AstNode::Const(42isize.into())),
    };

    // children should include only value
    let children = assign.children();
    assert_eq!(children.len(), 1);

    // Assign returns the type of the value
    let inferred = assign.infer_type();
    assert_eq!(inferred, DType::Isize);
}

#[test]
fn test_load_store_map_children() {
    let load = AstNode::Load {
        ptr: Box::new(AstNode::Const(1isize.into())),
        offset: Box::new(AstNode::Const(2isize.into())),
        count: 4,
    };

    // Map children: multiply each constant by 2
    let mapped = load.map_children(&|node| match node {
        AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n * 2)),
        _ => node.clone(),
    });

    if let AstNode::Load { ptr, offset, count } = mapped {
        assert_eq!(*ptr, AstNode::Const(Literal::Isize(2)));
        assert_eq!(*offset, AstNode::Const(Literal::Isize(4)));
        assert_eq!(count, 4);
    } else {
        panic!("Expected Load node");
    }
}

#[test]
fn test_assign_map_children() {
    let assign = AstNode::Assign {
        var: "x".to_string(),
        value: Box::new(AstNode::Const(10isize.into())),
    };

    // Map children: increment constant
    let mapped = assign.map_children(&|node| match node {
        AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n + 1)),
        _ => node.clone(),
    });

    if let AstNode::Assign { var, value } = mapped {
        assert_eq!(var, "x");
        assert_eq!(*value, AstNode::Const(Literal::Isize(11)));
    } else {
        panic!("Expected Assign node");
    }
}

// Scope tests
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
            }),
            Box::new(AstNode::Const(2.0f32.into())),
        )),
    };

    assert!(expr.check_scope(&scope).is_ok());
}

// Range tests
#[test]
fn test_range_basic() {
    let mut scope = Scope::new();
    scope
        .declare(
            "i".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0usize.into())),
        step: Box::new(AstNode::Const(1usize.into())),
        stop: Box::new(AstNode::Const(10usize.into())),
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
        .declare(
            "i".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0usize.into())),
        step: Box::new(AstNode::Const(1usize.into())),
        stop: Box::new(AstNode::Const(10usize.into())),
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
        .declare(
            "N".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::Shared,
        )
        .unwrap();

    outer_scope
        .declare(
            "input".to_string(),
            DType::F32.to_ptr(),
            Mutability::Immutable,
            AccessRegion::Shared,
        )
        .unwrap();

    outer_scope
        .declare(
            "output".to_string(),
            DType::F32.to_ptr(),
            Mutability::Mutable,
            AccessRegion::ShardedBy(vec![0]),
        )
        .unwrap();

    // ループ内のスコープ
    let mut loop_scope = Scope::with_parent(outer_scope.clone());
    loop_scope
        .declare(
            "i".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    // for i in 0..N: output[i] = input[i] * 2
    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0usize.into())),
        step: Box::new(AstNode::Const(1usize.into())),
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
        .declare(
            "N".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::Shared,
        )
        .unwrap();

    // ループスコープにループ変数を宣言しない
    let loop_scope = Scope::with_parent(outer_scope.clone());

    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0usize.into())),
        step: Box::new(AstNode::Const(1usize.into())),
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
        .declare(
            "N".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::Shared,
        )
        .unwrap();

    // 外側のループスコープ
    let mut outer_loop_scope = Scope::with_parent(outer_scope.clone());
    outer_loop_scope
        .declare(
            "i".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    // 内側のループスコープ
    let mut inner_loop_scope = Scope::with_parent(outer_loop_scope.clone());
    inner_loop_scope
        .declare(
            "j".to_string(),
            DType::Usize,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    // for j in 0..N: use i and j
    let inner_range = AstNode::Range {
        var: "j".to_string(),
        start: Box::new(AstNode::Const(0usize.into())),
        step: Box::new(AstNode::Const(1usize.into())),
        stop: Box::new(AstNode::Var("N".to_string())),
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Var("i".to_string()), AstNode::Var("j".to_string())],
            scope: Box::new(inner_loop_scope),
        }),
    };

    // for i in 0..N: ...
    let outer_range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(0usize.into())),
        step: Box::new(AstNode::Const(1usize.into())),
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
        .declare(
            "x".to_string(),
            DType::Isize,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    let block = AstNode::Block {
        statements: vec![
            AstNode::Var("x".to_string()),
            AstNode::Const(42isize.into()),
        ],
        scope: Box::new(scope),
    };

    // Blockは最後の文の型を返す
    assert_eq!(block.infer_type(), DType::Isize);
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
        .declare(
            "a".to_string(),
            DType::F32,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    let block = AstNode::Block {
        statements: vec![var("a"), AstNode::Const(1.0f32.into())],
        scope: Box::new(scope),
    };

    // スコープチェックが成功するはず
    let outer_scope = Scope::new();
    assert!(block.check_scope(&outer_scope).is_ok());
}

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
        AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n * 2)),
        _ => node.clone(),
    });

    if let AstNode::Call { name, args } = mapped {
        assert_eq!(name, "mul");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], AstNode::Const(Literal::Isize(6)));
        assert_eq!(args[1], AstNode::Const(Literal::Isize(8)));
    } else {
        panic!("Expected Call node");
    }
}

#[test]
fn test_call_check_scope() {
    let mut scope = Scope::new();
    scope
        .declare(
            "x".to_string(),
            DType::F32,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
        .unwrap();
    scope
        .declare(
            "y".to_string(),
            DType::F32,
            Mutability::Immutable,
            AccessRegion::ThreadLocal,
        )
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
            AccessRegion::ThreadLocal,
        )
        .unwrap();

    let ret = AstNode::Return {
        value: Box::new(var("result")),
    };

    assert!(ret.check_scope(&scope).is_ok());
}

// Function tests
#[test]
fn test_function_new() {
    let params = vec![
        VarDecl {
            name: "a".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            region: AccessRegion::ThreadLocal,
            kind: VarKind::Normal,
        },
        VarDecl {
            name: "b".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            region: AccessRegion::ThreadLocal,
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
        dtype: DType::Isize,
        mutability: Mutability::Immutable,
        region: AccessRegion::ThreadLocal,
        kind: VarKind::Normal,
    }];
    let return_type = DType::Isize;
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

    let func =
        Function::new(FunctionKind::Normal, vec![], DType::Tuple(vec![]), vec![]).unwrap();
    assert!(program.add_function("main".to_string(), func).is_ok());
    assert_eq!(program.functions.len(), 1);
}

#[test]
fn test_program_get_function() {
    let mut program = Program::new("main".to_string());
    let func =
        Function::new(FunctionKind::Normal, vec![], DType::Tuple(vec![]), vec![]).unwrap();
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
    let func =
        Function::new(FunctionKind::Normal, vec![], DType::Tuple(vec![]), vec![]).unwrap();
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
            dtype: DType::Isize,
            mutability: Mutability::Immutable,
            region: AccessRegion::ThreadLocal,
            kind: VarKind::Normal,
        }],
        DType::Isize,
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
        DType::Isize,
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
