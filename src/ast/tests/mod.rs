#![allow(clippy::approx_constant)]

use super::*;
use crate::ast::helper::*;

// Sub-modules
mod call_return_tests;
mod control_flow_tests;
mod function_program_tests;

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
        Literal::Int(v) => assert_eq!(v, 42),
        _ => panic!("Expected Int literal"),
    }

    let lit = Literal::from(-10isize);
    match lit {
        Literal::Int(v) => assert_eq!(v, -10),
        _ => panic!("Expected Int literal"),
    }
}

#[test]
fn test_literal_from_usize() {
    let lit: Literal = 100usize.into();
    match lit {
        Literal::Int(v) => assert_eq!(v, 100),
        _ => panic!("Expected Int literal"),
    }

    let lit = Literal::from(256usize);
    match lit {
        Literal::Int(v) => assert_eq!(v, 256),
        _ => panic!("Expected Int literal"),
    }
}

#[test]
fn test_literal_dtype() {
    let f32_lit = Literal::F32(3.14);
    assert_eq!(f32_lit.dtype(), DType::F32);

    let int_lit = Literal::Int(42);
    assert_eq!(int_lit.dtype(), DType::Int);

    let int_lit = Literal::Int(100);
    assert_eq!(int_lit.dtype(), DType::Int);
}

#[test]
fn test_children_const() {
    let node = const_f32(3.14);
    let children = node.children();
    assert_eq!(children.len(), 0);
}

#[test]
fn test_children_binary_ops() {
    let a = const_f32(1.0);
    let b = const_f32(2.0);
    let node = a + b;
    let children = node.children();
    assert_eq!(children.len(), 2);

    let node = const_int(3) * const_int(4);
    let children = node.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_children_unary_ops() {
    let node = sqrt(const_f32(4.0));
    let children = node.children();
    assert_eq!(children.len(), 1);

    let node = sin(const_f32(1.0));
    let children = node.children();
    assert_eq!(children.len(), 1);

    let node = recip(const_f32(2.0));
    let children = node.children();
    assert_eq!(children.len(), 1);
}

#[test]
fn test_children_cast() {
    let node = cast(const_f32(3.14), DType::Int);
    let children = node.children();
    assert_eq!(children.len(), 1);
}

#[test]
fn test_children_composite() {
    // (a + b) * c
    let a = const_f32(1.0);
    let b = const_f32(2.0);
    let c = const_f32(3.0);
    let product = (a + b) * c;

    let children = product.children();
    assert_eq!(children.len(), 2);

    // The first child should be the sum node
    let sum_children = children[0].children();
    assert_eq!(sum_children.len(), 2);
}

#[test]
fn test_infer_type_const() {
    let node = const_f32(3.14);
    assert_eq!(node.infer_type(), DType::F32);

    let node = const_int(42);
    assert_eq!(node.infer_type(), DType::Int);

    let node = const_int(100);
    assert_eq!(node.infer_type(), DType::Int);
}

#[test]
fn test_infer_type_binary_ops() {
    // Same types should return that type
    let node = const_f32(1.0) + const_f32(2.0);
    assert_eq!(node.infer_type(), DType::F32);

    let node = const_int(3) * const_int(4);
    assert_eq!(node.infer_type(), DType::Int);

    // Mixed types should return Unknown
    let node = const_f32(1.0) + const_int(2);
    assert_eq!(node.infer_type(), DType::Unknown);
}

#[test]
fn test_infer_type_unary_ops() {
    // Recip preserves type
    let node = recip(const_f32(2.0));
    assert_eq!(node.infer_type(), DType::F32);

    // Math operations return F32
    let node = sqrt(const_f32(4.0));
    assert_eq!(node.infer_type(), DType::F32);

    let node = sin(const_f32(1.0));
    assert_eq!(node.infer_type(), DType::F32);

    let node = log2(const_f32(8.0));
    assert_eq!(node.infer_type(), DType::F32);

    let node = exp2(const_f32(3.0));
    assert_eq!(node.infer_type(), DType::F32);
}

#[test]
fn test_infer_type_cast() {
    let node = cast(const_f32(3.14), DType::Int);
    assert_eq!(node.infer_type(), DType::Int);

    let node = cast(const_int(42), DType::F32);
    assert_eq!(node.infer_type(), DType::F32);
}

#[test]
fn test_infer_type_composite() {
    // (a + b) * c where all are F32
    let a = const_f32(1.0);
    let b = const_f32(2.0);
    let c = const_f32(3.0);
    let expr = (a + b) * c;
    assert_eq!(expr.infer_type(), DType::F32);

    // sqrt(a + b) where a, b are F32
    let a = const_f32(4.0);
    let b = const_f32(5.0);
    let expr = sqrt(a + b);
    assert_eq!(expr.infer_type(), DType::F32);

    // Complex expression with cast
    let a = const_int(10);
    let b = const_int(20);
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
    let scalar = DType::Int;
    assert_eq!(scalar.element_type(), &DType::Int);
}

#[test]
fn test_dtype_deref_type() {
    // Ptr should return pointee type
    let ptr_type = DType::F32.to_ptr();
    assert_eq!(ptr_type.deref_type(), &DType::F32);

    // Non-ptr should return self
    let scalar = DType::Int;
    assert_eq!(scalar.deref_type(), &DType::Int);
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
    let v = var("x");
    assert_eq!(v.children().len(), 0);
    assert_eq!(v.infer_type(), DType::Unknown);
}

#[test]
fn test_load_scalar() {
    let load_node = load(var("input0"), const_int(0), DType::F32);

    // children should include ptr and offset
    let children = load_node.children();
    assert_eq!(children.len(), 2);

    // Type inference: Now returns the explicit dtype field
    let inferred = load_node.infer_type();
    assert_eq!(inferred, DType::F32);
}

#[test]
fn test_load_vector() {
    // Create a proper pointer type for testing
    let ptr_node = cast(var("buffer"), DType::F32.to_ptr());

    let load_node = load_vec(ptr_node, const_int(0), 4, DType::F32.to_vec(4));

    // Type should be Vec<F32, 4>
    let inferred = load_node.infer_type();
    assert_eq!(inferred, DType::F32.to_vec(4));

    // children should include ptr and offset
    let children = load_node.children();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_store() {
    let store_node = store(var("output0"), const_int(0), const_f32(3.14));

    // children should include ptr, offset, and value
    let children = store_node.children();
    assert_eq!(children.len(), 3);

    // Store returns unit type (empty tuple)
    let inferred = store_node.infer_type();
    assert_eq!(inferred, DType::Tuple(vec![]));
}

#[test]
fn test_assign() {
    let assign_node = assign("alu0", const_int(42));

    // children should include only value
    let children = assign_node.children();
    assert_eq!(children.len(), 1);

    // Assign returns the type of the value
    let inferred = assign_node.infer_type();
    assert_eq!(inferred, DType::Int);
}

#[test]
fn test_load_store_map_children() {
    let load_node = load_vec(const_int(1), const_int(2), 4, DType::F32);

    // Map children: multiply each constant by 2
    let mapped = load_node.map_children(&|node| match node {
        AstNode::Const(Literal::Int(n)) => const_int(n * 2),
        _ => node.clone(),
    });

    if let AstNode::Load {
        ptr,
        offset,
        count,
        dtype: _,
    } = mapped
    {
        assert_eq!(*ptr, const_int(2));
        assert_eq!(*offset, const_int(4));
        assert_eq!(count, 4);
    } else {
        panic!("Expected Load node");
    }
}

#[test]
fn test_assign_map_children() {
    let assign_node = assign("x", const_int(10));

    // Map children: increment constant
    let mapped = assign_node.map_children(&|node| match node {
        AstNode::Const(Literal::Int(n)) => const_int(n + 1),
        _ => node.clone(),
    });

    if let AstNode::Assign {
        var: var_name,
        value,
    } = mapped
    {
        assert_eq!(var_name, "x");
        assert_eq!(*value, const_int(11));
    } else {
        panic!("Expected Assign node");
    }
}

// Scope tests
#[test]
fn test_scope_declare() {
    let mut scope = Scope::new();

    scope
        .declare("x".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    assert!(scope.get("x").is_some());
    assert_eq!(scope.get("x").unwrap().dtype, DType::F32);
}

#[test]
fn test_scope_duplicate_declare() {
    let mut scope = Scope::new();

    scope
        .declare("x".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let result = scope.declare("x".to_string(), DType::Int, Mutability::Mutable);

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
        .declare("x".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let result = scope.check_write("x", &DType::F32);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("immutable"));
}

#[test]
fn test_scope_check_write_mutable() {
    let mut scope = Scope::new();

    scope
        .declare("output".to_string(), DType::F32, Mutability::Mutable)
        .unwrap();

    assert!(scope.check_write("output", &DType::F32).is_ok());
}

#[test]
fn test_scope_check_write_type_mismatch() {
    let mut scope = Scope::new();

    scope
        .declare("x".to_string(), DType::F32, Mutability::Mutable)
        .unwrap();

    let result = scope.check_write("x", &DType::Int);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Type mismatch"));
}

#[test]
fn test_scope_parent_lookup() {
    let mut parent = Scope::new();
    parent
        .declare("x".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let child = Scope::with_parent(parent);

    // 親スコープの変数にアクセスできる
    assert!(child.check_read("x").is_ok());
}

#[test]
fn test_check_scope_var() {
    let mut scope = Scope::new();

    scope
        .declare("x".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let var_node = var("x");
    assert!(var_node.check_scope(&scope).is_ok());

    let undefined_var = var("undefined");
    assert!(undefined_var.check_scope(&scope).is_err());
}

#[test]
fn test_check_scope_assign() {
    let mut scope = Scope::new();

    scope
        .declare("x".to_string(), DType::F32, Mutability::Mutable)
        .unwrap();

    let assign_node = assign("x", const_f32(3.14));

    assert!(assign_node.check_scope(&scope).is_ok());
}

#[test]
fn test_check_scope_assign_immutable() {
    let mut scope = Scope::new();

    scope
        .declare("x".to_string(), DType::F32, Mutability::Immutable)
        .unwrap();

    let assign_node = assign("x", const_f32(3.14));

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
        .declare("i".to_string(), DType::Int, Mutability::Immutable)
        .unwrap();

    // output[i] = input[i] * 2.0
    let expr = store(
        var("output"),
        var("i"),
        load(var("input"), var("i"), DType::F32) * const_f32(2.0),
    );

    assert!(expr.check_scope(&scope).is_ok());
}
