use super::super::*;
use crate::ast::helper::*;

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
