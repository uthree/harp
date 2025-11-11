use super::super::*;
use crate::ast::helper::*;

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
        dtype: DType::F32,
    };

    // children should include ptr and offset
    let children = load.children();
    assert_eq!(children.len(), 2);

    // Type inference: Now returns the explicit dtype field
    let inferred = load.infer_type();
    assert_eq!(inferred, DType::F32);
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
        dtype: DType::F32.to_vec(4),
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
        dtype: DType::F32,
    };

    // Map children: multiply each constant by 2
    let mapped = load.map_children(&|node| match node {
        AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n * 2)),
        _ => node.clone(),
    });

    if let AstNode::Load { ptr, offset, count, dtype: _ } = mapped {
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
