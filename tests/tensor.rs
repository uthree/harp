use harp::node::{self, constant, Node};
use harp::op::{Load, OpAdd, OpSub, OpMul, OpDiv, Operator, Reduce, Reshape};
use harp::tensor::{ShapeTracker, Tensor};
use rstest::rstest;
use std::rc::Rc;
use std::sync::Arc;

#[test]
fn test_tensor_creation_with_shape() {
    let shape = vec![2, 3];
    let tensor = Tensor::new_load(shape.clone());

    assert_eq!(*tensor.shape(), shape);
    assert_eq!(tensor.data.op.name(), "Load");
    assert!(tensor.data.op.as_any().is::<Load>());
    assert!(tensor.data.src.is_empty());
}

#[rstest]
#[case("add", |a, b| a + b, "OpAdd")]
#[case("sub", |a, b| a - b, "OpSub")]
#[case("mul", |a, b| a * b, "OpMul")]
#[case("div", |a, b| a / b, "OpDiv")]
fn test_tensor_binary_ops(
    #[case] op_name: &str,
    #[case] op_fn: impl Fn(Tensor, Tensor) -> Tensor,
    #[case] expected_op_name: &str,
) {
    let shape = vec![4, 5];
    let a = Tensor::new_load(shape.clone());
    let b = Tensor::new_load(shape.clone());
    let c = op_fn(a.clone(), b.clone());

    assert_eq!(*c.shape(), shape);
    assert_eq!(
        c.data.op.name(),
        expected_op_name,
        "Failed for operation: {}",
        op_name
    );
    assert_eq!(c.data.src.len(), 2);
    assert!(Arc::ptr_eq(&a.data, &c.data.src[0].data));
    assert!(Arc::ptr_eq(&b.data, &c.data.src[1].data));
}

#[test]
#[should_panic(expected = "Tensor shapes must match for Add")]
fn test_add_shape_mismatch_panic() {
    let a = Tensor::new_load(vec![2, 3]);
    let b = Tensor::new_load(vec![3, 2]);
    let _ = a + b;
}

#[test]
fn test_tensor_reshape() {
    let a = Tensor::new_load(vec![2, 6]);
    let b = a.clone().reshape(vec![3, 4]);

    assert_eq!(*b.shape(), vec![3, 4]);
    assert_eq!(b.data.op.name(), "Reshape");
    assert!(b.data.op.as_any().is::<Reshape>());
    assert_eq!(b.data.src.len(), 1);
    assert!(Arc::ptr_eq(&a.data, &b.data.src[0].data));
}

#[test]
#[should_panic(expected = "Cannot reshape tensor of size 12 to shape [3, 5] with size 15")]
fn test_tensor_reshape_panic() {
    let a = Tensor::new_load(vec![2, 6]);
    a.reshape(vec![3, 5]);
}


#[test]
fn test_tensor_sum() {
    let a = Tensor::new_load(vec![2, 3, 4]);
    let axis_to_reduce = 1;
    let b = a.clone().sum(axis_to_reduce);

    assert_eq!(*b.shape(), vec![2, 4]);
    assert_eq!(b.data.op.name(), "Reduce");
    assert_eq!(b.data.src.len(), 1);
    assert!(Arc::ptr_eq(&a.data, &b.data.src[0].data));

    if let Some(reduce_op) = b.data.op.as_any().downcast_ref::<Reduce>() {
        assert_eq!(reduce_op.op.name(), "OpAdd");
        assert_eq!(reduce_op.axis, axis_to_reduce);
    } else {
        panic!("Operator was not a Reduce operator");
    }
}

#[test]
fn test_compile_add() {
    let shape = vec![10];
    let a = Tensor::new_load(shape.clone());
    let b = Tensor::new_load(shape.clone());
    let c = a + b;

    let idx = Rc::new(node::capture("idx"));
    let tracker = ShapeTracker {
        dims: shape.iter().map(|&d| Rc::new(constant(d))).collect(),
        index_expr: vec![idx.clone()],
    };

    let compiled_node = c.compile(&tracker);

    assert_eq!(compiled_node.op().name(), "OpAdd");
    assert_eq!(compiled_node.src().len(), 2);

    let lhs = &compiled_node.src()[0];
    assert_eq!(lhs.op().name(), "Load");
    assert_eq!(lhs.src().len(), 1);
    assert_eq!(lhs.src()[0], *idx);

    let rhs = &compiled_node.src()[1];
    assert_eq!(rhs.op().name(), "Load");
    assert_eq!(rhs.src().len(), 1);
    assert_eq!(rhs.src()[0], *idx);
}

#[test]
fn test_compile_reshape() {
    let a = Tensor::new_load(vec![2, 6]);
    let b = a.reshape(vec![3, 4]);

    let idx = Rc::new(node::capture("idx"));
    let initial_tracker = ShapeTracker {
        dims: vec![Rc::new(constant(3u64)), Rc::new(constant(4u64))],
        index_expr: vec![idx.clone()],
    };

    let compiled_node = b.compile(&initial_tracker);

    // The compiled node should be a Load from the original source,
    // because the current reshape compile logic just passes through.
    assert_eq!(compiled_node.op().name(), "Load");
    assert_eq!(compiled_node.src().len(), 1);
    assert_eq!(compiled_node.src()[0], *idx);
}

#[test]
fn test_compile_sum() {
    let a = Tensor::new_load(vec![4]);
    let c = a.sum(0);

    // Compile the sum. The output is a scalar, so the tracker is simple.
    let tracker = ShapeTracker {
        dims: vec![],
        index_expr: vec![], // No index for a scalar output
    };
    let compiled_node = c.compile(&tracker);

    // The result should be: (0.0 + a[0]) + a[1] + a[2] + a[3]
    // Let's check the structure.
    // Top node is OpAdd
    assert_eq!(compiled_node.op().name(), "OpAdd");
    // Right child is Load(3)
    let load_3 = &compiled_node.src()[1];
    assert_eq!(load_3.op().name(), "Load");
    assert_eq!(load_3.src()[0], constant(3.0));

    // Left child is another OpAdd
    let add_2 = &compiled_node.src()[0];
    assert_eq!(add_2.op().name(), "OpAdd");
    // Right child is Load(2)
    let load_2 = &add_2.src()[1];
    assert_eq!(load_2.op().name(), "Load");
    assert_eq!(load_2.src()[0], constant(2.0));
}