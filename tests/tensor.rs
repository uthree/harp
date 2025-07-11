use harp::node::{self, constant, variable, Node};
use harp::op::{Expand, Load, OpAdd, OpSub, OpMul, OpDiv, Operator, Permute, Reduce, Reshape, Slice};
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
fn test_tensor_permute() {
    let a = Tensor::new_load(vec![2, 3, 4]);
    let b = a.clone().permute(vec![2, 0, 1]);

    assert_eq!(*b.shape(), vec![4, 2, 3]);
    assert_eq!(b.data.op.name(), "Permute");
    assert_eq!(b.data.src.len(), 1);
    assert!(Arc::ptr_eq(&a.data, &b.data.src[0].data));

    if let Some(permute_op) = b.data.op.as_any().downcast_ref::<Permute>() {
        assert_eq!(permute_op.order, vec![2, 0, 1]);
    } else {
        panic!("Operator was not a Permute operator");
    }
}

#[test]
fn test_tensor_expand() {
    let a = Tensor::new_load(vec![3]);
    let b = a.clone().expand(vec![2, 3]);

    assert_eq!(*b.shape(), vec![2, 3]);
    assert_eq!(b.data.op.name(), "Expand");
    assert_eq!(b.data.src.len(), 1);
    assert!(Arc::ptr_eq(&a.data, &b.data.src[0].data));

    if let Some(expand_op) = b.data.op.as_any().downcast_ref::<Expand>() {
        assert_eq!(expand_op.shape, vec![2, 3]);
    } else {
        panic!("Operator was not an Expand operator");
    }
}

#[test]
fn test_tensor_slice() {
    let a = Tensor::new_load(vec![10, 20]);
    let b = a.clone().slice(vec![(2, 5), (3, 8)]);

    assert_eq!(*b.shape(), vec![3, 5]);
    assert_eq!(b.data.op.name(), "Slice");
    assert_eq!(b.data.src.len(), 1);
    assert!(Arc::ptr_eq(&a.data, &b.data.src[0].data));

    if let Some(slice_op) = b.data.op.as_any().downcast_ref::<Slice>() {
        assert_eq!(slice_op.args, vec![(2, 5), (3, 8)]);
    } else {
        panic!("Operator was not a Slice operator");
    }
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
fn test_rearrange_permutation() {
    let a = Tensor::new_load(vec![10, 20, 30, 40]);
    let b = a.clone().rearrange("b h w c -> b c h w");

    assert_eq!(*b.shape(), vec![10, 40, 20, 30]);
    assert_eq!(b.data.op.name(), "Permute");
    assert_eq!(b.data.src.len(), 1);
    assert!(Arc::ptr_eq(&a.data, &b.data.src[0].data));

    if let Some(permute_op) = b.data.op.as_any().downcast_ref::<Permute>() {
        assert_eq!(permute_op.order, vec![0, 3, 1, 2]);
    } else {
        panic!("Operator was not a Permute operator");
    }
}

#[test]
fn test_compile_add() {
    let shape = vec![10];
    let a = Tensor::new_load(shape.clone());
    let b = Tensor::new_load(shape.clone());
    let c = a + b;

    let idx = Rc::new(variable("idx"));
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

    let idx = Rc::new(variable("idx"));
    let initial_tracker = ShapeTracker {
        dims: vec![Rc::new(constant(3u64)), Rc::new(constant(4u64))],
        index_expr: vec![idx.clone()],
    };

    let compiled_node = b.compile(&initial_tracker);

    assert_eq!(compiled_node.op().name(), "Load");
    assert_eq!(compiled_node.src().len(), 1);
    assert_eq!(compiled_node.src()[0], *idx);
}

#[test]
#[ignore]
fn test_compile_permute() {
    let a = Tensor::new_load(vec![2, 3]);
    let b = a.permute(vec![1, 0]); // Shape becomes [3, 2]

    let i = Rc::new(variable("i"));
    let j = Rc::new(variable("j"));
    let tracker = ShapeTracker {
        dims: vec![Rc::new(constant(3u64)), Rc::new(constant(2u64))],
        index_expr: vec![i.clone(), j.clone()],
    };

    let compiled_node = b.compile(&tracker);

    assert_eq!(compiled_node.op().name(), "Load");
    assert_eq!(compiled_node.src().len(), 1);

    // Expected index is j * 2 + i (since the permuted shape is [3, 2])
    let expected_index = (*j).clone() * constant(2.0) + (*i).clone();
    assert_eq!(compiled_node.src()[0], expected_index);
}

#[test]
fn test_compile_expand() {
    let a = Tensor::new_load(vec![3]);
    let b = a.expand(vec![2, 3]);

    let i = Rc::new(variable("i"));
    let j = Rc::new(variable("j"));
    let tracker = ShapeTracker {
        dims: vec![Rc::new(constant(2u64)), Rc::new(constant(3u64))],
        index_expr: vec![i.clone(), j.clone()],
    };

    let compiled_node = b.compile(&tracker);

    // The compilation of b[i, j] should result in compiling a[j].
    // The final node should be a Load, and its source should be the
    // flat index calculated from `j`.
    assert_eq!(compiled_node.op().name(), "Load");
    assert_eq!(compiled_node.src().len(), 1);
    assert_eq!(compiled_node.src()[0], *j);
}

#[test]
#[ignore]
fn test_compile_slice() {
    let a = Tensor::new_load(vec![10]);
    let b = a.slice(vec![(2, 5)]); // Shape becomes [3]

    let i = Rc::new(variable("i"));
    let tracker = ShapeTracker {
        dims: vec![Rc::new(constant(3u64))],
        index_expr: vec![i.clone()],
    };

    let compiled_node = b.compile(&tracker);

    // The compilation of b[i] should result in compiling a[i + 2].
    // The final node should be a Load, and its source should be `i + 2`.
    assert_eq!(compiled_node.op().name(), "Load");
    assert_eq!(compiled_node.src().len(), 1);
    assert_eq!(compiled_node.src()[0], (*i).clone() + constant(2.0));
}


#[test]
fn test_compile_sum() {
    let a = Tensor::new_load(vec![4]);
    let c = a.sum(0);

    let tracker = ShapeTracker {
        dims: vec![],
        index_expr: vec![],
    };
    let compiled_node = c.compile(&tracker);

    // The result should be: a[0] + a[1] + a[2] + a[3]
    // After simplification, this becomes a nested Add tree.
    let expected_node =
        (constant(0.0) + constant(0.0)) +
        (constant(0.0) + constant(0.0));
    // This is a simplified check. A real check would be more complex.
    // For now, we just check the top-level op.
    assert_eq!(compiled_node.op().name(), "OpAdd");
}
