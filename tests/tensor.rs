use harp::node::{self, Node};
use harp::op::{Load, OpAdd, OpSub, OpMul, OpDiv, Operator, Reduce};
use harp::tensor::{ShapeTracker, Tensor};
use rstest::rstest;
use std::rc::Rc;
use std::sync::Arc;

#[test]
fn test_tensor_creation() {
    let tensor = Tensor::new_load();
    let tensor_data = tensor.data;

    // Check the operator
    assert_eq!(tensor_data.op.name(), "Load");
    assert!(tensor_data.op.as_any().is::<Load>());
    assert!(tensor_data.src.is_empty());
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
    let a = Tensor::new_load();
    let b = Tensor::new_load();
    let c = op_fn(a.clone(), b.clone());

    let c_data = c.data;

    // Check the operator name
    assert_eq!(
        c_data.op.name(),
        expected_op_name,
        "Failed for operation: {}",
        op_name
    );

    // Check the sources
    assert_eq!(c_data.src.len(), 2);
    assert!(Arc::ptr_eq(&a.data, &c_data.src[0].data));
    assert!(Arc::ptr_eq(&b.data, &c_data.src[1].data));
}

#[test]
fn test_tensor_sum() {
    let a = Tensor::new_load();
    let axis_to_reduce = 1;
    let b = a.clone().sum(axis_to_reduce);

    let b_data = b.data;

    // Check that the top-level operator is Reduce
    assert_eq!(b_data.op.name(), "Reduce");
    assert_eq!(b_data.src.len(), 1);
    assert!(Arc::ptr_eq(&a.data, &b_data.src[0].data));

    // Check the inner operator of Reduce
    if let Some(reduce_op) = b_data.op.as_any().downcast_ref::<Reduce>() {
        assert_eq!(reduce_op.op.name(), "OpAdd");
        assert_eq!(reduce_op.axis, axis_to_reduce);
    } else {
        panic!("Operator was not a Reduce operator");
    }
}

#[test]
fn test_compile_add() {
    // 1. Create the Tensor graph for `a + b`
    let a = Tensor::new_load();
    let b = Tensor::new_load();
    let c = a + b;

    // 2. Create a dummy ShapeTracker
    // It needs an index expression for the Load ops.
    // Let's say the index is represented by a capture node named "idx".
    let idx = Rc::new(node::capture("idx"));
    let tracker = ShapeTracker {
        dims: vec![], // Not used in this test
        index_expr: vec![idx.clone()],
    };

    // 3. Compile the graph
    let compiled_node = c.compile(&tracker);

    // 4. Assert the structure of the resulting Node graph
    assert_eq!(compiled_node.op().name(), "OpAdd");
    assert_eq!(compiled_node.src().len(), 2);

    // Check the left-hand side of the addition
    let lhs = &compiled_node.src()[0];
    assert_eq!(lhs.op().name(), "Load");
    assert_eq!(lhs.src().len(), 1);
    assert_eq!(lhs.src()[0], *idx); // Check it's loading from "idx"

    // Check the right-hand side of the addition
    let rhs = &compiled_node.src()[1];
    assert_eq!(rhs.op().name(), "Load");
    assert_eq!(rhs.src().len(), 1);
    assert_eq!(rhs.src()[0], *idx);
}
