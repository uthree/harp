use harp::op::{Load, OpAdd, OpSub, OpMul, OpDiv, Operator};
use harp::tensor::Tensor;
use rstest::rstest;
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
