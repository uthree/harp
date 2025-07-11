use harp::prelude::*;

#[test]
fn test_tensor_full() {
    let shape = vec![2, 3];
    let value = 7.0;
    let tensor = Tensor::full(shape.clone(), value);

    // The resulting tensor should be an Expand operation
    assert_eq!(*tensor.shape(), shape);
    assert_eq!(tensor.data.op.name(), "Expand");
    assert_eq!(tensor.data.src.len(), 1);

    // The source of the Expand should be a scalar Const tensor
    let source_tensor = &tensor.data.src[0];
    assert_eq!(source_tensor.shape(), &vec![] as &Vec<u64>); // Scalar shape
    assert_eq!(source_tensor.data.op.name(), "Const");

    if let Some(const_op) = source_tensor.data.op.as_any().downcast_ref::<Const>() {
        assert_eq!(*const_op.0.as_any().downcast_ref::<f64>().unwrap(), value);
    } else {
        panic!("Source operator was not a Const operator");
    }
}

#[test]
fn test_tensor_uniform() {
    let shape = vec![10, 10];
    let tensor = Tensor::uniform(shape.clone());

    // The resulting tensor should be a leaf node with the OpUniform operator
    assert_eq!(*tensor.shape(), shape);
    assert_eq!(tensor.data.op.name(), "OpUniform");
    assert!(tensor.data.op.as_any().is::<OpUniform>());
    assert!(tensor.data.src.is_empty());
}

#[test]
fn test_tensor_randn() {
    let shape = vec![100, 100];
    let tensor = Tensor::randn(shape.clone());

    // The resulting tensor should be a leaf node with the OpRandn operator
    assert_eq!(*tensor.shape(), shape);
    assert_eq!(tensor.data.op.name(), "OpRandn");
    assert!(tensor.data.op.as_any().is::<OpRandn>());
    assert!(tensor.data.src.is_empty());
}
