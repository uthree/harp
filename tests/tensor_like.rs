use harp::prelude::*;

fn check_full_structure(tensor: &Tensor, shape: &[u64], value: f64) {
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.data.op.name(), "Expand");
    assert_eq!(tensor.data.src.len(), 1);

    let source_tensor = &tensor.data.src[0];
    assert_eq!(source_tensor.shape(), &vec![] as &Vec<u64>);
    assert_eq!(source_tensor.data.op.name(), "Const");

    if let Some(const_op) = source_tensor.data.op.as_any().downcast_ref::<Const>() {
        assert_eq!(*const_op.0.as_any().downcast_ref::<f64>().unwrap(), value);
    } else {
        panic!("Source operator was not a Const operator");
    }
}

#[test]
fn test_tensor_zeros() {
    let shape = vec![2, 3];
    let tensor = Tensor::zeros(shape.clone());
    check_full_structure(&tensor, &shape, 0.0);
}

#[test]
fn test_tensor_ones() {
    let shape = vec![4, 5, 6];
    let tensor = Tensor::ones(shape.clone());
    check_full_structure(&tensor, &shape, 1.0);
}

#[test]
fn test_tensor_zeros_like() {
    let shape = vec![7, 8];
    let other = Tensor::uniform(shape.clone());
    let tensor = Tensor::zeros_like(&other);
    check_full_structure(&tensor, &shape, 0.0);
}

#[test]
fn test_tensor_ones_like() {
    let shape = vec![1];
    let other = Tensor::uniform(shape.clone());
    let tensor = Tensor::ones_like(&other);
    check_full_structure(&tensor, &shape, 1.0);
}
