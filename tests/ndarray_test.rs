use harp::prelude::*;
use ndarray::{arr2, ArrayD};

#[test]
fn test_ndarray_roundtrip() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr1: ArrayD<f32> = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
    let tensor: Tensor<f32> = arr1.clone().into();
    let arr2: ArrayD<f32> = tensor.into();
    assert_eq!(arr1, arr2);
}

#[test]
fn test_ndarray_computation() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
    let arr_b: ArrayD<f32> = arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn();

    // Perform computation using harp's Tensor
    let tensor_a: Tensor<f32> = arr_a.clone().into();
    let tensor_b: Tensor<f32> = arr_b.clone().into();
    let tensor_c = &tensor_a + &tensor_b;
    let arr_c_from_tensor: ArrayD<f32> = tensor_c.into();

    // Perform computation using ndarray
    let expected_arr_c = &arr_a + &arr_b;

    assert_eq!(arr_c_from_tensor, expected_arr_c);
}