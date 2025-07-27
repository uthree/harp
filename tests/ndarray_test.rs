use harp::prelude::*;
use ndarray::{ArrayD, arr1, arr2};

#[test]
fn test_ndarray_roundtrip() {
    let arr1: ArrayD<f32> = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let tensor: Tensor = arr1.clone().into();
    let arr2: ArrayD<f32> = tensor.into();
    assert_eq!(arr1, arr2);
}

#[test]
fn test_ndarray_computation() {
    let arr_a: ArrayD<f32> = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
    let arr_b: ArrayD<f32> = arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn();

    let tensor_a: Tensor = arr_a.clone().into();
    let tensor_b: Tensor = arr_b.clone().into();

    let result_tensor = tensor_a + tensor_b;
    let result_arr: ArrayD<f32> = result_tensor.into();

    let expected_arr = arr_a + arr_b;
    assert_eq!(result_arr, expected_arr);
}
