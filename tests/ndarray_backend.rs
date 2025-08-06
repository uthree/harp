// tests/ndarray_backend.rs

use harp::backend::{Buffer, TryIntoNdarray};
use harp::ast::DType;
use ndarray::{arr2, ArrayD};

/// Tests that the `Buffer` trait is implemented correctly for `ndarray::Array` with `f32` elements.
#[test]
fn test_ndarray_buffer_trait_f32() {
    let mut arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
    let buffer: &mut dyn Buffer = &mut arr;

    assert_eq!(buffer.dtype(), DType::F32);
    assert_eq!(buffer.shape(), vec![2, 2]);
    assert_eq!(buffer.size(), 4);
}

/// Tests that the `Buffer` trait is implemented correctly for `ndarray::Array` with `i64` elements.
#[test]
fn test_ndarray_buffer_trait_i64() {
    let mut arr = arr2(&[[1i64, 2], [3, 4]]);
    let buffer: &mut dyn Buffer = &mut arr;

    assert_eq!(buffer.dtype(), DType::I64);
    assert_eq!(buffer.shape(), vec![2, 2]);
}

/// Tests that a `Buffer` can be converted back to an `ndarray::ArrayD` and that the data is preserved.
#[test]
fn test_buffer_to_ndarray_roundtrip() {
    let mut arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
    let original_arr_dyn = arr.clone().into_dyn();

    // Convert back to ndarray using the extension trait
    let new_arr_f32 = arr.try_into_ndarray::<f32>().unwrap();
    assert_eq!(new_arr_f32.shape(), &[2, 2]);
    assert_eq!(new_arr_f32, original_arr_dyn);

    // Try converting to a wrong type
    let new_arr_i64 = arr.try_into_ndarray::<i64>();
    assert!(new_arr_i64.is_none());
}

/// Tests that an empty `Buffer` can be converted to an `ndarray::ArrayD`.
#[test]
fn test_empty_buffer_to_ndarray() {
    let mut arr = ArrayD::<f32>::from_shape_vec(vec![2, 0, 2], vec![]).unwrap();

    assert_eq!(arr.size(), 0);
    let new_arr = arr.try_into_ndarray::<f32>().unwrap();
    assert_eq!(new_arr.shape(), &[2, 0, 2]);
    assert_eq!(new_arr.len(), 0);
}