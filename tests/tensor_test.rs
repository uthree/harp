use harp::backends::{Backend, CpuBackend};
use harp::dot::ToDot;
use harp::dtype::DType;
use harp::tensor::Tensor;
use ndarray::array;
use rstest::rstest;
use std::sync::Arc;

#[test]
fn test_tensor_addition() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let t1 = Tensor::from_vec(vec![1.0f32; 10], vec![10], backend.clone());
    let t2 = Tensor::from_vec(vec![2.0f32; 10], vec![10], backend.clone());
    let t3 = &t1 + &t2;
    let result = t3.to_ndarray::<f32>();
    assert_eq!(
        result,
        array![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0].into_dyn()
    );
}

#[test]
fn test_tensor_multiplication() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let t1 = Tensor::from_vec(vec![2.0f32; 10], vec![10], backend.clone());
    let t2 = Tensor::from_vec(vec![3.0f32; 10], vec![10], backend.clone());
    let t3 = &t1 * &t2;
    let result = t3.to_ndarray::<f32>();
    assert_eq!(
        result,
        array![6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0].into_dyn()
    );
}

#[test]
#[ignore]
fn test_tensor_reshape() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let t1 = Tensor::from_vec((0..12).map(|x| x as f32).collect(), vec![3, 4], backend);
    let t2 = t1.reshape(vec![4, 3]);
    let result = t2.to_ndarray::<f32>();
    let expected = array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 10., 11.]].into_dyn();
    assert_eq!(t2.0.borrow().tracker.shape(), &[4, 3]);
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_data_io() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

    // Create ndarray
    let arr1 = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
    let arr2 = array![[5.0f32, 6.0], [7.0, 8.0]].into_dyn();

    // from_ndarray
    let t1 = Tensor::from_ndarray(&arr1, backend.clone());
    let t2 = Tensor::from_ndarray(&arr2, backend.clone());

    // Check shape and dtype
    assert_eq!(t1.0.borrow().tracker.shape(), &[2, 2]);
    assert_eq!(t1.0.borrow().dtype, DType::F32);

    // Perform operation
    let t3 = &t1 + &t2;

    // to_ndarray
    let result_arr = t3.to_ndarray::<f32>();
    let expected_arr = array![[6.0f32, 8.0], [10.0, 12.0]].into_dyn();

    assert_eq!(result_arr, expected_arr);

    // to_vec
    let result_vec = t3.to_vec::<f32>();
    assert_eq!(result_vec, vec![6.0, 8.0, 10.0, 12.0]);
}