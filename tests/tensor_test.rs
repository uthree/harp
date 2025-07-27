use harp::prelude::*;
use ndarray::{ArrayD, arr2};
use std::rc::Rc;

#[test]
fn test_tensor_addition() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = backend("clang");
    let shape = vec![10];

    // Create two 'leaf' tensors from loaded data
    let t1: Tensor<f32> = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        f32::into_dtype(),
        backend.clone(),
    );
    let t2: Tensor<f32> = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        f32::into_dtype(),
        backend.clone(),
    );

    // Perform addition
    let t3 = &t1 + &t2;

    // Check the resulting tensor's properties
    assert!(matches!(t3.0.op, TensorOp::Binary(harp::uop::Op::Add)));
    assert_eq!(t3.0.src.len(), 2);
    assert!(Rc::ptr_eq(&t3.0.src[0].0, &t1.0));
    assert!(Rc::ptr_eq(&t3.0.src[1].0, &t2.0));
    assert_eq!(t3.shape(), &shape);

    // Realize the result
    // This will trigger the compilation and execution
    let _result_variable = t3.realize();

    println!("Tensor addition test completed successfully!");

    let dot_string = t3.to_dot();
    println!("\n--- Tensor DOT --- \n{}", dot_string);
    assert!(dot_string.starts_with("digraph G"));
    assert!(dot_string.contains("[label=\"op: Load\\nshape: [10]\\ndtype: F32\"]"));
    assert!(dot_string.contains("[label=\"op: Binary(Add)\\nshape: [10]\\ndtype: F32\"]"));
}

#[test]
fn test_tensor_binary_ops() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = arr2(&[[10.0, 20.0], [30.0, 40.0]]).into_dyn();
    let arr_b: ArrayD<f32> = arr2(&[[5.0, 2.0], [3.0, 8.0]]).into_dyn();

    // Subtraction
    let tensor_a: Tensor<f32> = arr_a.clone().into();
    let tensor_b: Tensor<f32> = arr_b.clone().into();
    let result_sub = &tensor_a - &tensor_b;
    let expected_sub = &arr_a - &arr_b;
    assert_eq!(ArrayD::<f32>::from(result_sub), expected_sub);

    // Division
    let tensor_a: Tensor<f32> = arr_a.clone().into();
    let tensor_b: Tensor<f32> = arr_b.clone().into();
    let result_div = &tensor_a / &tensor_b;
    let expected_div = &arr_a / &arr_b;
    assert_eq!(ArrayD::<f32>::from(result_div), expected_div);
}

#[test]
fn test_tensor_unary_ops() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr: ArrayD<f32> = arr2(&[[1.0, -2.0], [3.0, 4.0]]).into_dyn();

    // Negation
    let tensor_a: Tensor<f32> = arr.clone().into();
    let result_neg = -&tensor_a;
    let expected_neg = -arr.clone();
    assert_eq!(ArrayD::<f32>::from(result_neg), expected_neg);

    // Sqrt
    let arr_sqrt = arr.mapv(|a| a.abs()); // sqrt requires non-negative values
    let tensor_sqrt: Tensor<f32> = arr_sqrt.clone().into();
    let result_sqrt = tensor_sqrt.sqrt();
    let expected_sqrt = arr_sqrt.mapv(f32::sqrt);
    assert_eq!(ArrayD::<f32>::from(result_sqrt), expected_sqrt);

    // Exp2
    let tensor_exp: Tensor<f32> = arr.clone().into();
    let result_exp = tensor_exp.exp2();
    let expected_exp = arr.mapv(f32::exp2);
    assert_eq!(ArrayD::<f32>::from(result_exp), expected_exp);

    // Log2
    let arr_log = arr.mapv(|a| a.abs() + 1.0); // log requires positive values
    let tensor_log: Tensor<f32> = arr_log.clone().into();
    let result_log = tensor_log.log2();
    let expected_log = arr_log.mapv(f32::log2);
    assert_eq!(ArrayD::<f32>::from(result_log), expected_log);
}

#[test]
fn test_tensor_reshape() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = backend("clang");
    let original_shape = vec![10, 20];
    let new_shape = vec![200];

    let t1: Tensor<f32> = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(original_shape.clone()),
        f32::into_dtype(),
        backend.clone(),
    );

    let t2 = t1.reshape(new_shape.clone());

    assert_eq!(t2.shape(), &new_shape);
    // Check that the underlying operation and sources are unchanged
    assert!(matches!(t2.0.op, TensorOp::Load));
    assert!(t2.0.src.is_empty());

    // Realizing the reshaped tensor should work
    let _result_variable = t2.realize();
    println!("Reshape test completed successfully!");
}

#[test]
fn test_tensor_sum() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
    let tensor_a: Tensor<f32> = arr_a.clone().into();

    // Sum along axis 0
    let result_sum = tensor_a.sum(0);
    let result_arr: ArrayD<f32> = result_sum.into();

    // Expected result from ndarray
    let expected_arr = arr_a.sum_axis(ndarray::Axis(0));

    let expected_arr_reshaped = expected_arr.into_shape(vec![2]).unwrap();

    assert_eq!(result_arr, expected_arr_reshaped);
}
