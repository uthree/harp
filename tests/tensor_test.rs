use harp::optimization::pattern::{TPat, TPatRule, TensorPatternMatcher};
use harp::prelude::*;
use harp::uop::Op;
use ndarray::{ArrayD, Zip, arr2, array};
use std::rc::Rc;

#[test]
fn test_tensor_addition() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = backend("clang");
    let shape = vec![10];

    // Create two 'leaf' tensors from loaded data
    let t1: Tensor = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        DType::F32,
        backend.clone(),
    );
    let t2: Tensor = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        DType::F32,
        backend.clone(),
    );

    // Perform addition
    let t3 = &t1 + &t2;

    // Check the resulting tensor's properties
    assert!(matches!(t3.op, TensorOp::Binary(harp::uop::Op::Add)));
    assert_eq!(t3.src.len(), 2);
    assert!(Rc::ptr_eq(&t3.src[0].0, &t1.0));
    assert!(Rc::ptr_eq(&t3.src[1].0, &t2.0));
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
    let tensor_a: Tensor = arr_a.clone().into();
    let tensor_b: Tensor = arr_b.clone().into();
    let result_sub = &tensor_a - &tensor_b;
    let expected_sub = &arr_a - &arr_b;
    assert_eq!(ArrayD::<f32>::from(result_sub), expected_sub);

    // Division
    let tensor_a: Tensor = arr_a.clone().into();
    let tensor_b: Tensor = arr_b.clone().into();
    let result_div = &tensor_a / &tensor_b;
    let expected_div = &arr_a / &arr_b;
    assert_eq!(ArrayD::<f32>::from(result_div), expected_div);
}

#[test]
fn test_tensor_unary_ops() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr: ArrayD<f32> = arr2(&[[1.0, -2.0], [3.0, 4.0]]).into_dyn();

    // Negation
    let tensor_a: Tensor = arr.clone().into();
    let result_neg = -&tensor_a;
    let expected_neg = -arr.clone();
    assert_eq!(ArrayD::<f32>::from(result_neg), expected_neg);

    // Sqrt
    let arr_sqrt = arr.mapv(|a| a.abs()); // sqrt requires non-negative values
    let tensor_sqrt: Tensor = arr_sqrt.clone().into();
    let result_sqrt = tensor_sqrt.sqrt();
    let expected_sqrt = arr_sqrt.mapv(f32::sqrt);
    assert_eq!(ArrayD::<f32>::from(result_sqrt), expected_sqrt);

    // Exp2
    let tensor_exp: Tensor = arr.clone().into();
    let result_exp = tensor_exp.exp2();
    let expected_exp = arr.mapv(f32::exp2);
    assert_eq!(ArrayD::<f32>::from(result_exp), expected_exp);

    // Log2
    let arr_log = arr.mapv(|a| a.abs() + 1.0); // log requires positive values
    let tensor_log: Tensor = arr_log.clone().into();
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

    let t1: Tensor = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(original_shape.clone()),
        DType::F32,
        backend.clone(),
    );

    let t2 = t1.reshape(new_shape.clone());

    assert_eq!(t2.shape(), &new_shape);
    // Check that the underlying operation and sources are unchanged
    assert!(matches!(t2.op, TensorOp::Load));
    assert!(t2.src.is_empty());

    // Realizing the reshaped tensor should work
    let _result_variable = t2.realize();
    println!("Reshape test completed successfully!");
}

#[test]
fn test_tensor_sum() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
    let tensor_a: Tensor = arr_a.clone().into();

    // Sum along axis 0
    let result_sum = tensor_a.sum(0);
    let result_arr: ArrayD<f32> = result_sum.into();

    // Expected result from ndarray
    let expected_arr = arr_a.sum_axis(ndarray::Axis(0));

    let expected_arr_reshaped = expected_arr.into_shape(vec![2]).unwrap();

    assert_eq!(result_arr, expected_arr_reshaped);
}

#[test]
fn test_tensor_sum_axis_1() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let tensor_a: Tensor = arr_a.clone().into();

    // Sum along axis 1
    let result_sum = tensor_a.sum(1);
    let result_arr: ArrayD<f32> = result_sum.into();

    // Expected result from ndarray
    let expected_arr = arr_a.sum_axis(ndarray::Axis(1));
    let expected_arr_reshaped = expected_arr.into_shape(vec![2]).unwrap();

    assert_eq!(result_arr, expected_arr_reshaped);
}

#[test]
fn test_tensor_sum_3d() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = array![[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]].into_dyn();
    let tensor_a: Tensor = arr_a.clone().into();

    // Sum along axis 1
    let result_sum = tensor_a.sum(1);
    let result_arr: ArrayD<f32> = result_sum.into();

    // Expected result from ndarray
    let expected_arr = arr_a.sum_axis(ndarray::Axis(1));
    let expected_arr_reshaped = expected_arr.into_shape(vec![2, 2]).unwrap();

    assert_eq!(result_arr, expected_arr_reshaped);
}

#[test]
fn test_tensor_sum_to_scalar() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = array![1.0, 2.0, 3.0, 4.0].into_dyn();
    let tensor_a: Tensor = arr_a.clone().into();

    // Sum all elements to get a scalar result
    let result_sum = tensor_a.sum(0);
    assert_eq!(result_sum.shape(), &[] as &[usize]);

    let result_arr: ArrayD<f32> = result_sum.into();
    let expected_arr = arr_a.sum_axis(ndarray::Axis(0));

    // The result from ndarray is a 0-dimensional array, which is what we want.
    assert_eq!(result_arr, expected_arr);
    assert_eq!(result_arr.first().unwrap(), &10.0f32);
}

#[test]
fn test_tensor_zeros() {
    let _ = env_logger::builder().is_test(true).try_init();
    let shape = vec![2, 3];
    let tensor = Tensor::zeros(shape.clone(), DType::F32);
    let result: ArrayD<f32> = tensor.into();
    let expected = ArrayD::zeros(shape);
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_ones() {
    let _ = env_logger::builder().is_test(true).try_init();
    let shape = vec![3, 2];
    let tensor = Tensor::ones(shape.clone(), DType::F32);
    let result: ArrayD<f32> = tensor.into();
    let expected = ArrayD::ones(shape);
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_full() {
    let _ = env_logger::builder().is_test(true).try_init();
    let shape = vec![2, 3];
    let fill_value = 7.0f32;
    let tensor = Tensor::full(shape.clone(), fill_value);
    let result: ArrayD<f32> = tensor.into();
    let expected = ArrayD::from_elem(shape, fill_value);
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_max() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arr_a: ArrayD<f32> = array![[1.0, 5.0], [6.0, 2.0]].into_dyn();
    let arr_b: ArrayD<f32> = array![[4.0, 3.0], [6.0, 7.0]].into_dyn();
    let tensor_a: Tensor = arr_a.clone().into();
    let tensor_b: Tensor = arr_b.clone().into();

    let result_tensor = tensor_a.max(&tensor_b);
    let result_arr: ArrayD<f32> = result_tensor.into();

    let mut expected_arr = ArrayD::zeros(arr_a.shape());
    Zip::from(&mut expected_arr)
        .and(&arr_a)
        .and(&arr_b)
        .for_each(|expected, &a, &b| *expected = f32::max(a, b));
    assert_eq!(result_arr, expected_arr);
}

#[test]
fn test_tensor_macros() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Test float_tensor!
    let ft = float_tensor![[1, 2], [3, 4]];
    assert_eq!(ft.shape(), &[2, 2]);
    assert_eq!(ft.dtype, DType::F32);
    let ft_arr: ArrayD<f32> = ft.into();
    assert_eq!(ft_arr, array![[1.0, 2.0], [3.0, 4.0]].into_dyn());

    // Test long_tensor!
    let lt = long_tensor![[10, 20], [30, 40]];
    assert_eq!(lt.shape(), &[2, 2]);
    assert_eq!(lt.dtype, DType::I64);
    let lt_arr: ArrayD<i64> = lt.into();
    assert_eq!(lt_arr, array![[10i64, 20i64], [30i64, 40i64]].into_dyn());

    // Test double_tensor!
    let dt = double_tensor![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(dt.shape(), &[2, 2]);
    assert_eq!(dt.dtype, DType::F64);
    let dt_arr: ArrayD<f64> = dt.into();
    assert_eq!(
        dt_arr,
        array![[1.0f64, 2.0f64], [3.0f64, 4.0f64]].into_dyn()
    );

    // Test int_tensor!
    let it = int_tensor![[100, 200], [300, 400]];
    assert_eq!(it.shape(), &[2, 2]);
    assert_eq!(it.dtype, DType::I32);
    let it_arr: ArrayD<i32> = it.into();
    assert_eq!(
        it_arr,
        array![[100i32, 200i32], [300i32, 400i32]].into_dyn()
    );
}

#[test]
fn test_tensor_optimization_double_neg() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Rule: -(-x) => x
    let rule = TPatRule::new(
        "double_neg",
        TPat::Unary(
            Op::Neg,
            Box::new(TPat::Unary(Op::Neg, Box::new(TPat::Capture(0)))),
        ),
        |_| true, // No special condition needed
        |captures| {
            let x = captures.get(&0).unwrap();
            Some(x.clone())
        },
    );
    let matcher = TensorPatternMatcher::new(vec![rule]);

    let a = Tensor::full(vec![2, 2], 5.0f32);
    let expr = -(-a.clone());

    // Check that the structure is as expected before optimization
    assert!(matches!(expr.op, TensorOp::Unary(Op::Neg)));
    assert!(matches!(expr.src[0].op, TensorOp::Unary(Op::Neg)));

    let optimized_expr = expr.optimize(&matcher);

    // The optimized expression should be pointer-equal to the original tensor `a`
    assert!(Rc::ptr_eq(&a.0, &optimized_expr.0));
}

#[test]
fn test_tensor_optimization_mul_one() {
    use harp::dtype::Number;

    let _ = env_logger::builder().is_test(true).try_init();

    // Rule: x * 1.0 => x
    let rule = TPatRule::new(
        "mul_one",
        TPat::Binary(
            Op::Mul,
            Box::new(TPat::Capture(0)),
            Box::new(TPat::Capture(1)),
        ),
        |captures| {
            // Condition: check if the second captured tensor is a constant 1.0
            if let Some(c_tensor) = captures.get(&1) {
                if let TensorOp::Constant(Number::F32(val)) = c_tensor.op {
                    return val == 1.0;
                }
            }
            false
        },
        |captures| {
            // Replacer: return the first captured tensor
            captures.get(&0).cloned()
        },
    );
    let matcher = TensorPatternMatcher::new(vec![rule]);

    let a = Tensor::full(vec![2, 2], 5.0f32);
    let one = Tensor::full(vec![2, 2], 1.0f32);
    let expr = &a * &one;

    let optimized_expr = expr.optimize(&matcher);

    // The optimized expression should be pointer-equal to `a`
    assert!(Rc::ptr_eq(&a.0, &optimized_expr.0));

    // Test with a reshaped tensor to ensure it still works
    let one_reshaped = Tensor::full(vec![4], 1.0f32).reshape(vec![2, 2]);
    let expr_reshaped = &a * &one_reshaped;
    let optimized_expr_reshaped = expr_reshaped.optimize(&matcher);
    assert!(Rc::ptr_eq(&a.0, &optimized_expr_reshaped.0));
}
