use harp::backends::{Backend, GccBackend};
use harp::dot::ToDot;
use harp::dtype::DType;
use harp::shapetracker::ShapeTracker;
use harp::tensor::{Tensor, TensorOp};
use std::rc::Rc;
use std::sync::Arc;

#[test]
fn test_tensor_addition() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(GccBackend::new());
    let shape = vec![10];

    // Create two 'leaf' tensors from loaded data
    let t1 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        DType::F32,
        backend.clone(),
    );
    let t2 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        DType::F32,
        backend.clone(),
    );

    // Perform addition
    let t3 = &t1 + &t2;

    // Check the resulting tensor's properties
    assert!(matches!(t3.0.op, TensorOp::Binary(harp::uop::Op::Add)));
    assert_eq!(t3.0.src.len(), 2);
    assert!(Rc::ptr_eq(&t3.0.src[0].0, &t1.0));
    assert!(Rc::ptr_eq(&t3.0.src[1].0, &t2.0));
    assert_eq!(t3.0.tracker.shape(), &shape);

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
fn test_tensor_multiplication() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(GccBackend::new());
    let shape = vec![10];

    let t1 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        DType::F32,
        backend.clone(),
    );
    let t2 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(shape.clone()),
        DType::F32,
        backend.clone(),
    );

    let t3 = &t1 * &t2;

    assert!(matches!(t3.0.op, TensorOp::Binary(harp::uop::Op::Mul)));
    assert_eq!(t3.0.src.len(), 2);
    assert!(Rc::ptr_eq(&t3.0.src[0].0, &t1.0));
    assert!(Rc::ptr_eq(&t3.0.src[1].0, &t2.0));
    assert_eq!(t3.0.tracker.shape(), &shape);

    let _result_variable = t3.realize();

    let dot_string = t3.to_dot();
    println!("\n--- Tensor DOT --- \n{}", dot_string);
    assert!(dot_string.contains("[label=\"op: Binary(Mul)\\nshape: [10]\\ndtype: F32\"]"));
}

#[test]
fn test_tensor_reshape() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(GccBackend::new());
    let original_shape = vec![10, 20];
    let new_shape = vec![200];

    let t1 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(original_shape.clone()),
        DType::F32,
        backend.clone(),
    );

    let t2 = t1.reshape(new_shape.clone());

    assert_eq!(t2.0.tracker.shape(), &new_shape);
    // Check that the underlying operation and sources are unchanged
    assert!(matches!(t2.0.op, TensorOp::Load));
    assert!(t2.0.src.is_empty());

    // Realizing the reshaped tensor should work
    let _result_variable = t2.realize();
    println!("Reshape test completed successfully!");
}
