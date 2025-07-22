use harp::backends::{Backend, CpuBackend};
use harp::dtype::DType;
use harp::tensor::{Tensor, TensorOp};
use std::rc::Rc;
use std::sync::Arc;

#[test]
fn test_tensor_addition() {
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

    // Create two 'leaf' tensors from loaded data
    let t1 = Tensor::new(
        TensorOp::Load,
        vec![],
        vec![10],
        DType::F32,
        backend.clone(),
    );
    let t2 = Tensor::new(
        TensorOp::Load,
        vec![],
        vec![10],
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
    assert_eq!(t3.0.shape, vec![10]);

    // Realize the result
    // This will trigger the (currently mocked) compilation and execution
    let _result_variable = t3.realize();

    println!("Tensor addition test completed successfully!");
}
