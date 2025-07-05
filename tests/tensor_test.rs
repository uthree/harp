use harp::graph::Graph;
use harp::ops::Input;
use harp::shape::symbolic::Expr;
use harp::tensor::Tensor;

#[test]
fn test_tensor_downgrade_upgrade() {
    let mut graph = Graph::new();
    let shape = vec![Expr::from(10), Expr::from(20)];
    let tensor = graph.input(shape);

    let tensor_ref = tensor.downgrade();
    let upgraded_tensor = tensor_ref.upgrade().unwrap();

    assert_eq!(
        tensor.data.borrow().shape_tracker,
        upgraded_tensor.data.borrow().shape_tracker
    );
}

#[test]
fn test_tensor_clone() {
    let mut graph = Graph::new();
    let shape = vec![Expr::from(10), Expr::from(20)];
    let tensor = graph.input(shape);
    let tensor_clone = tensor.clone();

    assert_eq!(
        tensor.data.borrow().shape_tracker,
        tensor_clone.data.borrow().shape_tracker
    );
    // Ensure they are different objects
    assert!(!std::ptr::eq(&tensor, &tensor_clone));
}
