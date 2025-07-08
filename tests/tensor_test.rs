use harp::{
    graph::Graph,
    shape::tracker::ShapeTracker,
    tensor::{Tensor, TensorData},
    dtype::DType,
};
use ndarray::array;
use std::sync::{Arc, Mutex};

#[test]
fn test_tensor_data_creation() {
    let data = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let tensor_data = TensorData { data: data.clone(), dtype: DType::F32 };
    assert_eq!(tensor_data.data, data);
    assert_eq!(tensor_data.dtype, DType::F32);
}

#[test]
fn test_tensor_exp2() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let input_shape: ShapeTracker = input_data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.exp2();

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    assert_eq!(result_tensor.shape, input_shape);
    assert_eq!(result_tensor.dtype, DType::F32);
}

#[test]
fn test_tensor_log2() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = array![[2.0, 4.0], [8.0, 16.0]].into_dyn();
    let input_shape: ShapeTracker = input_data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.log2();

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    assert_eq!(result_tensor.shape, input_shape);
    assert_eq!(result_tensor.dtype, DType::F32);
}

#[test]
fn test_tensor_sin() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = array![[0.0, std::f32::consts::PI / 2.0]].into_dyn();
    let input_shape: ShapeTracker = input_data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.sin();

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    assert_eq!(result_tensor.shape, input_shape);
    assert_eq!(result_tensor.dtype, DType::F32);
}

#[test]
fn test_tensor_sqrt() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = array![[1.0, 4.0]].into_dyn();
    let input_shape: ShapeTracker = input_data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.sqrt();

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    assert_eq!(result_tensor.shape, input_shape);
    assert_eq!(result_tensor.dtype, DType::F32);
}

#[test]
fn test_tensor_recip() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = array![[1.0, 2.0]].into_dyn();
    let input_shape: ShapeTracker = input_data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.recip();

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    assert_eq!(result_tensor.shape, input_shape);
    assert_eq!(result_tensor.dtype, DType::F32);
}

#[test]
fn test_tensor_cast() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_shape: ShapeTracker = vec![2, 2].into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.cast(DType::I32);

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    assert_eq!(result_tensor.shape, input_shape);
    assert_eq!(result_tensor.dtype, DType::I32);
}

#[test]
fn test_tensor_sum_reduce() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let input_shape: ShapeTracker = input_data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.sum_reduce(0);

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    // Expected shape after sum_reduce(0) on [2, 2] is [2]
    assert_eq!(result_tensor.shape.max, ShapeTracker::from(vec![2usize]).max);
    assert_eq!(result_tensor.dtype, DType::F32);
}

#[test]
fn test_tensor_max_reduce() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let input_shape: ShapeTracker = input_data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), input_shape.clone(), DType::F32);

    let result_tensor = input_tensor.max_reduce(1);

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);
    // Expected shape after max_reduce(1) on [2, 2] is [2]
    assert_eq!(result_tensor.shape.max, ShapeTracker::from(vec![2usize]).max);
    assert_eq!(result_tensor.dtype, DType::F32);
}

#[test]
fn test_tensor_add() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 2].into();
    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);

    let c = &a + &b;

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 3);
    assert_eq!(c.shape, shape);
    assert_eq!(c.dtype, DType::F32);
}

#[test]
fn test_tensor_mul() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 2].into();
    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);

    let c = &a * &b;

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 3);
    assert_eq!(c.shape, shape);
    assert_eq!(c.dtype, DType::F32);
}

#[test]
fn test_tensor_rem() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 2].into();
    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);

    let c = &a % &b;

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 3);
    assert_eq!(c.shape, shape);
    assert_eq!(c.dtype, DType::F32);
}

#[test]
fn test_tensor_less_than() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 2].into();
    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);

    let c = a.less_than(&b);

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 3);
    assert_eq!(c.shape, shape);
    assert_eq!(c.dtype, DType::F32);
}