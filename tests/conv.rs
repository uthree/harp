use harp::{
    ast::DType,
    backend::c::{CBackend, CBuffer},
    backend::{Backend, Buffer},
    graph::Graph,
};

fn run_c_backend(graph: &Graph, inputs: Vec<CBuffer>) -> Vec<CBuffer> {
    let backend = CBackend::new();
    let result_map = backend.execute(&graph, inputs, vec![]);
    graph
        .outputs
        .borrow()
        .iter()
        .map(|node_id| result_map.get(node_id).unwrap().clone())
        .collect()
}

#[test]
fn test_conv1d() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 4.into(), 10.into()]);
    let w = graph.input(DType::F32, vec![2.into(), 4.into(), 3.into()]);
    let _ = x.conv1d(w, 3, 1, 1).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&vec![0.0; 40]);
    x_buf.shape = vec![1, 4, 10];
    let mut w_buf = CBuffer::from_slice::<f32>(&vec![0.0; 24]);
    w_buf.shape = vec![2, 4, 3];
    let outputs = run_c_backend(&graph, vec![x_buf, w_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[1, 2, 8]);
}

#[test]
fn test_conv2d() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 3.into(), 10.into(), 10.into()]);
    let w = graph.input(DType::F32, vec![2.into(), 3.into(), 3.into(), 3.into()]);
    let _ = x.conv2d(w, (3, 3), (1, 1), 1).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&vec![0.0; 300]);
    x_buf.shape = vec![1, 3, 10, 10];
    let mut w_buf = CBuffer::from_slice::<f32>(&vec![0.0; 54]);
    w_buf.shape = vec![2, 3, 3, 3];
    let outputs = run_c_backend(&graph, vec![x_buf, w_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[1, 2, 8, 8]);
}