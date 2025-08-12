use harp::{
    ast::DType,
    backend::Backend,
    backend::c::{CBackend, CBuffer},
    graph::Graph,
};

fn run_c_backend(graph: &Graph, inputs: Vec<CBuffer>) -> Vec<CBuffer> {
    let backend = CBackend::with_config(Default::default());
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

    let x_shape = vec![1, 4, 10];
    let w_shape = vec![2, 4, 3];
    let x_buf = CBuffer::allocate(DType::F32, x_shape);
    let w_buf = CBuffer::allocate(DType::F32, w_shape);
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

    let x_shape = vec![1, 3, 10, 10];
    let w_shape = vec![2, 3, 3, 3];
    let x_buf = CBuffer::allocate(DType::F32, x_shape);
    let w_buf = CBuffer::allocate(DType::F32, w_shape);
    let outputs = run_c_backend(&graph, vec![x_buf, w_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[1, 2, 8, 8]);
}
