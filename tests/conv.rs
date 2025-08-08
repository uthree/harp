use harp::{
    ast::DType,
    backend::{Backend, c::CBackend},
    cbuffer::CBuffer,
    graph::Graph,
};

fn run_c_backend(graph: &Graph, inputs: Vec<CBuffer>) -> CBuffer {
    let mut backend = CBackend::new();
    let outputs = backend.call(graph.clone(), inputs, vec![]);
    outputs[0].clone()
}

#[test]
fn test_conv1d() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 1.into(), 10.into()]);
    let w = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into()]);
    let _ = x.conv1d(w, 3, 1, 1).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    x_buf.shape = vec![1, 1, 10];
    let mut w_buf = CBuffer::from_slice::<f32>(&[1., 1., 1.]);
    w_buf.shape = vec![1, 1, 3];

    let y_buf = run_c_backend(&graph, vec![x_buf, w_buf]);

    let expected = &[6., 9., 12., 15., 18., 21., 24., 27.];
    assert_eq!(y_buf.as_slice::<f32>(), expected);
}

#[test]
fn test_conv2d() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 1.into(), 4.into(), 4.into()]);
    let w = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into(), 3.into()]);
    let _ = x.conv2d(w, (3, 3), (1, 1), 1).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ]);
    x_buf.shape = vec![1, 1, 4, 4];
    let mut w_buf = CBuffer::from_slice::<f32>(&[1., 1., 1., 1., 1., 1., 1., 1., 1.]);
    w_buf.shape = vec![1, 1, 3, 3];

    let y_buf = run_c_backend(&graph, vec![x_buf, w_buf]);

    let expected = &[54., 63., 90., 99.];
    assert_eq!(y_buf.as_slice::<f32>(), expected);
}
