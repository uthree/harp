use harp::{
    ast::DType,
    backend::{Backend, Buffer, c::CBackend},
    backend::c::CBuffer,
    graph::Graph,
};

fn run_c_backend(graph: &Graph, inputs: Vec<CBuffer>) -> Vec<CBuffer> {
    let mut backend = CBackend::new();
    let inputs: Vec<Box<dyn Buffer>> = inputs
        .into_iter()
        .map(|b| Box::new(b) as Box<dyn Buffer>)
        .collect();
    let outputs = backend.call(graph.clone(), inputs, vec![]);
    outputs
        .into_iter()
        .map(|b| {
            b.as_any()
                .downcast_ref::<CBuffer>()
                .unwrap()
                .clone()
        })
        .collect()
}

#[test]
fn test_op_reshape() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![6.into()]);
    let _ = x.reshape(vec![2.into(), 3.into()]).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[1., 2., 3., 4., 5., 6.]);
    x_buf.shape = vec![6];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[2, 3]);
    assert_eq!(y_buf.as_slice::<f32>(), &[1., 2., 3., 4., 5., 6.]);
}

#[test]
fn test_op_sum() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into()]);
    let _ = x.sum(1).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[1., 2., 3., 4., 5., 6.]);
    x_buf.shape = vec![2, 3];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[2]);
    assert_eq!(y_buf.as_slice::<f32>(), &[6., 15.]);
}
