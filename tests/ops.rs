use harp::ast::AstOp;
use harp::graph::op::GraphOp;
use harp::graph::ops::elementwise::ElementwiseOps;
use harp::{
    ast::DType,
    backend::c::CBuffer,
    backend::{Backend, c::CBackend},
    graph::Graph,
    graph::shape::expr::Expr,
};
use rstest::rstest;

fn run_c_backend(graph: &Graph, inputs: Vec<CBuffer>) -> Vec<CBuffer> {
    let mut backend = CBackend::new();
    backend.execute(&graph, inputs, vec![])
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
fn test_op_reshape_multiple() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);
    let a = x.reshape(vec![6.into(), 4.into()]);
    let _ = a.reshape(vec![2.into(), 12.into()]).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
        21., 22., 23., 24.,
    ]);
    x_buf.shape = vec![2, 3, 4];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[2, 12]);
    assert_eq!(
        y_buf.as_slice::<f32>(),
        &[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.
        ]
    );
}

#[test]
#[should_panic(expected = "reshape shape must have the same number of elements")]
fn test_op_reshape_invalid_numel() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into()]);
    // This should panic because 2 * 3 != 5
    let _ = x.reshape(vec![5.into()]).as_output();
}

#[test]
fn test_op_slice() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 4.into(), 5.into()]);
    let _ = x
        .slice(vec![
            (0.into(), 1.into()),
            (1.into(), 3.into()),
            (2.into(), 4.into()),
        ])
        .as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&(0..40).map(|x| x as f32).collect::<Vec<_>>());
    x_buf.shape = vec![2, 4, 5];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[1, 2, 2]);
    assert_eq!(y_buf.as_slice::<f32>(), &[7., 8., 12., 13.]);
}

#[test]
#[should_panic(expected = "Slice arguments must match number of dimensions")]
fn test_op_slice_invalid_args() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 4.into(), 5.into()]);
    // This should panic because the number of slice args is 2, but ndim is 3.
    let _ = x
        .slice(vec![(0.into(), 1.into()), (1.into(), 3.into())])
        .as_output();
}

#[test]
fn test_op_contiguous() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into()]);
    let _ = x.permute(vec![1, 0]).contiguous().as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[1., 2., 3., 4., 5., 6.]);
    x_buf.shape = vec![2, 3];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[3, 2]);
    // The data should now be contiguous in the new shape's layout.
    // Original x: [[1, 2, 3], [4, 5, 6]]
    // Permuted x: [[1, 4], [2, 5], [3, 6]]
    // Contiguous output should be the flattened version of the permuted data.
    assert_eq!(y_buf.as_slice::<f32>(), &[1., 4., 2., 5., 3., 6.]);
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

#[rstest]
#[case(
    "add",
    |g: &Graph, a, b| g.add(a, b),
    AstOp::Add
)]
#[case(
    "sub",
    |g: &Graph, a, b| g.sub(a, b),
    AstOp::Sub
)]
#[case(
    "mul",
    |g: &Graph, a, b| g.mul(a, b),
    AstOp::Mul
)]
#[case(
    "div",
    |g: &Graph, a, b| g.div(a, b),
    AstOp::Mul // div is implemented as mul(a, recip(b))
)]
fn test_elementwise_binary_ops(
    #[case] op_name: &str,
    #[case] op_func: impl Fn(
        &Graph,
        harp::graph::node::NodeId,
        harp::graph::node::NodeId,
    ) -> harp::graph::node::NodeId,
    #[case] expected_op: AstOp,
) {
    let graph = Graph::new();
    let shape: Vec<Expr> = vec![1.into(), 2.into(), 3.into()];
    let a_id = graph.input(DType::F32, shape.clone()).id;
    let b_id = graph.input(DType::F32, shape.clone()).id;

    let c_id = op_func(&graph, a_id, b_id);

    let nodes = graph.nodes.borrow();
    let c_node = &nodes[c_id.0];

    if op_name == "div" {
        // div creates two nodes: recip and mul
        assert_eq!(nodes.len(), 4);
        let recip_node = &nodes[c_node.src[1].0];
        assert_eq!(recip_node.op, GraphOp::Elementwise(AstOp::Recip));
    } else {
        assert_eq!(nodes.len(), 3);
    }

    // For div, the final operation is Mul. We need to check the inputs carefully.
    if op_name != "div" {
        assert_eq!(c_node.src, vec![a_id, b_id]);
    } else {
        assert_eq!(c_node.src[0], a_id);
    }

    assert_eq!(c_node.op, GraphOp::Elementwise(expected_op));
    assert_eq!(c_node.shape, shape);
}

#[test]
fn test_op_permute() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);
    let _ = x.permute(vec![1, 2, 0]).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
        21., 22., 23., 24.,
    ]);
    x_buf.shape = vec![2, 3, 4];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[3, 4, 2]);
    assert_eq!(
        y_buf.as_slice::<f32>(),
        &[
            1., 13., 2., 14., 3., 15., 4., 16., 5., 17., 6., 18., 7., 19., 8., 20., 9., 21., 10.,
            22., 11., 23., 12., 24.
        ]
    );
}

#[test]
#[should_panic(expected = "assertion failed: self.ndim() == axes.len()")]
fn test_op_permute_invalid_axes_len() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);
    let _ = x.permute(vec![1, 0]).as_output();
}

#[test]
#[should_panic(expected = "index out of bounds: the len is 3 but the index is 3")]
fn test_op_permute_invalid_axis() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);
    let _ = x.permute(vec![0, 1, 3]).as_output();
}

#[test]
#[should_panic(expected = "duplicate axis in permute")]
fn test_op_permute_duplicate_axis() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);
    // This should panic because axis 1 is duplicated.
    let _ = x.permute(vec![0, 1, 1]).as_output();
}

#[test]
fn test_op_squeeze_unsqueeze() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 2.into(), 1.into(), 3.into()]);
    let squeezed = x.squeeze(2);
    let _ = squeezed.unsqueeze(0).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[1., 2., 3., 4., 5., 6.]);
    x_buf.shape = vec![1, 2, 1, 3];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[1, 1, 2, 3]);
    assert_eq!(y_buf.as_slice::<f32>(), &[1., 2., 3., 4., 5., 6.]);
}

#[test]
fn test_op_expand() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 3.into(), 1.into()]);
    let _ = x.expand(vec![2.into(), 3.into(), 4.into()]).as_output();

    let mut x_buf = CBuffer::from_slice::<f32>(&[1., 2., 3.]);
    x_buf.shape = vec![1, 3, 1];
    let outputs = run_c_backend(&graph, vec![x_buf]);
    let y_buf = &outputs[0];

    assert_eq!(y_buf.shape, &[2, 3, 4]);
    assert_eq!(
        y_buf.as_slice::<f32>(),
        &[
            1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 1., 1., 1., 1., 2., 2., 2., 2., 3., 3.,
            3., 3.
        ]
    );
}

#[test]
#[should_panic(expected = "can only squeeze an axis of size 1")]
fn test_op_squeeze_invalid() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 2.into(), 3.into()]);
    let _ = x.squeeze(1).as_output();
}

#[test]
#[should_panic(expected = "can only expand a dimension of size 1")]
fn test_op_expand_invalid() {
    let graph = Graph::new();
    let x = graph.input(DType::F32, vec![1.into(), 2.into(), 3.into()]);
    let _ = x.expand(vec![2.into(), 3.into(), 4.into()]).as_output();
}
