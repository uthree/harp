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
use serial_test::serial;

fn run_c_backend(graph: &Graph, inputs: Vec<CBuffer>) -> Vec<CBuffer> {
    let mut backend = CBackend::new();
    backend.execute(&graph, inputs, vec![])
}

#[test]
#[serial]
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
#[serial]
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
#[serial]
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
