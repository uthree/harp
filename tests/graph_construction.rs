use harp::{
    dtype::{DType, Scalar},
    graph::Graph,
    shape::tracker::ShapeTracker,
};
use std::sync::{Arc, Mutex};

#[test]
fn test_simple_graph_construction() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 3].into();

    let a = Graph::new_input(graph.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph.clone(), shape.clone(), DType::F32);

    let c = &a + &b;
    let d = c.exp2();

    Graph::add_output_node(graph.clone(), &d);

    let graph_locked = graph.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 4);
    assert_eq!(graph_locked.edge_count(), 3);
    assert_eq!(graph_locked.inputs.len(), 2);
    assert_eq!(graph_locked.outputs.len(), 1);
}

#[test]
fn test_to_dot_output() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 3].into();

    let a = Graph::new_input(graph.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph.clone(), shape.clone(), DType::F32);

    let c = &a + &b;
    let d = c.exp2();

    Graph::add_output_node(graph.clone(), &d);

    let graph_locked = graph.lock().unwrap();
    let dot_output = graph_locked.to_dot();

    // Basic checks for DOT format
    assert!(dot_output.starts_with("digraph {"));
    assert!(dot_output.ends_with("}"));
    assert!(dot_output.contains("Input { dtype: F32 }\nshape=[2, 3], map=(idx0*3) + idx1\nF32"));
    assert!(dot_output.contains("Add\nshape=[2, 3], map=(idx0*3) + idx1\nF32"));
    assert!(dot_output.contains("Exp2\nshape=[2, 3], map=(idx0*3) + idx1\nF32"));
    assert!(dot_output.contains("->"));
}

#[test]
fn test_ones_and_zeros_construction() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![10, 10].into();

    let ones_tensor = Graph::ones(graph.clone(), shape.clone(), DType::F32);
    let zeros_tensor = Graph::zeros(graph.clone(), shape.clone(), DType::F32);

    let graph_locked = graph.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);

    let ones_node = &graph_locked.graph[ones_tensor.node_index];
    let zeros_node = &graph_locked.graph[zeros_tensor.node_index];

    let ones_op = ones_node
        .op()
        .as_any()
        .downcast_ref::<harp::operator::Const>()
        .unwrap();
    let zeros_op = zeros_node
        .op()
        .as_any()
        .downcast_ref::<harp::operator::Const>()
        .unwrap();

    assert_eq!(ones_op.scalar, Scalar::F32(1.0));
    assert_eq!(zeros_op.scalar, Scalar::F32(0.0));
}

#[test]
fn test_rand_nodes_construction() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![10, 10].into();

    let randu_tensor = Graph::randu(graph.clone(), shape.clone(), DType::F32);
    let randn_tensor = Graph::randn(graph.clone(), shape.clone(), DType::F32);

    let graph_locked = graph.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2);

    let randu_node = &graph_locked.graph[randu_tensor.node_index];
    let randn_node = &graph_locked.graph[randn_tensor.node_index];

    let randu_op = randu_node
        .op()
        .as_any()
        .downcast_ref::<harp::operator::RandU>()
        .unwrap();
    let randn_op = randn_node
        .op()
        .as_any()
        .downcast_ref::<harp::operator::RandN>()
        .unwrap();

    assert_eq!(randu_op.dtype, DType::F32);
    assert_eq!(randn_op.dtype, DType::F32);
}
