use harp::{
    dtype,
    graph::Graph,
    shape::tracker::ShapeTracker,
};
use std::sync::{Arc, Mutex};

#[test]
fn test_simple_graph_construction() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 3].into();

    let a = Graph::new_input(graph.clone(), shape.clone(), dtype::F32_DTYPE);
    let b = Graph::new_input(graph.clone(), shape.clone(), dtype::F32_DTYPE);

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

    let a = Graph::new_input(graph.clone(), shape.clone(), dtype::F32_DTYPE);
    let b = Graph::new_input(graph.clone(), shape.clone(), dtype::F32_DTYPE);

    let c = &a + &b;
    let d = c.exp2();

    Graph::add_output_node(graph.clone(), &d);

    let graph_locked = graph.lock().unwrap();
    let dot_output = graph_locked.to_dot();

    // Basic checks for DOT format
    assert!(dot_output.starts_with("digraph {"));
    assert!(dot_output.ends_with("}"));
    assert!(dot_output.contains("Input { dtype: F32 }\nshape=[2, 3], map=(idx * 3) + idx\nF32"));
    assert!(dot_output.contains("Add\nshape=[2, 3], map=(idx * 3) + idx\nF32"));
    assert!(dot_output.contains("Exp2\nshape=[2, 3], map=(idx * 3) + idx\nF32"));
    assert!(dot_output.contains("->"));
}
