use harp::{
    interpreter::Interpreter,
    node::Node,
    operator::{self, Add, Const, Exp2, Input, Mul},
    optimizer::{ConstantFolding, EliminateUnusedNodes, GraphOptimizer},
    prelude::*,
    tensor::TensorData,
};
use ndarray::ArrayD;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[test]
fn test_interpreter_simple_add() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into()]);

    let a_data = TensorData(ArrayD::from_elem(vec![1], 2.0));
    let b_data = TensorData(ArrayD::from_elem(vec![1], 3.0));

    let a_node = Node::new(
        operator::Const {
            data: a_data.clone(),
        },
        shape.clone(),
    );
    let b_node = Node::new(
        operator::Const {
            data: b_data.clone(),
        },
        shape.clone(),
    );

    let mut g = graph.lock().unwrap();
    let a_idx = g.add_node(a_node);
    let b_idx = g.add_node(b_node);

    let add_node = Node::new(operator::Add, shape.clone());
    let add_idx = g.add_node(add_node);
    g.add_edge(a_idx, add_idx, 0);
    g.add_edge(b_idx, add_idx, 1);

    let mut interpreter = Interpreter::new();
    let inputs = HashMap::new(); // No external inputs for this test
    let result = interpreter.evaluate(add_idx, &g.graph, &inputs).unwrap();

    assert_eq!(result.0[[0]], 5.0);
}

#[test]
fn test_eliminate_unused_nodes() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into()]);

    let a = Graph::new_input(graph.clone(), shape.clone());
    let b = Graph::new_input(graph.clone(), shape.clone());
    let c = &a + &b; // Used
    let d = &a * &b; // Unused

    // Mark c as an output
    graph.lock().unwrap().add_output(&c);

    let initial_node_count = graph.lock().unwrap().node_count();
    assert_eq!(initial_node_count, 4); // a, b, c, d

    let mut optimizer = EliminateUnusedNodes {};
    optimizer.optimize(&mut graph.lock().unwrap());

    let optimized_node_count = graph.lock().unwrap().node_count();
    // a, b, c should remain. d should be removed.
    assert_eq!(optimized_node_count, 3);
}

#[test]
fn test_constant_folding() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into()]);

    let a_data = TensorData(ArrayD::from_elem(vec![1], 2.0));
    let b_data = TensorData(ArrayD::from_elem(vec![1], 3.0));

    let a_node = Node::new(
        operator::Const {
            data: a_data.clone(),
        },
        shape.clone(),
    );
    let b_node = Node::new(
        operator::Const {
            data: b_data.clone(),
        },
        shape.clone(),
    );

    let mut g = graph.lock().unwrap();
    let a_idx = g.add_node(a_node);
    let b_idx = g.add_node(b_node);

    let add_node = Node::new(operator::Add, shape.clone());
    let add_idx = g.add_node(add_node);
    g.add_edge(a_idx, add_idx, 0);
    g.add_edge(b_idx, add_idx, 1);

    let initial_node_count = g.node_count();
    assert_eq!(initial_node_count, 3); // a, b, add

    drop(g); // Release lock

    let mut optimizer = ConstantFolding {};
    optimizer.optimize(&mut graph.lock().unwrap());

    let g = graph.lock().unwrap();
    // After constant folding, the add_idx node should be replaced by a new Const node.
    // The count might remain the same if we just replace, or decrease if we also remove old consts.
    // For now, we expect a new Const node to be added, and the old nodes might still exist.
    // This test needs refinement based on the exact replacement strategy.
    // For simplicity, let's just check if a new Const node with the correct value exists.

    let mut found_folded_const = false;
    for node_idx in g.graph.node_indices() {
        let node = g.node_weight(node_idx).unwrap();
        if let Some(const_op) = node.op().as_any().downcast_ref::<operator::Const>() {
            if const_op.data.0[[0]] == 5.0 {
                found_folded_const = true;
                break;
            }
        }
    }
    assert!(found_folded_const);
}

#[test]
fn test_graph_inputs() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into()]);

    let a = Graph::new_input(graph.clone(), shape.clone());
    let b = Graph::new_input(graph.clone(), shape.clone());

    let g = graph.lock().unwrap();
    assert_eq!(g.inputs.len(), 2);
    assert!(g.inputs.contains(&a.node_index));
    assert!(g.inputs.contains(&b.node_index));
}
