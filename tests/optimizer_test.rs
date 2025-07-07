use harp::{
    interpreter::Interpreter,
    node::Node,
    operator::{self, Add, Const, Exp2, Input, Mul},
    optimizer::{ConstantFolding, EliminateUnusedNodes, GraphOptimizer, OptimizerPipeline},
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

#[test]
fn test_optimizer_pipeline() {
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

    let mul_node = Node::new(operator::Mul, shape.clone());
    let mul_idx = g.add_node(mul_node);
    g.add_edge(add_idx, mul_idx, 0);
    g.add_edge(a_idx, mul_idx, 1);

    let exp2_node = Node::new(operator::Exp2, shape.clone());
    let exp2_idx = g.add_node(exp2_node);
    g.add_edge(mul_idx, exp2_idx, 0);

    // Mark exp2_idx as an output
    g.outputs.push(exp2_idx);

    let initial_node_count = g.node_count();
    assert_eq!(initial_node_count, 6); // 2 Const + Add + Mul + Exp2 + 1 unused (a_idx, b_idx, add_idx, mul_idx, exp2_idx)

    drop(g); // Release lock

    let mut pipeline = OptimizerPipeline::new(
        vec![
            Box::new(ConstantFolding {}),
            Box::new(EliminateUnusedNodes {}),
        ],
        10, // Max iterations
    );

    pipeline.optimize(&mut graph.lock().unwrap());

    let g = graph.lock().unwrap();
    // After optimization, we expect the graph to be simplified.
    // The (2.0 + 3.0) * 2.0 should be folded into a single Const node (10.0).
    // The Exp2(10.0) should remain.
    // The initial Const nodes (a_idx, b_idx) should be removed if they are no longer used.

    // Check if the graph contains a Const node with value 10.0
    let mut found_folded_const = false;
    for node_idx in g.graph.node_indices() {
        let node = g.node_weight(node_idx).unwrap();
        if let Some(const_op) = node.op().as_any().downcast_ref::<operator::Const>() {
            if const_op.data.0[[0]] == 10.0 {
                found_folded_const = true;
                break;
            }
        }
    }
    assert!(found_folded_const);

    // Check if the graph contains the Exp2 node
    let mut found_exp2 = false;
    for node_idx in g.graph.node_indices() {
        let node = g.node_weight(node_idx).unwrap();
        if node.op().as_any().downcast_ref::<operator::Exp2>().is_some() {
            found_exp2 = true;
            break;
        }
    }
    assert!(found_exp2);

    // The final graph should have 2 nodes: the folded Const and the Exp2 node.
    assert_eq!(g.node_count(), 2);
}