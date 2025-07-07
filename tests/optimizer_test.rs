use harp::{
    interpreter::Interpreter,
    node::Node,
    operator::{
        self, Add, Const, Exp2, Input, LessThan, Log2, MaxReduce, Mul, Recip, Rem, Sin, Sqrt,
        SumReduce,
    },
    optimizer::{ConstantFolding, EliminateUnusedNodes, GraphOptimizer, OptimizerPipeline},
    prelude::*,
    tensor::TensorData,
};
use ndarray::{ArrayD, Axis};
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
    let result = interpreter
        .evaluate(add_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();

    assert_eq!(result.0[[0]], 5.0);
}

#[test]
fn test_interpreter_unary_ops() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into()]);

    let data = TensorData(ArrayD::from_elem(vec![1], 1.0));
    let const_node = Node::new(operator::Const { data }, shape.clone());
    let mut g = graph.lock().unwrap();
    let const_idx = g.add_node(const_node);

    let mut interpreter = Interpreter::new();
    let inputs = HashMap::new();

    // Exp2
    let exp2_node = Node::new(operator::Exp2, shape.clone());
    let exp2_idx = g.add_node(exp2_node);
    g.add_edge(const_idx, exp2_idx, 0);
    let result = interpreter
        .evaluate(exp2_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 2.0f32.powf(1.0));

    // Log2
    let log2_node = Node::new(operator::Log2, shape.clone());
    let log2_idx = g.add_node(log2_node);
    g.add_edge(const_idx, log2_idx, 0);
    let result = interpreter
        .evaluate(log2_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 1.0f32.log2());

    // Sin
    let sin_node = Node::new(operator::Sin, shape.clone());
    let sin_idx = g.add_node(sin_node);
    g.add_edge(const_idx, sin_idx, 0);
    let result = interpreter
        .evaluate(sin_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 1.0f32.sin());

    // Sqrt
    let sqrt_node = Node::new(operator::Sqrt, shape.clone());
    let sqrt_idx = g.add_node(sqrt_node);
    g.add_edge(const_idx, sqrt_idx, 0);
    let result = interpreter
        .evaluate(sqrt_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 1.0f32.sqrt());

    // Recip
    let recip_node = Node::new(operator::Recip, shape.clone());
    let recip_idx = g.add_node(recip_node);
    g.add_edge(const_idx, recip_idx, 0);
    let result = interpreter
        .evaluate(recip_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 1.0 / 1.0);
}

#[test]
fn test_interpreter_binary_ops() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into()]);

    let a_data = TensorData(ArrayD::from_elem(vec![1], 5.0));
    let b_data = TensorData(ArrayD::from_elem(vec![1], 2.0));

    let a_node = Node::new(operator::Const { data: a_data }, shape.clone());
    let b_node = Node::new(operator::Const { data: b_data }, shape.clone());

    let mut g = graph.lock().unwrap();
    let a_idx = g.add_node(a_node);
    let b_idx = g.add_node(b_node);

    let mut interpreter = Interpreter::new();
    let inputs = HashMap::new();

    // Mul
    let mul_node = Node::new(operator::Mul, shape.clone());
    let mul_idx = g.add_node(mul_node);
    g.add_edge(a_idx, mul_idx, 0);
    g.add_edge(b_idx, mul_idx, 1);
    let result = interpreter
        .evaluate(mul_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 10.0);

    // Rem
    let rem_node = Node::new(operator::Rem, shape.clone());
    let rem_idx = g.add_node(rem_node);
    g.add_edge(a_idx, rem_idx, 0);
    g.add_edge(b_idx, rem_idx, 1);
    let result = interpreter
        .evaluate(rem_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 1.0); // 5 % 2 = 1

    // LessThan
    let lt_node = Node::new(operator::LessThan, shape.clone());
    let lt_idx = g.add_node(lt_node);
    g.add_edge(a_idx, lt_idx, 0);
    g.add_edge(b_idx, lt_idx, 1);
    let result = interpreter
        .evaluate(lt_idx, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 0.0); // 5 < 2 is false (0.0)

    let a_data_lt = TensorData(ArrayD::from_elem(vec![1], 1.0));
    let b_data_lt = TensorData(ArrayD::from_elem(vec![1], 2.0));
    let a_node_lt = Node::new(operator::Const { data: a_data_lt }, shape.clone());
    let b_node_lt = Node::new(operator::Const { data: b_data_lt }, shape.clone());
    let a_idx_lt = g.add_node(a_node_lt);
    let b_idx_lt = g.add_node(b_node_lt);
    let lt_node_2 = Node::new(operator::LessThan, shape.clone());
    let lt_idx_2 = g.add_node(lt_node_2);
    g.add_edge(a_idx_lt, lt_idx_2, 0);
    g.add_edge(b_idx_lt, lt_idx_2, 1);
    let result = interpreter
        .evaluate(lt_idx_2, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0[[0]], 1.0); // 1 < 2 is true (1.0)
}

#[test]
fn test_interpreter_reduce_ops() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![2.into(), 3.into()]); // Shape (2, 3)

    let data =
        TensorData(ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let const_node = Node::new(operator::Const { data }, shape.clone());
    let mut g = graph.lock().unwrap();
    let const_idx = g.add_node(const_node);

    let mut interpreter = Interpreter::new();
    let inputs = HashMap::new();

    // SumReduce dim 0
    let sum_reduce_node_0 = Node::new(
        operator::SumReduce { dim: 0 },
        ShapeTracker::full(vec![3.into()]),
    );
    let sum_reduce_idx_0 = g.add_node(sum_reduce_node_0);
    g.add_edge(const_idx, sum_reduce_idx_0, 0);
    let result = interpreter
        .evaluate(sum_reduce_idx_0, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0.into_raw_vec(), vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

    // SumReduce dim 1
    let sum_reduce_node_1 = Node::new(
        operator::SumReduce { dim: 1 },
        ShapeTracker::full(vec![2.into()]),
    );
    let sum_reduce_idx_1 = g.add_node(sum_reduce_node_1);
    g.add_edge(const_idx, sum_reduce_idx_1, 0);
    let result = interpreter
        .evaluate(sum_reduce_idx_1, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0.into_raw_vec(), vec![6.0, 15.0]); // [1+2+3, 4+5+6]

    // MaxReduce dim 0
    let max_reduce_node_0 = Node::new(
        operator::MaxReduce { dim: 0 },
        ShapeTracker::full(vec![3.into()]),
    );
    let max_reduce_idx_0 = g.add_node(max_reduce_node_0);
    g.add_edge(const_idx, max_reduce_idx_0, 0);
    let result = interpreter
        .evaluate(max_reduce_idx_0, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0.into_raw_vec(), vec![4.0, 5.0, 6.0]); // [max(1,4), max(2,5), max(3,6)]

    // MaxReduce dim 1
    let max_reduce_node_1 = Node::new(
        operator::MaxReduce { dim: 1 },
        ShapeTracker::full(vec![2.into()]),
    );
    let max_reduce_idx_1 = g.add_node(max_reduce_node_1);
    g.add_edge(const_idx, max_reduce_idx_1, 0);
    let result = interpreter
        .evaluate(max_reduce_idx_1, &g.graph, &inputs, &HashMap::new())
        .unwrap();
    assert_eq!(result.0.into_raw_vec(), vec![3.0, 6.0]); // [max(1,2,3), max(4,5,6)]
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
fn test_constant_folding_complex() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into()]);

    let a_data = TensorData(ArrayD::from_elem(vec![1], 2.0));
    let b_data = TensorData(ArrayD::from_elem(vec![1], 3.0));
    let c_data = TensorData(ArrayD::from_elem(vec![1], 4.0));

    let a = Graph::new_const(graph.clone(), a_data, shape.clone());
    let b = Graph::new_const(graph.clone(), b_data, shape.clone());
    let c = Graph::new_const(graph.clone(), c_data, shape.clone());

    let mut g = graph.lock().unwrap();
    let a_idx = a.node_index;
    let b_idx = b.node_index;
    let c_idx = c.node_index;

    // (a + b) * c
    let add_node = Node::new(operator::Add, shape.clone());
    let add_idx = g.add_node(add_node);
    g.add_edge(a_idx, add_idx, 0);
    g.add_edge(b_idx, add_idx, 1);

    let mul_node = Node::new(operator::Mul, shape.clone());
    let mul_idx = g.add_node(mul_node);
    g.add_edge(add_idx, mul_idx, 0);
    g.add_edge(c_idx, mul_idx, 1);

    let initial_node_count = g.node_count();
    assert_eq!(initial_node_count, 5); // a, b, c, add, mul

    drop(g); // Release lock

    let mut optimizer = ConstantFolding {};
    optimizer.optimize(&mut graph.lock().unwrap());

    let g = graph.lock().unwrap();
    // Expected: (2.0 + 3.0) * 4.0 = 20.0
    let mut found_folded_const = false;
    for node_idx in g.graph.node_indices() {
        let node = g.node_weight(node_idx).unwrap();
        if let Some(const_op) = node.op().as_any().downcast_ref::<operator::Const>() {
            if const_op.data.0[[0]] == 20.0 {
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
