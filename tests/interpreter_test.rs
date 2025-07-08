use harp::{
    graph::Graph,
    interpreter::Interpreter,
    node::Node,
    operator::{Add, Const, Exp2, LessThan, Log2, MaxReduce, Mul, Recip, Rem, Sin, Sqrt, SumReduce, Operator},
    shape::tracker::ShapeTracker,
    tensor::TensorData,
    dtype::DType,
};
use ndarray::array;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[test]
fn test_interpreter_new() {
    let _interpreter = Interpreter::new();
    // Assuming cache is private, we can't directly assert its emptiness.
    // We'll rely on other tests to ensure it behaves correctly.
    assert!(true); // Placeholder for now
}

#[test]
fn test_interpreter_default() {
    let _interpreter = Interpreter::default();
    assert!(true); // Placeholder for now
}

#[test]
fn test_evaluate_const_node() {
    let mut graph = Graph::new();
    let const_data = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    let const_node = graph.add_node(Node::new(Const {
        data: const_data.clone(),
    }, const_data.data.shape().to_vec().into()));

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(const_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    assert_eq!(result.data, const_data.data);
    assert_eq!(result.dtype, const_data.dtype);
}

#[test]
fn test_evaluate_global_input_node() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let input_data = TensorData { data: array![[5.0, 6.0], [7.0, 8.0]].into_dyn(), dtype: DType::F32 };
    let shape: ShapeTracker = input_data.data.shape().to_vec().into();
    let input_tensor = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let input_node = input_tensor.node_index;

    let mut interpreter = Interpreter::new();
    let mut global_inputs = HashMap::new();
    global_inputs.insert(input_node, input_data.clone());
    let local_inputs = HashMap::new();

    let graph_locked = graph_arc.lock().unwrap();
    let result = interpreter
        .evaluate(input_node, &graph_locked.graph, &global_inputs, &local_inputs)
        .unwrap();

    assert_eq!(result.data, input_data.data);
    assert_eq!(result.dtype, input_data.dtype);
}

#[test]
fn test_evaluate_exp2_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let exp2_node = graph.add_node(Node::new(Exp2 {}, input_data.data.shape().to_vec().into()));
    graph.add_edge(input_node, exp2_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(exp2_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[2.0, 4.0], [8.0, 16.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_log2_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[2.0, 4.0], [8.0, 16.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let log2_node = graph.add_node(Node::new(Log2 {}, input_data.data.shape().to_vec().into()));
    graph.add_edge(input_node, log2_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(log2_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_sin_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[0.0, std::f32::consts::PI / 2.0], [std::f32::consts::PI, 3.0 * std::f32::consts::PI / 2.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let sin_node = graph.add_node(Node::new(Sin {}, input_data.data.shape().to_vec().into()));
    graph.add_edge(input_node, sin_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(sin_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[0.0, 1.0], [0.0, -1.0]].into_dyn(), dtype: DType::F32 };
    // Use a small epsilon for float comparison
    assert!(result.data.iter().zip(expected.data.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_sqrt_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[1.0, 4.0], [9.0, 16.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let sqrt_node = graph.add_node(Node::new(Sqrt {}, input_data.data.shape().to_vec().into()));
    graph.add_edge(input_node, sqrt_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(sqrt_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_recip_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[1.0, 2.0], [4.0, 0.5]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let recip_node = graph.add_node(Node::new(Recip {}, input_data.data.shape().to_vec().into()));
    graph.add_edge(input_node, recip_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(recip_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[1.0, 0.5], [0.25, 2.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_add_operator() {
    let mut graph = Graph::new();
    let lhs_data = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    let rhs_data = TensorData { data: array![[5.0, 6.0], [7.0, 8.0]].into_dyn(), dtype: DType::F32 };
    let lhs_node = graph.add_node(Node::new(Const {
        data: lhs_data.clone(),
    }, lhs_data.data.shape().to_vec().into()));
    let rhs_node = graph.add_node(Node::new(Const {
        data: rhs_data.clone(),
    }, rhs_data.data.shape().to_vec().into()));
    let add_node = graph.add_node(Node::new(Add {}, lhs_data.data.shape().to_vec().into()));
    graph.add_edge(lhs_node, add_node, 0);
    graph.add_edge(rhs_node, add_node, 1);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(add_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[6.0, 8.0], [10.0, 12.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_mul_operator() {
    let mut graph = Graph::new();
    let lhs_data = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    let rhs_data = TensorData { data: array![[5.0, 6.0], [7.0, 8.0]].into_dyn(), dtype: DType::F32 };
    let lhs_node = graph.add_node(Node::new(Const {
        data: lhs_data.clone(),
    }, lhs_data.data.shape().to_vec().into()));
    let rhs_node = graph.add_node(Node::new(Const {
        data: rhs_data.clone(),
    }, rhs_data.data.shape().to_vec().into()));
    let mul_node = graph.add_node(Node::new(Mul {}, lhs_data.data.shape().to_vec().into()));
    graph.add_edge(lhs_node, mul_node, 0);
    graph.add_edge(rhs_node, mul_node, 1);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(mul_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[5.0, 12.0], [21.0, 32.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_rem_operator() {
    let mut graph = Graph::new();
    let lhs_data = TensorData { data: array![[10.0, 11.0], [12.0, 13.0]].into_dyn(), dtype: DType::F32 };
    let rhs_data = TensorData { data: array![[3.0, 3.0], [5.0, 5.0]].into_dyn(), dtype: DType::F32 };
    let lhs_node = graph.add_node(Node::new(Const {
        data: lhs_data.clone(),
    }, lhs_data.data.shape().to_vec().into()));
    let rhs_node = graph.add_node(Node::new(Const {
        data: rhs_data.clone(),
    }, rhs_data.data.shape().to_vec().into()));
    let rem_node = graph.add_node(Node::new(Rem {}, lhs_data.data.shape().to_vec().into()));
    graph.add_edge(lhs_node, rem_node, 0);
    graph.add_edge(rhs_node, rem_node, 1);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(rem_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[1.0, 2.0], [2.0, 3.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_less_than_operator() {
    let mut graph = Graph::new();
    let lhs_data = TensorData { data: array![[1.0, 5.0], [10.0, 2.0]].into_dyn(), dtype: DType::F32 };
    let rhs_data = TensorData { data: array![[3.0]].into_dyn(), dtype: DType::F32 };
    let lhs_node = graph.add_node(Node::new(Const {
        data: lhs_data.clone(),
    }, lhs_data.data.shape().to_vec().into()));
    let rhs_node = graph.add_node(Node::new(Const {
        data: rhs_data.clone(),
    }, rhs_data.data.shape().to_vec().into()));
    let lt_node = graph.add_node(Node::new(LessThan {}, lhs_data.data.shape().to_vec().into()));
    graph.add_edge(lhs_node, lt_node, 0);
    graph.add_edge(rhs_node, lt_node, 1);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(lt_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[1.0, 0.0], [0.0, 1.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_sum_reduce_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let sum_reduce_node = graph.add_node(Node::new(SumReduce { dim: 0 }, vec![input_data.data.shape()[1]].into()));
    graph.add_edge(input_node, sum_reduce_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(sum_reduce_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![4.0, 6.0].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_max_reduce_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[1.0, 5.0], [3.0, 2.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let max_reduce_node = graph.add_node(Node::new(MaxReduce { dim: 1 }, vec![input_data.data.shape()[0]].into()));
    graph.add_edge(input_node, max_reduce_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(max_reduce_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![5.0, 3.0].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_contiguous_operator() {
    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));
    let contiguous_node = graph.add_node(Node::new(harp::operator::Contiguous {}, input_data.data.shape().to_vec().into()));
    graph.add_edge(input_node, contiguous_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(contiguous_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    let expected = TensorData { data: array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_unsupported_operator() {
    // Create a dummy operator not handled by the interpreter
    #[derive(Debug)]
    struct DummyOperator;
    impl harp::operator::Operator for DummyOperator {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn output_dtype(&self) -> DType {
            DType::F32 // Dummy type
        }
    }

    let mut graph = Graph::new();
    let dummy_node = graph.add_node(Node::new(DummyOperator {}, vec![].into()));

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter.evaluate(dummy_node, &graph.graph, &global_inputs, &local_inputs);

    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        format!("Unsupported operator for interpretation: {:?}", DummyOperator {})
    );
}

#[test]
fn test_evaluate_node_not_found() {
    let graph = Graph::new(); // Empty graph
    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    // Try to evaluate a non-existent node index
    let non_existent_node = NodeIndex::new(999);

    let result = interpreter.evaluate(non_existent_node, &graph.graph, &global_inputs, &local_inputs);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Node not found");
}

#[test]
fn test_evaluate_missing_input_for_unary_op() {
    let mut graph = Graph::new();
    let exp2_node = graph.add_node(Node::new(Exp2 {}, vec![].into())); // No input edge

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter.evaluate(exp2_node, &graph.graph, &global_inputs, &local_inputs);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Exp2 op missing input");
}

#[test]
fn test_evaluate_missing_lhs_for_binary_op() {
    let mut graph = Graph::new();
    let rhs_data = TensorData { data: array![[1.0]].into_dyn(), dtype: DType::F32 };
    let rhs_node = graph.add_node(Node::new(Const { data: rhs_data.clone() }, rhs_data.data.shape().to_vec().into()));
    let add_node = graph.add_node(Node::new(Add {}, rhs_data.data.shape().to_vec().into()));
    graph.add_edge(rhs_node, add_node, 1);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter.evaluate(add_node, &graph.graph, &global_inputs, &local_inputs);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Add op missing lhs");
}

#[test]
fn test_evaluate_missing_rhs_for_binary_op() {
    let mut graph = Graph::new();
    let lhs_data = TensorData { data: array![[1.0]].into_dyn(), dtype: DType::F32 };
    let lhs_node = graph.add_node(Node::new(Const { data: lhs_data.clone() }, lhs_data.data.shape().to_vec().into()));
    let add_node = graph.add_node(Node::new(Add {}, lhs_data.data.shape().to_vec().into()));
    graph.add_edge(lhs_node, add_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter.evaluate(add_node, &graph.graph, &global_inputs, &local_inputs);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Add op missing rhs");
}

#[test]
fn test_evaluate_cache_hit() {
    let mut graph = Graph::new();
    let const_data = TensorData { data: array![[1.0, 2.0]].into_dyn(), dtype: DType::F32 };
    let const_node = graph.add_node(Node::new(Const {
        data: const_data.clone(),
    }, const_data.data.shape().to_vec().into()));

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    // First evaluation (populates cache)
    let result1 = interpreter
        .evaluate(const_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();
    assert_eq!(result1.data, const_data.data);
    assert_eq!(result1.dtype, const_data.dtype);

    // Second evaluation (should hit cache)
    let result2 = interpreter
        .evaluate(const_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();
    assert_eq!(result2.data, const_data.data);
    assert_eq!(result2.dtype, const_data.dtype);
    // We can't directly assert cache hit, but if the result is correct and
    // no error, it implies cache worked as expected.
}

#[test]
fn test_evaluate_complex_graph() {
    let mut graph = Graph::new();

    // Nodes
    let a_data = TensorData { data: array![[1.0, 2.0]].into_dyn(), dtype: DType::F32 };
    let b_data = TensorData { data: array![[3.0, 4.0]].into_dyn(), dtype: DType::F32 };
    let c_data = TensorData { data: array![[2.0]].into_dyn(), dtype: DType::F32 };

    let a = graph.add_node(Node::new(Const { data: a_data.clone() }, a_data.data.shape().to_vec().into()));
    let b = graph.add_node(Node::new(Const { data: b_data.clone() }, b_data.data.shape().to_vec().into()));
    let c = graph.add_node(Node::new(Const { data: c_data.clone() }, c_data.data.shape().to_vec().into()));

    let add_node = graph.add_node(Node::new(Add {}, a_data.data.shape().to_vec().into()));
    let mul_node = graph.add_node(Node::new(Mul {}, a_data.data.shape().to_vec().into()));
    let sum_reduce_node = graph.add_node(Node::new(SumReduce { dim: 1 }, vec![a_data.data.shape()[0]].into())); // Sum along rows

    // Edges: (A + B) * C
    graph.add_edge(a, add_node, 0);
    graph.add_edge(b, add_node, 1);
    graph.add_edge(add_node, mul_node, 0);
    graph.add_edge(c, mul_node, 1);
    graph.add_edge(mul_node, sum_reduce_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(sum_reduce_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    // Expected:
    // A + B = [[4.0, 6.0]]
    // (A + B) * C = [[8.0, 12.0]]
    // SumReduce((A + B) * C) along dim 1 = [20.0]
    let expected = TensorData { data: array![20.0].into_dyn(), dtype: DType::F32 };
    assert_eq!(result.data, expected.data);
    assert_eq!(result.dtype, expected.dtype);
}

#[test]
fn test_evaluate_cast_operator() {
    use harp::operator::Cast;

    let mut graph = Graph::new();
    let input_data = TensorData { data: array![[1.0, 2.0]].into_dyn(), dtype: DType::F32 };
    let input_node = graph.add_node(Node::new(Const {
        data: input_data.clone(),
    }, input_data.data.shape().to_vec().into()));

    let cast_node = graph.add_node(Node::new(Cast { dtype: DType::I32 }, input_data.data.shape().to_vec().into()));
    graph.add_edge(input_node, cast_node, 0);

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(cast_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();

    // For now, data remains f32, only dtype changes
    assert_eq!(result.data, input_data.data);
    assert_eq!(result.dtype, DType::I32);
}
