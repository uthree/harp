use harp::{
    dtype::DType,
    graph::Graph,
    interpreter::Interpreter,
    node::Node,
    operator::{
        Add, Const, Exp2, Input, LessThan, Log2, MaxReduce, Mul, Operator, Recip, Rem, Sin, Sqrt,
        SumReduce,
    },
    optimizer::{ConstantFolding, EliminateUnusedNodes, GraphOptimizer, OptimizerPipeline},
    shape::tracker::ShapeTracker,
    tensor::{Tensor, TensorData},
};
use ndarray::{ArrayD, array};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[test]
fn test_eliminate_unused_nodes() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();

    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let _c = &a + &b; // c is not used as output

    let mut optimizer = OptimizerPipeline::new(vec![Box::new(EliminateUnusedNodes {})], 10);
    optimizer.optimize(&mut graph_arc.lock().unwrap());

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 2); // Only inputs should remain
}

#[test]
fn test_constant_folding() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();

    let a_data = TensorData {
        data: ArrayD::from_elem(vec![1], 2.0),
        dtype: DType::F32,
    };
    let b_data = TensorData {
        data: ArrayD::from_elem(vec![1], 3.0),
        dtype: DType::F32,
    };

    let a = Graph::new_const(graph_arc.clone(), a_data, shape.clone());
    let b = Graph::new_const(graph_arc.clone(), b_data, shape.clone());

    let c = &a + &b;
    Graph::add_output_node(graph_arc.clone(), &c);

    let mut optimizer = OptimizerPipeline::new(vec![Box::new(ConstantFolding {})], 10);
    optimizer.optimize(&mut graph_arc.lock().unwrap());

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 1); // Should be folded into a single Const node

    let node_index = graph_locked.outputs[0];
    let node = graph_locked.node_weight(node_index).unwrap();
    let op = node.op();

    assert!(op.as_any().is::<Const>());
    let const_op = op.as_any().downcast_ref::<Const>().unwrap();
    assert_eq!(const_op.data.data[[0]], 5.0);
}

#[test]
fn test_interpreter_unary_ops() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();
    let data = TensorData {
        data: ArrayD::from_elem(vec![1], 1.0),
        dtype: DType::F32,
    };
    let input = Graph::new_const(graph_arc.clone(), data, shape.clone());

    let exp2_tensor = input.exp2();
    Graph::add_output_node(graph_arc.clone(), &exp2_tensor);

    let log2_tensor = input.log2();
    Graph::add_output_node(graph_arc.clone(), &log2_tensor);

    let sin_tensor = input.sin();
    Graph::add_output_node(graph_arc.clone(), &sin_tensor);

    let sqrt_tensor = input.sqrt();
    Graph::add_output_node(graph_arc.clone(), &sqrt_tensor);

    let recip_tensor = input.recip();
    Graph::add_output_node(graph_arc.clone(), &recip_tensor);

    let mut interpreter = Interpreter::new();
    let graph_locked = graph_arc.lock().unwrap();

    let result = interpreter
        .evaluate(
            exp2_tensor.node_index,
            &graph_locked.graph,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(result.data[[0]], 2.0f32.powf(1.0));

    let result = interpreter
        .evaluate(
            log2_tensor.node_index,
            &graph_locked.graph,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(result.data[[0]], 1.0f32.log2());

    let result = interpreter
        .evaluate(
            sin_tensor.node_index,
            &graph_locked.graph,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(result.data[[0]], 1.0f32.sin());

    let result = interpreter
        .evaluate(
            sqrt_tensor.node_index,
            &graph_locked.graph,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(result.data[[0]], 1.0f32.sqrt());

    let result = interpreter
        .evaluate(
            recip_tensor.node_index,
            &graph_locked.graph,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(result.data[[0]], 1.0 / 1.0);
}

/*
#[test]
fn test_interpreter_binary_ops() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();

    let a_data = TensorData { data: ArrayD::from_elem(vec![1], 5.0), dtype: DType::F32 };
    let b_data = TensorData { data: ArrayD::from_elem(vec![1], 2.0), dtype: DType::F32 };

    let a = Graph::new_const(graph_arc.clone(), a_data, shape.clone());
    let b = Graph::new_const(graph_arc.clone(), b_data, shape.clone());

    let add_tensor = &a + &b;
    Graph::add_output_node(graph_arc.clone(), &add_tensor);

    let mul_tensor = &a * &b;
    Graph::add_output_node(graph_arc.clone(), &mul_tensor);

    let rem_tensor = &a % &b;
    Graph::add_output_node(graph_arc.clone(), &rem_tensor);

    let lt_tensor = a.less_than(&b);
    Graph::add_output_node(graph_arc.clone(), &lt_tensor);

    let mut interpreter = Interpreter::new();
    let graph_locked = graph_arc.lock().unwrap();

    let result = interpreter.evaluate(add_tensor.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data[[0]], 7.0);

    let result = interpreter.evaluate(mul_tensor.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data[[0]], 10.0);

    let result = interpreter.evaluate(rem_tensor.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data[[0]], 1.0); // 5 % 2 = 1

    let result = interpreter.evaluate(lt_tensor.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data[[0]], 0.0); // 5 < 2 is false (0.0)

    let a_data_lt = TensorData { data: ArrayD::from_elem(vec![1], 1.0), dtype: DType::F32 };
    let b_data_lt = TensorData { data: ArrayD::from_elem(vec![1], 2.0), dtype: DType::F32 };
    let a_lt = Graph::new_const(graph_arc.clone(), a_data_lt, shape.clone());
    let b_lt = Graph::new_const(graph_arc.clone(), b_data_lt, shape.clone());
    let lt_tensor_true = a_lt.less_than(&b_lt);
    Graph::add_output_node(graph_arc.clone(), &lt_tensor_true);

    let result = interpreter.evaluate(lt_tensor_true.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data[[0]], 1.0); // 1 < 2 is true (1.0)
}

#[test]
fn test_interpreter_reduce_ops() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 3].into();
    let data = TensorData { data: ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), dtype: DType::F32 };
    let input = Graph::new_const(graph_arc.clone(), data, shape.clone());

    let sum_reduce_0 = input.sum_reduce(0);
    Graph::add_output_node(graph_arc.clone(), &sum_reduce_0);

    let sum_reduce_1 = input.sum_reduce(1);
    Graph::add_output_node(graph_arc.clone(), &sum_reduce_1);

    let max_reduce_0 = input.max_reduce(0);
    Graph::add_output_node(graph_arc.clone(), &max_reduce_0);

    let max_reduce_1 = input.max_reduce(1);
    Graph::add_output_node(graph_arc.clone(), &max_reduce_1);

    let mut interpreter = Interpreter::new();
    let graph_locked = graph_arc.lock().unwrap();

    let result = interpreter.evaluate(sum_reduce_0.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data.into_raw_vec_and_offset().0, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

    let result = interpreter.evaluate(sum_reduce_1.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data.into_raw_vec_and_offset().0, vec![6.0, 15.0]); // [1+2+3, 4+5+6]

    let result = interpreter.evaluate(max_reduce_0.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data.into_raw_vec_and_offset().0, vec![4.0, 5.0, 6.0]); // [max(1,4), max(2,5), max(3,6)]

    let result = interpreter.evaluate(max_reduce_1.node_index, &graph_locked.graph, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(result.data.into_raw_vec_and_offset().0, vec![3.0, 6.0]); // [max(1,2,3), max(4,5,6)]
}

#[test]
fn test_interpreter_simple_add() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();

    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);

    let c = &a + &b;
    Graph::add_output_node(graph_arc.clone(), &c);

    let mut interpreter = Interpreter::new();
    let mut global_inputs = HashMap::new();
    global_inputs.insert(a.node_index, TensorData { data: array![1.0].into_dyn(), dtype: DType::F32 });
    global_inputs.insert(b.node_index, TensorData { data: array![2.0].into_dyn(), dtype: DType::F32 });

    let graph_locked = graph_arc.lock().unwrap();
    let result = interpreter.evaluate(c.node_index, &graph_locked.graph, &global_inputs, &HashMap::new()).unwrap();

    assert_eq!(result.data[[0]], 3.0);
}
    */

#[test]
fn test_constant_folding_complex() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();

    let a_data = TensorData {
        data: ArrayD::from_elem(vec![1], 2.0),
        dtype: DType::F32,
    };
    let b_data = TensorData {
        data: ArrayD::from_elem(vec![1], 3.0),
        dtype: DType::F32,
    };
    let c_data = TensorData {
        data: ArrayD::from_elem(vec![1], 4.0),
        dtype: DType::F32,
    };

    let a = Graph::new_const(graph_arc.clone(), a_data, shape.clone());
    let b = Graph::new_const(graph_arc.clone(), b_data, shape.clone());
    let c = Graph::new_const(graph_arc.clone(), c_data, shape.clone());

    let d = &a + &b;
    let e = &d * &c;
    Graph::add_output_node(graph_arc.clone(), &e);

    let mut optimizer = OptimizerPipeline::new(vec![Box::new(ConstantFolding {})], 10);
    optimizer.optimize(&mut graph_arc.lock().unwrap());

    let graph_locked = graph_arc.lock().unwrap();
    assert_eq!(graph_locked.node_count(), 1); // Should be folded into a single Const node

    let node_index = graph_locked.outputs[0];
    let node = graph_locked.node_weight(node_index).unwrap();
    let op = node.op();

    assert!(op.as_any().is::<Const>());
    let const_op = op.as_any().downcast_ref::<Const>().unwrap();
    assert_eq!(const_op.data.data[[0]], 20.0);
}
