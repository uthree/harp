use harp::dtype::DType;
use harp::interpreter::Interpreter;
use harp::prelude::*;
use harp::tensor::TensorData;
use ndarray::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
fn main() {
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();

    let a_data = TensorData {
        data: ArrayD::from_elem(vec![1], 5.0),
        dtype: DType::F32,
    };
    let b_data = TensorData {
        data: ArrayD::from_elem(vec![1], 2.0),
        dtype: DType::F32,
    };

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

    let a_data_lt = TensorData {
        data: ArrayD::from_elem(vec![1], 1.0),
        dtype: DType::F32,
    };
    let b_data_lt = TensorData {
        data: ArrayD::from_elem(vec![1], 2.0),
        dtype: DType::F32,
    };
    let a_lt = Graph::new_const(graph_arc.clone(), a_data_lt, shape.clone());
    let b_lt = Graph::new_const(graph_arc.clone(), b_data_lt, shape.clone());

    let lt_tensor_true = a_lt.less_than(&b_lt);
    Graph::add_output_node(graph_arc.clone(), &lt_tensor_true);

    // 5. 構築されたグラフをDOT形式で出力
    let graph = graph_arc.lock().unwrap();
    println!("{}", graph.to_dot());

    let mut interpreter = Interpreter::new();
    let global_inputs = HashMap::new();
    let local_inputs = HashMap::new();

    let result = interpreter
        .evaluate(sum_reduce_node, &graph.graph, &global_inputs, &local_inputs)
        .unwrap();
}
