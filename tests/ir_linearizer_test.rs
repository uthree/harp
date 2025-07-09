use harp::{
    graph::{
        dtype::{DType, Scalar},
        graph::Graph,
    },
    ir::linearizer::Linearizer,
    shape::tracker::ShapeTracker,
};
use std::sync::{Arc, Mutex};

#[test]
fn test_compile_const_node() {
    // 1. Create a simple graph with one constant node.
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![1].into();
    let _const_tensor = Graph::new_const(graph_arc.clone(), Scalar::F32(42.0), shape);

    // 2. Compile the graph.
    let mut compiler = Linearizer::new();
    let function = compiler.compile(&graph_arc.lock().unwrap(), "const_test");

    // 3. Verify the generated IR by comparing its string representation.
    let expected_ir = "function const_test:\nkernel main_kernel:\n  v0 = const 42\n";
    assert_eq!(function.to_string().trim(), expected_ir.trim());
}

#[test]
fn test_compile_rand_node() {
    // 1. Create a graph with one RandU node.
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 3].into();
    let _rand_tensor = Graph::randu(graph_arc.clone(), shape.clone(), DType::F32);

    // 2. Compile the graph.
    let mut compiler = Linearizer::new();
    let function = compiler.compile(&graph_arc.lock().unwrap(), "rand_test");

    // 3. Verify the generated IR.
    let expected_ir = "\nfunction rand_test:\nkernel main_kernel:\n  v0 = Uniform [shape=[2, 3], map=(idx * 3) + idx]\n";
    assert_eq!(function.to_string().trim(), expected_ir.trim());
}


