use harp::{
    graph::{
        dtype::{DType, Scalar},
        graph::Graph,
    },
    ir::compiler::Compiler,
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
    let mut compiler = Compiler::new();
    let function = compiler.compile(&graph_arc.lock().unwrap(), "const_test");

    // 3. Verify the generated IR by comparing its string representation.
    let expected_ir = "function const_test:\nkernel main_kernel:\n  v0 = const 42\n";
    assert_eq!(function.to_string().trim(), expected_ir.trim());
}

#[test]
fn test_compile_add_op() {
    // 1. Create a graph with two inputs and one add operation.
    let graph_arc = Arc::new(Mutex::new(Graph::new()));
    let shape: ShapeTracker = vec![2, 3].into();
    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let _c = &a + &b;

    // 2. Compile the graph.
    let mut compiler = Compiler::new();
    let function = compiler.compile(&graph_arc.lock().unwrap(), "add_test");

    // 3. Verify the generated IR.
    let expected_ir = "\nfunction add_test:\nkernel main_kernel:\n  v0 = load buf0[shape=[2, 3], map=(idx * 3) + idx]\n  v1 = load buf1[shape=[2, 3], map=(idx * 3) + idx]\n  v2 = Add v0, v1\n";
    assert_eq!(function.to_string().trim(), expected_ir.trim());
}

