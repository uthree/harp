use harp::{
    graph::{dtype::Scalar, graph::Graph},
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
