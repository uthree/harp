use harp::backend::c::CBackend;
use harp::backend::{Backend, Kernel};
use harp::node::constant;

#[test]
fn test_c_backend_is_available() {
    let backend = CBackend::new();
    // This test assumes gcc is in the PATH.
    assert!(backend.is_available());
}

#[test]
fn test_compile_and_execute_simple_graph() {
    let backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Skipping test: C backend (gcc) not found.");
        return;
    }

    // 1. Create a graph: 1.5 + 2.5
    let a = constant(1.5f32);
    let b = constant(2.5f32);
    let graph = a + b;

    // 2. Compile the graph using the backend
    let kernel = backend.compile(&graph).expect("Compilation failed");

    // 3. Execute and check the result
    let result = kernel.execute();
    assert_eq!(result, 4.0);
}

#[test]
fn test_compile_and_execute_fused_op() {
    let backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Skipping test: C backend (gcc) not found.");
        return;
    }

    // 1. Create a graph: 10.0 - 3.0
    let a = constant(10.0f32);
    let b = constant(3.0f32);
    let graph = a - b; // OpSub is a FusedOp

    // 2. Compile the graph using the backend
    let kernel = backend.compile(&graph).expect("Compilation failed");

    // 3. Execute and check the result
    let result = kernel.execute();
    assert_eq!(result, 7.0);
}
