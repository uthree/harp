use harp::backend::c::CCompiler;
use harp::backend::codegen::CodeGenerator;
use harp::backend::c::CRenderer;
use harp::backend::{Compiler, Kernel};
use harp::node::{constant, variable};

#[test]
fn test_c_compiler_is_available() {
    let compiler = CCompiler;
    // This test assumes gcc is in the PATH.
    assert!(compiler.is_available());
}

#[test]
fn test_compile_and_execute_simple_graph() {
    let compiler = CCompiler;
    if !compiler.is_available() {
        eprintln!("Skipping test: gcc not found.");
        return;
    }

    // 1. Create a graph: 1.5 + 2.5
    let a = constant(1.5f32);
    let b = constant(2.5f32);
    let graph = a + b;

    // 2. Generate C code
    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let code = codegen.generate(&graph);

    // 3. Compile the code
    let kernel = compiler.compile(&code).expect("Compilation failed");

    // 4. Execute and check the result
    let result = kernel.execute();
    assert_eq!(result, 4.0);
}

#[test]
fn test_compile_and_execute_fused_op() {
    let compiler = CCompiler;
    if !compiler.is_available() {
        eprintln!("Skipping test: gcc not found.");
        return;
    }

    // 1. Create a graph: 10.0 - 3.0
    let a = constant(10.0f32);
    let b = constant(3.0f32);
    let graph = a - b; // OpSub is a FusedOp

    // 2. Generate C code
    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let code = codegen.generate(&graph);

    // 3. Compile the code
    let kernel = compiler.compile(&code).expect("Compilation failed");

    // 4. Execute and check the result
    let result = kernel.execute();
    assert_eq!(result, 7.0);
}
