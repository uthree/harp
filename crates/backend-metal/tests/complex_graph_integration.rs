//! Metal integration tests for complex graphs
//!
//! This test verifies that complex graphs with multiple kernels
//! (including Contiguous nodes) work correctly.

#![cfg(target_os = "macos")]

use harp_backend_metal::{MetalCompiler, MetalDevice, MetalRenderer};
use harp_core::ast::DType as AstDType;
use harp_core::backend::pipeline::Pipeline;
use harp_core::backend::traits::{Buffer, Compiler};
use harp_core::graph::{DType, Graph};
use std::collections::HashMap;

/// Helper to get Metal device, returns None if unavailable
fn get_device() -> Option<MetalDevice> {
    match MetalDevice::new() {
        Ok(d) => {
            println!("Using Metal device: {}", d.device_name());
            Some(d)
        }
        Err(e) => {
            eprintln!("Skipping test: No Metal device available: {}", e);
            None
        }
    }
}

/// Test: Simple add with two inputs
///
/// This tests basic multiple buffer input handling.
#[test]
fn test_simple_add_two_inputs() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // Create graph: c = a + b
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![4]);
    let b = graph.input("b", DType::F32, vec![4]);
    let c = a + b;
    graph.output("c", c);

    // Create pipeline
    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    // Compile
    let program = pipeline.compile_program(graph).expect("Failed to compile");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline
        .allocate_buffer(vec![4], AstDType::F32)
        .expect("Failed to allocate a");
    let mut buf_b = pipeline
        .allocate_buffer(vec![4], AstDType::F32)
        .expect("Failed to allocate b");
    let mut buf_c = pipeline
        .allocate_buffer(vec![4], AstDType::F32)
        .expect("Failed to allocate c");

    // Initialize input data
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let data_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    buf_a.write_vec(&data_a).expect("Failed to write a");
    buf_b.write_vec(&data_b).expect("Failed to write b");

    // Execute using HashMap-based API
    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    // Read result
    let result: Vec<f32> = buf_c.read_vec().expect("Failed to read result");

    // Verify
    let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "Simple add result mismatch");
}

/// Test: Complex graph with transpose and contiguous
///
/// This creates a graph that requires multiple kernels:
/// 1. Transpose (permute) + contiguous for both inputs
/// 2. Element-wise addition
#[test]
fn test_transpose_contiguous_add() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // Create graph with transpose + contiguous
    // a: [2, 3] -> transpose -> [3, 2] -> contiguous
    // b: [3, 2] (already contiguous)
    // c = a_transposed + b
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);
    let b = graph.input("b", DType::F32, vec![3, 2]);

    // Transpose a: [2, 3] -> [3, 2]
    let a_transposed_view = a.view.clone().permute(vec![1, 0]);
    let a_transposed = a.view(a_transposed_view);

    // Make contiguous (this should create a separate kernel)
    let a_contiguous = a_transposed.contiguous();

    // Add
    let c = a_contiguous + b;
    graph.output("c", c);

    // Create pipeline
    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    // Compile
    let program = match pipeline.compile_program(graph) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            panic!("Failed to compile graph");
        }
    };

    // Check that multiple kernels were generated
    println!("Number of kernels: {}", program.kernel_count());
    println!("Execution waves: {:?}", program.execution_waves);
    println!("Input names: {:?}", program.input_names);
    println!("Output names: {:?}", program.output_names);
    println!("Kernels: {:?}", program.kernels.keys().collect::<Vec<_>>());

    // Allocate buffers
    let mut buf_a = pipeline
        .allocate_buffer(vec![2, 3], AstDType::F32)
        .expect("Failed to allocate a");
    let mut buf_b = pipeline
        .allocate_buffer(vec![3, 2], AstDType::F32)
        .expect("Failed to allocate b");
    let mut buf_c = pipeline
        .allocate_buffer(vec![3, 2], AstDType::F32)
        .expect("Failed to allocate c");

    // Initialize input data
    // a: [[1, 2, 3], [4, 5, 6]] (row-major)
    // a transposed: [[1, 4], [2, 5], [3, 6]]
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // b: [[10, 20], [30, 40], [50, 60]]
    let data_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    buf_a.write_vec(&data_a).expect("Failed to write a");
    buf_b.write_vec(&data_b).expect("Failed to write b");

    // Execute using HashMap-based API
    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    // Read result
    let result: Vec<f32> = buf_c.read_vec().expect("Failed to read result");

    // Verify
    // a_transposed: [[1, 4], [2, 5], [3, 6]]
    // b:            [[10, 20], [30, 40], [50, 60]]
    // c = a_t + b:  [[11, 24], [32, 45], [53, 66]]
    let expected: Vec<f32> = vec![11.0, 24.0, 32.0, 45.0, 53.0, 66.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(
        result, expected,
        "Transpose + contiguous + add result mismatch"
    );
}

/// Test: Chain of operations with multiple contiguous nodes
///
/// a -> transpose -> contiguous -> add(b) -> transpose -> contiguous -> output
#[test]
fn test_multiple_contiguous_chain() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);
    let b = graph.input("b", DType::F32, vec![3, 2]);

    // First transpose + contiguous
    let a_t = a.view(a.view.clone().permute(vec![1, 0])).contiguous();

    // Add
    let sum = a_t + b;

    // Second transpose + contiguous
    let sum_t = sum.view(sum.view.clone().permute(vec![1, 0])).contiguous();

    graph.output("c", sum_t);

    // Create pipeline
    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    // Compile
    let program = match pipeline.compile_program(graph) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            panic!("Failed to compile graph");
        }
    };

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers (output is [2, 3] after double transpose)
    let mut buf_a = pipeline
        .allocate_buffer(vec![2, 3], AstDType::F32)
        .expect("Failed to allocate a");
    let mut buf_b = pipeline
        .allocate_buffer(vec![3, 2], AstDType::F32)
        .expect("Failed to allocate b");
    let mut buf_c = pipeline
        .allocate_buffer(vec![2, 3], AstDType::F32)
        .expect("Failed to allocate c");

    // Initialize
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    buf_a.write_vec(&data_a).expect("Failed to write a");
    buf_b.write_vec(&data_b).expect("Failed to write b");

    // Execute using HashMap-based API
    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    // Read result
    let result: Vec<f32> = buf_c.read_vec().expect("Failed to read result");

    // Verify
    // a:                [[1, 2, 3], [4, 5, 6]]
    // a^T:              [[1, 4], [2, 5], [3, 6]]
    // b:                [[10, 20], [30, 40], [50, 60]]
    // a^T + b:          [[11, 24], [32, 45], [53, 66]]
    // (a^T + b)^T:      [[11, 32, 53], [24, 45, 66]]
    let expected: Vec<f32> = vec![11.0, 32.0, 53.0, 24.0, 45.0, 66.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(
        result, expected,
        "Multiple contiguous chain result mismatch"
    );
}

/// Test: Four inputs with fusion
///
/// (a + b) * (c + d) - tests complex multi-input scenarios
#[test]
fn test_four_inputs_fused() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![4]);
    let b = graph.input("b", DType::F32, vec![4]);
    let c = graph.input("c", DType::F32, vec![4]);
    let d = graph.input("d", DType::F32, vec![4]);

    let sum1 = a + b;
    let sum2 = c + d;
    let result = sum1 * sum2;
    graph.output("result", result);

    // Create pipeline
    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    // Compile
    let program = match pipeline.compile_program(graph) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            panic!("Failed to compile graph");
        }
    };

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![4], AstDType::F32).unwrap();
    let mut buf_b = pipeline.allocate_buffer(vec![4], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![4], AstDType::F32).unwrap();
    let mut buf_d = pipeline.allocate_buffer(vec![4], AstDType::F32).unwrap();
    let mut buf_result = pipeline.allocate_buffer(vec![4], AstDType::F32).unwrap();

    // Initialize
    buf_a.write_vec(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    buf_b.write_vec(&[1.0f32, 1.0, 1.0, 1.0]).unwrap();
    buf_c.write_vec(&[10.0f32, 10.0, 10.0, 10.0]).unwrap();
    buf_d.write_vec(&[0.0f32, 1.0, 2.0, 3.0]).unwrap();

    // Execute using HashMap-based API
    let inputs: HashMap<String, _> = [
        ("a".to_string(), &buf_a),
        ("b".to_string(), &buf_b),
        ("c".to_string(), &buf_c),
        ("d".to_string(), &buf_d),
    ]
    .into_iter()
    .collect();
    let mut outputs: HashMap<String, _> = [("result".to_string(), &mut buf_result)]
        .into_iter()
        .collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    // Read result
    let result: Vec<f32> = buf_result.read_vec().unwrap();

    // Verify
    // (a + b) = [2, 3, 4, 5]
    // (c + d) = [10, 11, 12, 13]
    // result = [20, 33, 48, 65]
    let expected: Vec<f32> = vec![20.0, 33.0, 48.0, 65.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "Four inputs fused result mismatch");
}

/// Test: Reduce operation with multiple inputs
///
/// reduce_sum(a * b, axis=1)
#[test]
fn test_reduce_with_two_inputs() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);
    let b = graph.input("b", DType::F32, vec![2, 3]);

    let product = a * b;
    let sum = product.reduce_sum(1); // [2, 3] -> [2]
    graph.output("c", sum);

    // Create pipeline
    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    // Compile
    let program = match pipeline.compile_program(graph) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            panic!("Failed to compile graph");
        }
    };

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_b = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![2], AstDType::F32).unwrap();

    // Initialize
    // a: [[1, 2, 3], [4, 5, 6]]
    // b: [[1, 1, 1], [2, 2, 2]]
    buf_a.write_vec(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    buf_b.write_vec(&[1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();

    // Execute using HashMap-based API
    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    // Read result
    let result: Vec<f32> = buf_c.read_vec().unwrap();

    // Verify
    // a * b: [[1, 2, 3], [8, 10, 12]]
    // sum axis=1: [6, 30]
    let expected: Vec<f32> = vec![6.0, 30.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "Reduce with two inputs result mismatch");
}
