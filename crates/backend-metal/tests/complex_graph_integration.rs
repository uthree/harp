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

    // Debug: Get the AST before AST optimization
    use harp_core::lowerer::{create_lowering_optimizer, extract_program};
    use harp_core::opt::graph::GraphOptimizer;

    let optimizer = create_lowering_optimizer(4, 5000);
    let (optimized_graph, _) = optimizer.optimize_with_history(graph.clone());
    let ast_before_opt = extract_program(optimized_graph);
    println!("AST before AST optimization:\n{:#?}", ast_before_opt);

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

// ============================================================================
// Non-contiguous View operation tests
// These tests verify that strided/transposed/broadcast Views are correctly
// handled without requiring explicit contiguous() calls.
// ============================================================================

/// Test: Elementwise with transposed input (no contiguous)
///
/// Tests that transpose View is correctly handled in elementwise operations
/// without requiring an explicit contiguous() call.
#[test]
fn test_elementwise_with_transpose_no_contiguous() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // a: [2, 3] -> transpose -> [3, 2] (strided view, NOT contiguous)
    // b: [3, 2] (contiguous)
    // c = a^T + b (should work without explicit contiguous)
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);
    let b = graph.input("b", DType::F32, vec![3, 2]);

    // Transpose a without making it contiguous
    let a_transposed = a.view(a.view.clone().permute(vec![1, 0]));

    // Direct addition with transposed view
    let c = a_transposed + b;
    graph.output("c", c);

    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    let program = pipeline
        .compile_program(graph)
        .expect("Failed to compile graph with transposed view");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_b = pipeline.allocate_buffer(vec![3, 2], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![3, 2], AstDType::F32).unwrap();

    // a: [[1, 2, 3], [4, 5, 6]] in row-major
    // a^T: [[1, 4], [2, 5], [3, 6]]
    buf_a.write_vec(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    // b: [[10, 20], [30, 40], [50, 60]]
    buf_b
        .write_vec(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0])
        .unwrap();

    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    let result: Vec<f32> = buf_c.read_vec().unwrap();

    // a^T + b = [[11, 24], [32, 45], [53, 66]]
    let expected: Vec<f32> = vec![11.0, 24.0, 32.0, 45.0, 53.0, 66.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(
        result, expected,
        "Elementwise with transpose (no contiguous) mismatch"
    );
}

/// Test: Elementwise with broadcast input
///
/// Tests that broadcast View is correctly handled in elementwise operations.
#[test]
fn test_elementwise_with_broadcast() {
    use harp_core::graph::shape::Expr;

    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // a: [3, 1] -> broadcast -> [3, 4]
    // b: [3, 4]
    // c = a_broadcast + b
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![3, 1]);
    let b = graph.input("b", DType::F32, vec![3, 4]);

    // Broadcast a to [3, 4]
    let a_broadcast = a.broadcast_to(vec![Expr::from(3), Expr::from(4)]);

    let c = a_broadcast + b;
    graph.output("c", c);

    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    let program = pipeline
        .compile_program(graph)
        .expect("Failed to compile graph with broadcast view");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![3, 1], AstDType::F32).unwrap();
    let mut buf_b = pipeline.allocate_buffer(vec![3, 4], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![3, 4], AstDType::F32).unwrap();

    // a: [[1], [2], [3]] -> broadcast -> [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
    buf_a.write_vec(&[1.0f32, 2.0, 3.0]).unwrap();
    // b: [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
    buf_b
        .write_vec(&[
            10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ])
        .unwrap();

    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    let result: Vec<f32> = buf_c.read_vec().unwrap();

    // a_broadcast + b = [[11, 21, 31, 41], [52, 62, 72, 82], [93, 103, 113, 123]]
    let expected: Vec<f32> = vec![
        11.0, 21.0, 31.0, 41.0, 52.0, 62.0, 72.0, 82.0, 93.0, 103.0, 113.0, 123.0,
    ];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "Elementwise with broadcast mismatch");
}

/// Test: Reduce with transposed input
///
/// Tests that reduce correctly handles transposed (strided) input.
#[test]
fn test_reduce_with_transposed_input() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // a: [2, 3] -> transpose -> [3, 2] -> reduce_sum(axis=1) -> [3]
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);

    let a_t = a.view(a.view.clone().permute(vec![1, 0]));
    let sum = a_t.reduce_sum(1); // [3, 2] -> [3]
    graph.output("c", sum);

    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    let program = pipeline
        .compile_program(graph)
        .expect("Failed to compile graph with transposed reduce");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![3], AstDType::F32).unwrap();

    // a: [[1, 2, 3], [4, 5, 6]]
    // a^T: [[1, 4], [2, 5], [3, 6]]
    buf_a.write_vec(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a)].into_iter().collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    let result: Vec<f32> = buf_c.read_vec().unwrap();

    // reduce_sum(a^T, axis=1) = [1+4, 2+5, 3+6] = [5, 7, 9]
    let expected: Vec<f32> = vec![5.0, 7.0, 9.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "Reduce with transposed input mismatch");
}

/// Test: Cast with transposed input
///
/// Tests that cast correctly handles transposed (strided) input.
///
/// NOTE: Currently ignored due to a pre-existing bug in Metal cast rendering
/// where float bits are reinterpreted as integers instead of proper conversion.
#[test]
#[ignore = "Metal cast F32->I32 rendering bug - float bits reinterpreted as int"]
fn test_cast_with_transposed_input() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // a: [2, 3] F32 -> transpose -> [3, 2] -> cast to I32 -> [3, 2]
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);

    let a_t = a.view(a.view.clone().permute(vec![1, 0]));
    let cast = a_t.cast(DType::I32);
    graph.output("c", cast);

    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    let program = pipeline
        .compile_program(graph)
        .expect("Failed to compile graph with transposed cast");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![3, 2], AstDType::I32).unwrap();

    // a: [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
    // a^T: [[1.5, 4.5], [2.5, 5.5], [3.5, 6.5]]
    buf_a.write_vec(&[1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap();

    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a)].into_iter().collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    let result: Vec<i32> = buf_c.read_vec().unwrap();

    // cast(a^T) = [[1, 4], [2, 5], [3, 6]]
    let expected: Vec<i32> = vec![1, 4, 2, 5, 3, 6];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "Cast with transposed input mismatch");
}

/// Test: Combined transpose and broadcast
///
/// Tests complex View chain: transpose + broadcast + elementwise.
#[test]
fn test_transpose_and_broadcast_combined() {
    use harp_core::graph::shape::Expr;

    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // a: [2, 3] -> transpose -> [3, 2] (strided)
    // b: [1, 2] -> broadcast -> [3, 2] (broadcast)
    // c = a^T * b_broadcast
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);
    let b = graph.input("b", DType::F32, vec![1, 2]);

    let a_t = a.view(a.view.clone().permute(vec![1, 0]));
    let b_broadcast = b.broadcast_to(vec![Expr::from(3), Expr::from(2)]);

    let c = a_t * b_broadcast;
    graph.output("c", c);

    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    let program = pipeline
        .compile_program(graph)
        .expect("Failed to compile graph with transpose + broadcast");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_b = pipeline.allocate_buffer(vec![1, 2], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![3, 2], AstDType::F32).unwrap();

    // a: [[1, 2, 3], [4, 5, 6]]
    // a^T: [[1, 4], [2, 5], [3, 6]]
    buf_a.write_vec(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    // b: [[10, 100]] -> broadcast -> [[10, 100], [10, 100], [10, 100]]
    buf_b.write_vec(&[10.0f32, 100.0]).unwrap();

    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    let result: Vec<f32> = buf_c.read_vec().unwrap();

    // a^T * b_broadcast = [[1*10, 4*100], [2*10, 5*100], [3*10, 6*100]]
    //                   = [[10, 400], [20, 500], [30, 600]]
    let expected: Vec<f32> = vec![10.0, 400.0, 20.0, 500.0, 30.0, 600.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(
        result, expected,
        "Transpose and broadcast combined mismatch"
    );
}

/// Test: Both inputs transposed
///
/// Tests that both inputs can have non-contiguous Views.
#[test]
fn test_both_inputs_transposed() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // a: [2, 3] -> transpose -> [3, 2]
    // b: [2, 3] -> transpose -> [3, 2]
    // c = a^T + b^T
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3]);
    let b = graph.input("b", DType::F32, vec![2, 3]);

    let a_t = a.view(a.view.clone().permute(vec![1, 0]));
    let b_t = b.view(b.view.clone().permute(vec![1, 0]));

    let c = a_t + b_t;
    graph.output("c", c);

    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    let program = pipeline
        .compile_program(graph)
        .expect("Failed to compile graph with both inputs transposed");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_b = pipeline.allocate_buffer(vec![2, 3], AstDType::F32).unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![3, 2], AstDType::F32).unwrap();

    // a: [[1, 2, 3], [4, 5, 6]] -> a^T: [[1, 4], [2, 5], [3, 6]]
    buf_a.write_vec(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    // b: [[10, 20, 30], [40, 50, 60]] -> b^T: [[10, 40], [20, 50], [30, 60]]
    buf_b
        .write_vec(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0])
        .unwrap();

    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a), ("b".to_string(), &buf_b)]
        .into_iter()
        .collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    let result: Vec<f32> = buf_c.read_vec().unwrap();

    // a^T + b^T = [[11, 44], [22, 55], [33, 66]]
    let expected: Vec<f32> = vec![11.0, 44.0, 22.0, 55.0, 33.0, 66.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "Both inputs transposed mismatch");
}

/// Test: 3D tensor transpose with reduce
///
/// Tests more complex multi-dimensional View operations.
#[test]
fn test_3d_transpose_reduce() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    // a: [2, 3, 4] -> permute(2, 0, 1) -> [4, 2, 3] -> reduce_sum(axis=2) -> [4, 2]
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![2, 3, 4]);

    let a_permuted = a.view(a.view.clone().permute(vec![2, 0, 1]));
    let sum = a_permuted.reduce_sum(2); // [4, 2, 3] -> [4, 2]
    graph.output("c", sum);

    let renderer = MetalRenderer::new();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device.clone());

    let program = pipeline
        .compile_program(graph)
        .expect("Failed to compile 3D transpose reduce");

    println!("Number of kernels: {}", program.kernel_count());

    // Allocate buffers
    let mut buf_a = pipeline
        .allocate_buffer(vec![2, 3, 4], AstDType::F32)
        .unwrap();
    let mut buf_c = pipeline.allocate_buffer(vec![4, 2], AstDType::F32).unwrap();

    // a[i,j,k] = i*12 + j*4 + k for simple verification
    // Shape [2, 3, 4]: 24 elements
    let data_a: Vec<f32> = (0..24).map(|x| x as f32).collect();
    buf_a.write_vec(&data_a).unwrap();

    let inputs: HashMap<String, _> = [("a".to_string(), &buf_a)].into_iter().collect();
    let mut outputs: HashMap<String, _> = [("c".to_string(), &mut buf_c)].into_iter().collect();

    program
        .execute(&device, &inputs, &mut outputs)
        .expect("Failed to execute");

    let result: Vec<f32> = buf_c.read_vec().unwrap();

    // After permute(2,0,1): a'[k,i,j] = a[i,j,k]
    // reduce_sum over axis=2 (j): sum_j a[i,j,k]
    // For k=0: [sum_j a[0,j,0], sum_j a[1,j,0]] = [0+4+8, 12+16+20] = [12, 48]
    // For k=1: [sum_j a[0,j,1], sum_j a[1,j,1]] = [1+5+9, 13+17+21] = [15, 51]
    // For k=2: [sum_j a[0,j,2], sum_j a[1,j,2]] = [2+6+10, 14+18+22] = [18, 54]
    // For k=3: [sum_j a[0,j,3], sum_j a[1,j,3]] = [3+7+11, 15+19+23] = [21, 57]
    let expected: Vec<f32> = vec![12.0, 48.0, 15.0, 51.0, 18.0, 54.0, 21.0, 57.0];
    println!("Result: {:?}", result);
    println!("Expected: {:?}", expected);
    assert_eq!(result, expected, "3D transpose reduce mismatch");
}
