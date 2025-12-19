//! Integration tests for Pipeline
//!
//! These tests verify the end-to-end compilation and execution of graphs
//! using the native GPU backends (OpenCL and Metal).
//! All tests use strict assertions to ensure correctness.

#![cfg(any(feature = "opencl", all(feature = "metal", target_os = "macos")))]

use harp::graph::{DType, Graph};

// Larger epsilon to account for GPU floating-point precision differences
// Reduce operations and complex arithmetic can accumulate small errors
const EPSILON: f32 = 1e-3;

/// Create a simple add graph: out = a + b
fn create_simple_add_graph() -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![1024]);
    let b = graph.input("b", DType::F32, vec![1024]);
    let c = a + b;
    graph.output("out", c);
    graph
}

/// Create a scale graph: out = input * 2.0
fn create_scale_graph() -> Graph {
    let mut graph = Graph::new();
    let input = graph.input("input", DType::F32, vec![1024]);
    let result = input * 2.0;
    graph.output("out", result);
    graph
}

/// Create a more complex graph with multiple operations: out = (a + b) * a / b
fn create_complex_graph() -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![256]);
    let b = graph.input("b", DType::F32, vec![256]);
    let c = a.clone() + b.clone();
    let d = c * a;
    let e = d / b;
    graph.output("out", e);
    graph
}

/// Create a reduce sum graph: out = reduce_sum(a + b, axis=0)
fn create_reduce_sum_graph() -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![64, 32]);
    let b = graph.input("b", DType::F32, vec![64, 32]);
    let c = (a + b).reduce_sum(0);
    graph.output("out", c);
    graph
}

/// Helper function to check if two f32 values are approximately equal
fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

/// Helper function to verify all elements in two slices are approximately equal
fn verify_results(actual: &[f32], expected: &[f32], test_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: Output length mismatch: expected {}, got {}",
        test_name,
        expected.len(),
        actual.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e, EPSILON),
            "{}: Value mismatch at index {}: expected {}, got {} (diff: {})",
            test_name,
            i,
            e,
            a,
            (a - e).abs()
        );
    }
}

#[cfg(feature = "opencl")]
mod opencl_tests {
    use super::*;
    use harp::ast::DType as AstDType;
    use harp::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice, OpenCLRenderer};
    use harp::backend::{Buffer, Compiler, Device, Pipeline};

    fn get_device() -> OpenCLDevice {
        assert!(
            OpenCLDevice::is_available(),
            "OpenCL is not available on this system"
        );
        OpenCLDevice::new().expect("Failed to create OpenCL device")
    }

    #[test]
    fn test_pipeline_creation() {
        let context = get_device();
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let _pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, context);
    }

    #[test]
    fn test_pipeline_compile_simple_graph() {
        let context = get_device();
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_simple_add_graph();
        let kernel = pipeline
            .compile_graph(graph)
            .expect("Failed to compile simple add graph");

        // Verify signature
        assert!(
            !kernel.signature().inputs.is_empty(),
            "Kernel should have inputs"
        );
        assert!(
            !kernel.signature().outputs.is_empty(),
            "Kernel should have outputs"
        );
    }

    #[test]
    fn test_pipeline_buffer_allocation() {
        let context = get_device();

        let buffer = OpenCLBuffer::allocate(&context, vec![1024], AstDType::F32)
            .expect("Failed to allocate buffer");

        assert_eq!(buffer.shape(), &[1024]);
        assert_eq!(buffer.dtype(), AstDType::F32);
        assert_eq!(buffer.byte_len(), 1024 * 4);
    }

    #[test]
    fn test_pipeline_buffer_data_transfer() {
        let context = get_device();

        let mut buffer = OpenCLBuffer::allocate(&context, vec![16], AstDType::F32)
            .expect("Failed to allocate buffer");

        let input_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        buffer.write_vec(&input_data).expect("Failed to write data");

        let output_data: Vec<f32> = buffer.read_vec().expect("Failed to read data");

        verify_results(&output_data, &input_data, "buffer_data_transfer");
    }

    #[test]
    fn test_scale_kernel_execution() {
        let context = get_device();
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_scale_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile scale graph");

        // Allocate buffers
        let mut input_buffer = OpenCLBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate input buffer");
        let mut output_buffer = OpenCLBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data
        let input_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        input_buffer
            .write_vec(&input_data)
            .expect("Failed to write input data");

        // Execute kernel
        let inputs = [&input_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: input * 2.0
        let expected: Vec<f32> = input_data.iter().map(|x| x * 2.0).collect();
        verify_results(&output_data, &expected, "scale_kernel");
    }

    #[test]
    fn test_add_kernel_execution() {
        let context = get_device();
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_simple_add_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile add graph");

        // Allocate buffers
        let mut a_buffer = OpenCLBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate buffer a");
        let mut b_buffer = OpenCLBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate buffer b");
        let mut output_buffer = OpenCLBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data
        let a_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();
        a_buffer.write_vec(&a_data).expect("Failed to write data a");
        b_buffer.write_vec(&b_data).expect("Failed to write data b");

        // Execute kernel
        let inputs = [&a_buffer, &b_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: a + b
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a + b)
            .collect();
        verify_results(&output_data, &expected, "add_kernel");
    }

    #[test]
    fn test_complex_kernel_execution() {
        let context = get_device();
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_complex_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile complex graph");

        // Allocate buffers
        let mut a_buffer = OpenCLBuffer::allocate(&context_clone, vec![256], AstDType::F32)
            .expect("Failed to allocate buffer a");
        let mut b_buffer = OpenCLBuffer::allocate(&context_clone, vec![256], AstDType::F32)
            .expect("Failed to allocate buffer b");
        let mut output_buffer = OpenCLBuffer::allocate(&context_clone, vec![256], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data (avoid division by zero)
        let a_data: Vec<f32> = (0..256).map(|i| (i + 1) as f32).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i + 1) as f32 * 0.5).collect();
        a_buffer.write_vec(&a_data).expect("Failed to write data a");
        b_buffer.write_vec(&b_data).expect("Failed to write data b");

        // Execute kernel
        let inputs = [&a_buffer, &b_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: (a + b) * a / b
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| (a + b) * a / b)
            .collect();
        verify_results(&output_data, &expected, "complex_kernel");
    }

    #[test]
    fn test_reduce_sum_kernel_execution() {
        let context = get_device();
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_reduce_sum_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile reduce sum graph");

        // Allocate buffers (input: [64, 32], output: [32])
        let mut a_buffer = OpenCLBuffer::allocate(&context_clone, vec![64, 32], AstDType::F32)
            .expect("Failed to allocate buffer a");
        let mut b_buffer = OpenCLBuffer::allocate(&context_clone, vec![64, 32], AstDType::F32)
            .expect("Failed to allocate buffer b");
        let mut output_buffer = OpenCLBuffer::allocate(&context_clone, vec![32], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data
        let a_data: Vec<f32> = (0..(64 * 32)).map(|i| (i % 100) as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..(64 * 32))
            .map(|i| ((i + 50) % 100) as f32 * 0.1)
            .collect();
        a_buffer.write_vec(&a_data).expect("Failed to write data a");
        b_buffer.write_vec(&b_data).expect("Failed to write data b");

        // Execute kernel
        let inputs = [&a_buffer, &b_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: reduce_sum(a + b, axis=0)
        // For each column j, sum over all rows i: sum((a[i,j] + b[i,j]) for i in 0..64)
        let mut expected = vec![0.0f32; 32];
        for j in 0..32 {
            for i in 0..64 {
                let idx = i * 32 + j;
                expected[j] += a_data[idx] + b_data[idx];
            }
        }
        verify_results(&output_data, &expected, "reduce_sum_kernel");
    }

    #[test]
    fn test_pipeline_config() {
        let context = get_device();
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, context);

        // Modify config
        {
            let config = pipeline.config_mut();
            config.graph_beam_width = 2;
            config.ast_beam_width = 2;
            config.max_steps = 100;
            config.show_progress = false;
        }

        // Compile with modified config
        let graph = create_simple_add_graph();
        let kernel = pipeline
            .compile_graph(graph)
            .expect("Failed to compile with custom config");

        assert!(
            !kernel.signature().inputs.is_empty(),
            "Kernel should have inputs"
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_tests {
    use super::*;
    use harp::ast::DType as AstDType;
    use harp::backend::metal::{MetalBuffer, MetalCompiler, MetalDevice, MetalRenderer};
    use harp::backend::{Buffer, Compiler, Device, Pipeline};

    fn get_device() -> MetalDevice {
        assert!(
            MetalDevice::is_available(),
            "Metal is not available on this system"
        );
        MetalDevice::new().expect("Failed to create Metal device")
    }

    #[test]
    fn test_metal_pipeline_creation() {
        let context = get_device();
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();
        let _pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, context);
    }

    #[test]
    fn test_metal_pipeline_compile() {
        let context = get_device();
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_simple_add_graph();
        let kernel = pipeline
            .compile_graph(graph)
            .expect("Failed to compile simple add graph on Metal");

        assert!(
            !kernel.signature().inputs.is_empty(),
            "Kernel should have inputs"
        );
        assert!(
            !kernel.signature().outputs.is_empty(),
            "Kernel should have outputs"
        );
    }

    #[test]
    fn test_metal_buffer_operations() {
        let context = get_device();

        let mut buffer = MetalBuffer::allocate(&context, vec![256], AstDType::F32)
            .expect("Failed to allocate Metal buffer");

        assert_eq!(buffer.shape(), &[256]);
        assert_eq!(buffer.dtype(), AstDType::F32);

        let input_data: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
        buffer.write_vec(&input_data).expect("Failed to write data");

        let output_data: Vec<f32> = buffer.read_vec().expect("Failed to read data");
        verify_results(&output_data, &input_data, "metal_buffer_operations");
    }

    #[test]
    fn test_metal_scale_kernel_execution() {
        let context = get_device();
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_scale_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile scale graph on Metal");

        // Allocate buffers
        let mut input_buffer = MetalBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate input buffer");
        let mut output_buffer = MetalBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data
        let input_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        input_buffer
            .write_vec(&input_data)
            .expect("Failed to write input data");

        // Execute kernel
        let inputs = [&input_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: input * 2.0
        let expected: Vec<f32> = input_data.iter().map(|x| x * 2.0).collect();
        verify_results(&output_data, &expected, "metal_scale_kernel");
    }

    #[test]
    fn test_metal_add_kernel_execution() {
        let context = get_device();
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_simple_add_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile add graph on Metal");

        // Allocate buffers
        let mut a_buffer = MetalBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate buffer a");
        let mut b_buffer = MetalBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate buffer b");
        let mut output_buffer = MetalBuffer::allocate(&context_clone, vec![1024], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data
        let a_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();
        a_buffer.write_vec(&a_data).expect("Failed to write data a");
        b_buffer.write_vec(&b_data).expect("Failed to write data b");

        // Execute kernel
        let inputs = [&a_buffer, &b_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: a + b
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a + b)
            .collect();
        verify_results(&output_data, &expected, "metal_add_kernel");
    }

    #[test]
    fn test_metal_complex_kernel_execution() {
        let context = get_device();
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_complex_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile complex graph on Metal");

        // Allocate buffers
        let mut a_buffer = MetalBuffer::allocate(&context_clone, vec![256], AstDType::F32)
            .expect("Failed to allocate buffer a");
        let mut b_buffer = MetalBuffer::allocate(&context_clone, vec![256], AstDType::F32)
            .expect("Failed to allocate buffer b");
        let mut output_buffer = MetalBuffer::allocate(&context_clone, vec![256], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data (avoid division by zero)
        let a_data: Vec<f32> = (0..256).map(|i| (i + 1) as f32).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i + 1) as f32 * 0.5).collect();
        a_buffer.write_vec(&a_data).expect("Failed to write data a");
        b_buffer.write_vec(&b_data).expect("Failed to write data b");

        // Execute kernel
        let inputs = [&a_buffer, &b_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: (a + b) * a / b
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| (a + b) * a / b)
            .collect();
        verify_results(&output_data, &expected, "metal_complex_kernel");
    }

    #[test]
    fn test_metal_reduce_sum_kernel_execution() {
        let context = get_device();
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();
        let context_clone = context.clone();
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, context);

        let graph = create_reduce_sum_graph();
        let compiled = pipeline
            .compile_graph(graph)
            .expect("Failed to compile reduce sum graph on Metal");

        // Allocate buffers (input: [64, 32], output: [32])
        let mut a_buffer = MetalBuffer::allocate(&context_clone, vec![64, 32], AstDType::F32)
            .expect("Failed to allocate buffer a");
        let mut b_buffer = MetalBuffer::allocate(&context_clone, vec![64, 32], AstDType::F32)
            .expect("Failed to allocate buffer b");
        let mut output_buffer = MetalBuffer::allocate(&context_clone, vec![32], AstDType::F32)
            .expect("Failed to allocate output buffer");

        // Initialize input data
        let a_data: Vec<f32> = (0..(64 * 32)).map(|i| (i % 100) as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..(64 * 32))
            .map(|i| ((i + 50) % 100) as f32 * 0.1)
            .collect();
        a_buffer.write_vec(&a_data).expect("Failed to write data a");
        b_buffer.write_vec(&b_data).expect("Failed to write data b");

        // Execute kernel
        let inputs = [&a_buffer, &b_buffer];
        let outputs = &mut [&mut output_buffer];
        compiled
            .execute(&inputs, outputs)
            .expect("Failed to execute kernel");

        // Read and verify output
        let output_data: Vec<f32> = output_buffer.read_vec().expect("Failed to read output");

        // Expected: reduce_sum(a + b, axis=0)
        let mut expected = vec![0.0f32; 32];
        for j in 0..32 {
            for i in 0..64 {
                let idx = i * 32 + j;
                expected[j] += a_data[idx] + b_data[idx];
            }
        }
        verify_results(&output_data, &expected, "metal_reduce_sum_kernel");
    }
}
