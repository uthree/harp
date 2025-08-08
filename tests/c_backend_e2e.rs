// tests/c_backend_e2e.rs

use harp::ast::DType;
use harp::backend::c::{CBuffer, CCompiler, CRenderer};
use harp::backend::{Compiler, Kernel, Renderer, TryIntoNdarray};
use harp::graph::Graph;
use harp::graph::lowerer::{Lowerer, LoweringOrchestrator};
use ndarray::ArrayD;
use std::ffi::c_void;

/// Helper function to create a CBuffer from a slice of data.
fn buffer_from_slice<T: Clone>(data: &[T], shape: &[usize], dtype: DType) -> CBuffer {
    assert_eq!(
        data.len(),
        shape.iter().product(),
        "Data length must match the product of shape dimensions"
    );
    let byte_size = std::mem::size_of_val(data);
    let ptr = unsafe { libc::malloc(byte_size) };
    assert!(!ptr.is_null(), "Failed to allocate memory for buffer");
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, byte_size);
    }
    CBuffer {
        ptr: ptr as *mut c_void,
        shape: shape.to_vec(),
        dtype,
    }
}

/// Helper function to create an empty CBuffer for output.
fn empty_buffer(shape: &[usize], dtype: DType) -> CBuffer {
    let size: usize = shape.iter().product();
    let byte_size = size * dtype.size_in_bytes();
    let ptr = unsafe { libc::malloc(byte_size) };
    assert!(!ptr.is_null(), "Failed to allocate memory for buffer");
    CBuffer {
        ptr: ptr as *mut c_void,
        shape: shape.to_vec(),
        dtype,
    }
}

#[test]
fn test_c_backend_e2e_add() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    // 1. Build Graph: c = a + b
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a + b).as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();

    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);
    let c_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer, c_buffer], &[]);

    // 5. Verify results
    let c_result_array = result_buffers
        .pop()
        .unwrap()
        .try_into_ndarray::<f32>()
        .unwrap();

    let expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(c_result_array, expected_array);
}
