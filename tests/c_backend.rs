// tests/c_backend.rs

use harp::ast::DType;
use harp::backend::c::{CBackend, CBuffer};
use harp::backend::{Backend, TryIntoNdarray};
use harp::graph::Graph;
use ndarray::ArrayD;
use std::ffi::c_void;

fn setup_logger() {
    // Initialize the logger for tests, ignoring errors if it's already set up
    let _ = env_logger::builder().is_test(true).try_init();
}

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

#[test]
fn test_cbackend_call_simple_add() {
    setup_logger();
    let backend = CBackend::new();

    // 1. Build Graph: c = a + b
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a + b).as_output();

    // 2. Prepare input data
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();

    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);

    // 3. Call the backend
    let inputs = vec![a_buffer, b_buffer];
    let mut result_buffers = backend.execute(&graph, inputs, vec![]);

    // 4. Verify results
    let output_id = graph.outputs.borrow()[0];
    let result_array = result_buffers
        .get_mut(&output_id)
        .unwrap()
        .try_into_ndarray::<f32>()
        .unwrap();

    let expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_multiple_outputs() {
    setup_logger();
    let backend = CBackend::new();

    // 1. Build Graph: c = a + b, d = a - b
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a + b).as_output();
    (a - b).as_output();

    // 2. Prepare input data
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();

    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);

    // 3. Call the backend
    let inputs = vec![a_buffer, b_buffer];
    let mut result_buffers = backend.execute(&graph, inputs, vec![]);

    // 4. Verify results
    assert!(result_buffers.contains_key(&graph.outputs.borrow()[0]));
    assert!(result_buffers.contains_key(&graph.outputs.borrow()[1]));

    // Verify the first output (c = a + b)
    let c_output_id = graph.outputs.borrow()[0];
    let c_result_array = result_buffers
        .get_mut(&c_output_id)
        .unwrap()
        .try_into_ndarray::<f32>()
        .unwrap();
    let c_expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let c_expected_array = ArrayD::from_shape_vec(shape.clone(), c_expected_data).unwrap();
    assert_eq!(c_result_array, c_expected_array);

    // Verify the second output (d = a - b)
    let d_output_id = graph.outputs.borrow()[1];
    let d_result_array = result_buffers
        .get_mut(&d_output_id)
        .unwrap()
        .try_into_ndarray::<f32>()
        .unwrap();
    let d_expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x - y)
        .collect();
    let d_expected_array = ArrayD::from_shape_vec(shape, d_expected_data).unwrap();
    assert_eq!(d_result_array, d_expected_array);
}

#[test]
#[ignore]
fn test_cbackend_cache() {
    setup_logger();
    let backend = CBackend::new();

    // Build Graph 1: c = a + b
    let graph1 = Graph::new();
    let shape = vec![10];
    let a1 = graph1.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b1 = graph1.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a1 + b1).as_output();

    // Prepare input data
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);

    // First run: should compile
    assert_eq!(*backend.compile_count.lock().unwrap(), 0);
    let _ = backend.execute(&graph1, vec![a_buffer.clone(), b_buffer.clone()], vec![]);
    assert_eq!(*backend.compile_count.lock().unwrap(), 1);

    // Second run with same graph: should use cache, no new compilation
    let _ = backend.execute(&graph1, vec![a_buffer.clone(), b_buffer.clone()], vec![]);
    assert_eq!(*backend.compile_count.lock().unwrap(), 2);

    // Build Graph 2 (different operation): d = a - b
    let graph2 = Graph::new();
    let a2 = graph2.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b2 = graph2.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a2 - b2).as_output();

    // Third run with different graph: should compile again
    let _ = backend.execute(&graph2, vec![a_buffer.clone(), b_buffer.clone()], vec![]);
    assert_eq!(*backend.compile_count.lock().unwrap(), 3);

    // Fourth run with graph2 again: should use cache
    let _ = backend.execute(&graph2, vec![a_buffer, b_buffer], vec![]);
    assert_eq!(*backend.compile_count.lock().unwrap(), 4);
}
