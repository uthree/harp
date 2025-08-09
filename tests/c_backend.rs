// tests/c_backend.rs

use harp::ast::DType;
use harp::backend::c::{CBackend, CBuffer};
use harp::backend::{Backend, Buffer, TryIntoNdarray};
use harp::graph::Graph;
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

#[test]
fn test_cbackend_call_simple_add() {
    harp::init_logger();
    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

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
    let inputs: Vec<Box<dyn Buffer>> = vec![Box::new(a_buffer) as _, Box::new(b_buffer) as _];
    let mut result_buffers = backend.call(graph, inputs, vec![]);

    // 4. Verify results
    let result_array = result_buffers
        .pop()
        .unwrap()
        .as_any_mut()
        .downcast_mut::<CBuffer>()
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
    harp::init_logger();
    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

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
    let inputs: Vec<Box<dyn Buffer>> = vec![Box::new(a_buffer) as _, Box::new(b_buffer) as _];
    let mut result_buffers = backend.call(graph, inputs, vec![]);

    // 4. Verify results
    assert_eq!(result_buffers.len(), 2);

    // Verify the first output (c = a + b)
    let c_result_array = result_buffers[0]
        .as_any_mut()
        .downcast_mut::<CBuffer>()
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
    let d_result_array = result_buffers[1]
        .as_any_mut()
        .downcast_mut::<CBuffer>()
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
