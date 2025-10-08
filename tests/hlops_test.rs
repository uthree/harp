#![cfg(feature = "backend-c")]

mod common;

use harp::ast::DType;
use harp::backend::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::Graph;

#[test]
fn test_natural_log() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Test log(e) = 1
    let input = graph.input(DType::F32, vec![3.into()]);
    let result = input.log();
    graph.output(result);

    // Input: [1.0, e, e^2]
    let e = std::f32::consts::E;
    let input_data = vec![1.0f32, e, e * e];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[3], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<f32>();

    // Expected: [0.0, 1.0, 2.0]
    let expected = vec![0.0f32, 1.0, 2.0];

    assert_eq!(outputs[0].shape(), &[3]);
    for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_natural_exp() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Test exp(x)
    let input = graph.input(DType::F32, vec![3.into()]);
    let result = input.exp();
    graph.output(result);

    // Input: [0.0, 1.0, 2.0]
    let input_data = vec![0.0f32, 1.0, 2.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[3], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<f32>();

    // Expected: [1.0, e, e^2]
    let e = std::f32::consts::E;
    let expected = vec![1.0f32, e, e * e];

    assert_eq!(outputs[0].shape(), &[3]);
    for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_cosine() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Test cos(x)
    let input = graph.input(DType::F32, vec![4.into()]);
    let result = input.cos();
    graph.output(result);

    // Input: [0.0, π/2, π, 2π]
    let pi = std::f32::consts::PI;
    let input_data = vec![0.0f32, pi / 2.0, pi, 2.0 * pi];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[4], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<f32>();

    // Expected: [1.0, 0.0, -1.0, 1.0]
    let expected = vec![1.0f32, 0.0, -1.0, 1.0];

    assert_eq!(outputs[0].shape(), &[4]);
    for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_log_exp_inverse() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Test exp(log(x)) = x
    let input = graph.input(DType::F32, vec![5.into()]);
    let result = input.log().exp();
    graph.output(result);

    let input_data = vec![1.0f32, 2.0, 5.0, 10.0, 100.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[5], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<f32>();

    assert_eq!(outputs[0].shape(), &[5]);
    for (i, (actual, expected)) in output_data.iter().zip(input_data.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_tangent() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Test tan(x)
    let input = graph.input(DType::F32, vec![4.into()]);
    let result = input.tan();
    graph.output(result);

    // Input: [0.0, π/4, -π/4, π/6]
    let pi = std::f32::consts::PI;
    let input_data = vec![0.0f32, pi / 4.0, -pi / 4.0, pi / 6.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[4], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<f32>();

    // Expected: [0.0, 1.0, -1.0, 1/√3]
    let sqrt_3 = 3.0f32.sqrt();
    let expected = vec![0.0f32, 1.0, -1.0, 1.0 / sqrt_3];

    assert_eq!(outputs[0].shape(), &[4]);
    for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}
