mod common;

use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::Graph;

#[test]
fn test_contiguous_after_permute() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Create a 2x3 input tensor
    let input = graph.input(DType::F32, vec![2.into(), 3.into()]);

    // Permute dimensions: (2, 3) -> (3, 2)
    let permuted = input.permute(vec![1, 0]);

    // Make it contiguous (this should copy the data into contiguous memory)
    let contiguous = permuted.contiguous();

    graph.output(contiguous);

    // Input data: [[1, 2, 3], [4, 5, 6]]
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[2, 3], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    // After permute: [[1, 4], [2, 5], [3, 6]]
    // Expected output (contiguous): [1, 4, 2, 5, 3, 6]
    let output_data = outputs[0].to_vec::<f32>();
    let expected = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];

    assert_eq!(outputs[0].shape(), &[3, 2]);
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
fn test_contiguous_after_unsqueeze_expand() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Create a (2, 1) input tensor
    let input = graph.input(DType::F32, vec![2.into(), 1.into()]);

    // Unsqueeze to add a dimension: (2, 1) -> (2, 1, 1)
    let unsqueezed = input.unsqueeze(2);

    // Expand: (2, 1, 1) -> (2, 1, 3)
    let expanded = unsqueezed.expand(vec![2.into(), 1.into(), 3.into()]);

    // Make contiguous
    let contiguous = expanded.contiguous();

    graph.output(contiguous);

    // Input data: [[1], [2]]
    let input_data = vec![1.0f32, 2.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[2, 1], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    // After expand: [[[1, 1, 1]], [[2, 2, 2]]]
    // Expected output (contiguous): [1, 1, 1, 2, 2, 2]
    let output_data = outputs[0].to_vec::<f32>();
    let expected = vec![1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0];

    assert_eq!(outputs[0].shape(), &[2, 1, 3]);
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
fn test_contiguous_noop_on_already_contiguous() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Create a contiguous 3x4 input tensor
    let input = graph.input(DType::F32, vec![3.into(), 4.into()]);

    // Apply contiguous (should be a no-op in terms of data layout)
    let contiguous = input.contiguous();

    graph.output(contiguous);

    // Input data: sequential values
    let input_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[3, 4], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    // Output should be identical to input
    let output_data = outputs[0].to_vec::<f32>();

    assert_eq!(outputs[0].shape(), &[3, 4]);
    for (i, (actual, expected)) in output_data.iter().zip(input_data.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}
