#![cfg(feature = "backend-c")]

mod common;

use harp::ast::DType;
use harp::backend::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::Graph;

#[test]
fn test_cast_f32_to_isize() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // F32からIsizeへのキャスト
    let input = graph.input(DType::F32, vec![4.into()]);
    let casted = input.cast(DType::Isize);
    graph.output(casted);

    // Input: [1.5, 2.7, -3.2, 4.9]
    let input_data = vec![1.5f32, 2.7, -3.2, 4.9];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[4], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    // Expected (truncated): [1, 2, -3, 4]
    let output_data = outputs[0].to_vec::<isize>();
    let expected = vec![1isize, 2, -3, 4];

    assert_eq!(outputs[0].shape(), &[4]);
    assert_eq!(output_data, expected);
}

#[test]
fn test_cast_isize_to_f32() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // IsizeからF32へのキャスト
    let input = graph.input(DType::Isize, vec![5.into()]);
    let casted = input.cast(DType::F32);
    graph.output(casted);

    // Input: [1, -2, 3, -4, 5]
    let input_data = vec![1isize, -2, 3, -4, 5];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[5], DType::Isize);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    // Expected: [1.0, -2.0, 3.0, -4.0, 5.0]
    let output_data = outputs[0].to_vec::<f32>();
    let expected = vec![1.0f32, -2.0, 3.0, -4.0, 5.0];

    assert_eq!(outputs[0].shape(), &[5]);
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
fn test_cast_same_type_noop() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // 同じ型へのキャスト（何もしない）
    let input = graph.input(DType::F32, vec![3.into()]);
    let casted = input.cast(DType::F32);
    graph.output(casted);

    let input_data = vec![1.0f32, 2.0, 3.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[3], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<f32>();

    assert_eq!(outputs[0].shape(), &[3]);
    assert_eq!(output_data, input_data);
}

#[test]
fn test_cast_with_operations() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // 演算後にキャスト
    let input = graph.input(DType::F32, vec![3.into()]);
    let doubled = input * harp::graph::GraphNode::f32(2.0);
    let casted = doubled.cast(DType::Isize);
    graph.output(casted);

    let input_data = vec![1.5f32, 2.3, 3.7];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[3], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<isize>();

    // Expected: [3, 4, 7] (1.5*2=3.0->3, 2.3*2=4.6->4, 3.7*2=7.4->7)
    let expected = vec![3isize, 4, 7];

    assert_eq!(outputs[0].shape(), &[3]);
    assert_eq!(output_data, expected);
}
