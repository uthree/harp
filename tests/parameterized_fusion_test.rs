mod common;

use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::ops::ReduceOps;
use harp::graph::Graph;
use rstest::rstest;

/// Test various tensor shapes for elementwise + reduce fusion
#[rstest]
#[case(vec![2, 3], 0, vec![3])]
#[case(vec![2, 3], 1, vec![2])]
#[case(vec![4, 5, 6], 0, vec![5, 6])]
#[case(vec![4, 5, 6], 1, vec![4, 6])]
#[case(vec![4, 5, 6], 2, vec![4, 5])]
#[case(vec![10], 0, vec![])]
fn test_elementwise_reduce_various_shapes(
    #[case] input_shape: Vec<usize>,
    #[case] reduce_axis: usize,
    #[case] expected_shape: Vec<usize>,
) {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let shape_exprs: Vec<_> = input_shape.iter().map(|&x| x.into()).collect();
    let a = graph.input(DType::F32, shape_exprs.clone());
    let b = graph.input(DType::F32, shape_exprs);

    // (a + b).sum(reduce_axis)
    let add = a + b;
    let result = add.sum(reduce_axis);

    graph.output(result);

    let total_size: usize = input_shape.iter().product();
    let a_data: Vec<f32> = (1..=total_size).map(|x| x as f32).collect();
    let b_data = vec![1.0f32; total_size];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &input_shape, DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &input_shape, DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    assert_eq!(outputs[0].shape(), expected_shape.as_slice());
}

/// Test various reduce operations with fusion
#[rstest]
#[case("sum")]
#[case("max")]
fn test_reduce_operations(#[case] op: &str) {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let b = graph.input(DType::F32, vec![3.into(), 4.into()]);

    // (a + b).reduce(op, axis=0)
    let add = a + b;
    let result = match op {
        "sum" => add.sum(0),
        "max" => add.max(0),
        _ => panic!("Unknown operation"),
    };

    graph.output(result);

    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b_data = vec![1.0f32; 12];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[3, 4], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[3, 4], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    assert_eq!(outputs[0].shape(), &[4]);

    let output_data = outputs[0].to_vec::<f32>();

    match op {
        "sum" => {
            // Expected: sum along axis 0 of (a + b)
            // Column 0: (1+1) + (5+1) + (9+1) = 2 + 6 + 10 = 18
            // Column 1: (2+1) + (6+1) + (10+1) = 3 + 7 + 11 = 21
            // Column 2: (3+1) + (7+1) + (11+1) = 4 + 8 + 12 = 24
            // Column 3: (4+1) + (8+1) + (12+1) = 5 + 9 + 13 = 27
            let expected = vec![18.0f32, 21.0, 24.0, 27.0];
            for (i, (out, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (out - exp).abs() < 1e-5,
                    "Mismatch at index {}: got {}, expected {}",
                    i,
                    out,
                    exp
                );
            }
        }
        "max" => {
            // Expected: max along axis 0 of (a + b)
            // Column 0: max(1+1, 5+1, 9+1) = max(2, 6, 10) = 10
            // Column 1: max(2+1, 6+1, 10+1) = max(3, 7, 11) = 11
            // Column 2: max(3+1, 7+1, 11+1) = max(4, 8, 12) = 12
            // Column 3: max(4+1, 8+1, 12+1) = max(5, 9, 13) = 13
            let expected = vec![10.0f32, 11.0, 12.0, 13.0];
            for (i, (out, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (out - exp).abs() < 1e-5,
                    "Mismatch at index {}: got {}, expected {}",
                    i,
                    out,
                    exp
                );
            }
        }
        _ => panic!("Unknown operation"),
    }
}

/// Test multi-reduce with various axis combinations
#[rstest]
#[case(vec![2, 3, 4], vec![0, 1], vec![4])]
#[case(vec![2, 3, 4], vec![0, 2], vec![3])]
#[case(vec![2, 3, 4], vec![1, 2], vec![2])]
#[case(vec![3, 4, 5, 6], vec![0, 1], vec![5, 6])]
#[case(vec![3, 4, 5, 6], vec![1, 3], vec![3, 5])]
fn test_multi_reduce_various_axes(
    #[case] input_shape: Vec<usize>,
    #[case] reduce_axes: Vec<usize>,
    #[case] expected_shape: Vec<usize>,
) {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let shape_exprs: Vec<_> = input_shape.iter().map(|&x| x.into()).collect();
    let input = graph.input(DType::F32, shape_exprs);

    // Apply multiple sum operations
    let mut result = input;
    for &axis in reduce_axes.iter().rev() {
        // Reverse order to maintain axis indices
        result = result.sum(axis);
    }

    graph.output(result);

    let total_size: usize = input_shape.iter().product();
    let input_data: Vec<f32> = (1..=total_size).map(|x| x as f32).collect();
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &input_shape, DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    assert_eq!(outputs[0].shape(), expected_shape.as_slice());
}

/// Test elementwise chain length
#[rstest]
#[case(2)] // a + b
#[case(3)] // (a + b) + c
#[case(4)] // ((a + b) + c) + d
#[case(5)] // (((a + b) + c) + d) + e
fn test_elementwise_chain_length(#[case] chain_length: usize) {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    // Create chain_length inputs
    let mut inputs = Vec::new();
    for _ in 0..chain_length {
        inputs.push(graph.input(DType::F32, vec![10.into()]));
    }

    // Chain them: (...((a + b) + c) + d)...
    let mut result = inputs[0].clone();
    for i in 1..chain_length {
        result = result + inputs[i].clone();
    }

    graph.output(result);

    // Create buffers - all with value 1.0
    let data = vec![1.0f32; 10];
    let buffers: Vec<_> = (0..chain_length)
        .map(|_| harp::backend::c::CBuffer::from_slice(&data, &[10], DType::F32))
        .collect();

    let outputs = backend.execute(&graph, buffers);

    // Expected: chain_length * 1.0 for each element
    let expected = vec![chain_length as f32; 10];
    let output_data = outputs[0].to_vec::<f32>();

    for (i, (out, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            out,
            exp
        );
    }
}

/// Test view transformations with various operations
#[rstest]
#[case("unsqueeze", vec![2, 3], 0, vec![1, 2, 3])]
#[case("unsqueeze", vec![2, 3], 1, vec![2, 1, 3])]
#[case("unsqueeze", vec![2, 3], 2, vec![2, 3, 1])]
fn test_view_transformations(
    #[case] op: &str,
    #[case] input_shape: Vec<usize>,
    #[case] axis: usize,
    #[case] expected_shape: Vec<usize>,
) {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let shape_exprs: Vec<_> = input_shape.iter().map(|&x| x.into()).collect();
    let input = graph.input(DType::F32, shape_exprs);

    let result = match op {
        "unsqueeze" => input.unsqueeze(axis),
        _ => panic!("Unknown operation"),
    };

    graph.output(result);

    let total_size: usize = input_shape.iter().product();
    let input_data: Vec<f32> = (1..=total_size).map(|x| x as f32).collect();
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &input_shape, DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    assert_eq!(outputs[0].shape(), expected_shape.as_slice());
}

/// Test matmul with various matrix sizes
#[rstest]
#[case(2, 3, 2)]
#[case(3, 4, 2)]
#[case(4, 5, 3)]
#[case(1, 10, 1)]
#[case(10, 1, 10)]
fn test_matmul_various_sizes(#[case] m: usize, #[case] k: usize, #[case] n: usize) {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    let a = graph.input(DType::F32, vec![m.into(), k.into()]);
    let b = graph.input(DType::F32, vec![k.into(), n.into()]);

    // Matrix multiplication using broadcast
    let a_expanded = a.unsqueeze(2).expand(vec![m.into(), k.into(), n.into()]);
    let b_expanded = b.unsqueeze(0).expand(vec![m.into(), k.into(), n.into()]);

    let multiplied = a_expanded * b_expanded;
    let result = multiplied.sum(1);

    graph.output(result);

    // Create simple test data
    let a_data: Vec<f32> = (1..=(m * k)).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=(k * n)).map(|x| x as f32).collect();

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[m, k], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[k, n], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    assert_eq!(outputs[0].shape(), &[m, n]);

    // Verify with manual matmul calculation
    let output_data = outputs[0].to_vec::<f32>();
    for i in 0..m {
        for j in 0..n {
            let mut expected = 0.0f32;
            for l in 0..k {
                expected += a_data[i * k + l] * b_data[l * n + j];
            }
            let actual = output_data[i * n + j];
            assert!(
                (actual - expected).abs() < 1e-4,
                "Mismatch at [{},{}]: got {}, expected {}",
                i,
                j,
                actual,
                expected
            );
        }
    }
}
