mod common;

use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::ops::ReduceOps;
use harp::graph::Graph;

/// Test: Elementwise chain + Reduce fusion
/// Pattern: ((a + b) * c - d).sum(axis)
/// Should fuse all elementwise operations with the reduce
#[test]
fn test_complex_elementwise_reduce_fusion() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let b = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let c = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let d = graph.input(DType::F32, vec![3.into(), 4.into()]);

    // Complex elementwise chain followed by reduce
    // ((a + b) * c - d).sum(0)
    let add = a + b;
    let mul = add * c;
    let sub = mul - d;
    let result = sub.sum(0);

    graph.output(result);

    // Create test data
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b_data = vec![1.0f32; 12];
    let c_data = vec![2.0f32; 12];
    let d_data = vec![1.0f32; 12];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[3, 4], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[3, 4], DType::F32);
    let c_buffer = harp::backend::c::CBuffer::from_slice(&c_data, &[3, 4], DType::F32);
    let d_buffer = harp::backend::c::CBuffer::from_slice(&d_data, &[3, 4], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer, c_buffer, d_buffer]);

    // Verify result shape
    assert_eq!(outputs[0].shape(), &[4]);

    // Expected: for each column, sum of ((a[i] + b[i]) * c[i] - d[i])
    // Column 0: ((1+1)*2-1) + ((5+1)*2-1) + ((9+1)*2-1) = 3 + 11 + 19 = 33
    // Column 1: ((2+1)*2-1) + ((6+1)*2-1) + ((10+1)*2-1) = 5 + 13 + 21 = 39
    // Column 2: ((3+1)*2-1) + ((7+1)*2-1) + ((11+1)*2-1) = 7 + 15 + 23 = 45
    // Column 3: ((4+1)*2-1) + ((8+1)*2-1) + ((12+1)*2-1) = 9 + 17 + 25 = 51
    let expected = vec![33.0f32, 39.0, 45.0, 51.0];
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

/// Test: Multiple reduce operations fusion
/// Pattern: input.sum(0).sum(0) on 3D tensor
/// Should fuse into a single multi-axis reduce
#[test]
fn test_multi_axis_reduce_fusion_3d() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![3.into(), 4.into(), 5.into()]);

    // Sum along axis 0, then axis 0 again (original axis 1)
    let sum1 = input.sum(0); // [3,4,5] -> [4,5]
    let sum2 = sum1.sum(0); // [4,5] -> [5]

    graph.output(sum2);

    // Create input: 1..60
    let input_data: Vec<f32> = (1..=60).map(|x| x as f32).collect();
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[3, 4, 5], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    assert_eq!(outputs[0].shape(), &[5]);

    // Expected: sum over first two axes for each position in last axis
    // For index k in last axis: sum of all elements where last_index == k
    let output_data = outputs[0].to_vec::<f32>();

    // Manual calculation: sum all elements at positions [..., ..., k]
    // Indices 0,5,10,15,20,25,30,35,40,45,50,55 (k=0): sum = 330
    // Indices 1,6,11,16,21,26,31,36,41,46,51,56 (k=1): sum = 342
    // etc.
    let expected: Vec<f32> = (0..5)
        .map(|k| {
            (0..3)
                .flat_map(|i| (0..4).map(move |j| (i * 20 + j * 5 + k + 1) as f32))
                .sum()
        })
        .collect();

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

/// Test: View transformations + Elementwise + Reduce
/// Pattern: input.unsqueeze(3).expand(...).sum(axis)
/// Should optimize view chain and fuse with reduce
#[test]
fn test_view_elementwise_reduce_fusion() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![2.into(), 3.into()]);
    let b = graph.input(DType::F32, vec![2.into(), 3.into()]);

    // Apply view transformations before elementwise operation
    // a: [2, 3] -> unsqueeze(2) -> [2, 3, 1] -> expand -> [2, 3, 4]
    // b: [2, 3] -> unsqueeze(2) -> [2, 3, 1] -> expand -> [2, 3, 4]
    let a_expanded = a.unsqueeze(2).expand(vec![2.into(), 3.into(), 4.into()]);
    let b_expanded = b.unsqueeze(2).expand(vec![2.into(), 3.into(), 4.into()]);

    // Elementwise operation followed by reduce
    let mul = a_expanded * b_expanded;
    let result = mul.sum(2); // Sum along last axis -> [2, 3]

    graph.output(result);

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[2, 3], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[2, 3], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    assert_eq!(outputs[0].shape(), &[2, 3]);

    // Expected: each element is multiplied by 2, then summed 4 times
    // a[i,j] * b[i,j] * 4 = a[i,j] * 2 * 4 = a[i,j] * 8
    let expected: Vec<f32> = a_data.iter().map(|&x| x * 2.0 * 4.0).collect();
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

/// Test: Matrix multiplication using broadcast + reduce
/// Pattern: A[M,K] @ B[K,N] using expand and sum
#[test]
fn test_matmul_fusion() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();

    let m = 3isize;
    let k = 4isize;
    let n = 2isize;

    let a = graph.input(DType::F32, vec![m.into(), k.into()]);
    let b = graph.input(DType::F32, vec![k.into(), n.into()]);

    // Matrix multiplication: A[M,K] @ B[K,N] = C[M,N]
    // A: [M, K] -> [M, K, 1] -> [M, K, N]
    // B: [K, N] -> [1, K, N] -> [M, K, N]
    // Multiply: [M, K, N]
    // Sum over K: [M, N]
    let a_expanded = a.unsqueeze(2).expand(vec![m.into(), k.into(), n.into()]);
    let b_expanded = b.unsqueeze(0).expand(vec![m.into(), k.into(), n.into()]);

    let multiplied = a_expanded * b_expanded;
    let result = multiplied.sum(1); // Sum along K dimension

    graph.output(result);

    // A = [[1, 2, 3, 4],
    //      [5, 6, 7, 8],
    //      [9, 10, 11, 12]]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6],
    //      [7, 8]]
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let a_buffer =
        harp::backend::c::CBuffer::from_slice(&a_data, &[m as usize, k as usize], DType::F32);
    let b_buffer =
        harp::backend::c::CBuffer::from_slice(&b_data, &[k as usize, n as usize], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    assert_eq!(outputs[0].shape(), &[m as usize, n as usize]);

    // Expected result:
    // C[0,0] = 1*1 + 2*3 + 3*5 + 4*7 = 1 + 6 + 15 + 28 = 50
    // C[0,1] = 1*2 + 2*4 + 3*6 + 4*8 = 2 + 8 + 18 + 32 = 60
    // C[1,0] = 5*1 + 6*3 + 7*5 + 8*7 = 5 + 18 + 35 + 56 = 114
    // C[1,1] = 5*2 + 6*4 + 7*6 + 8*8 = 10 + 24 + 42 + 64 = 140
    // C[2,0] = 9*1 + 10*3 + 11*5 + 12*7 = 9 + 30 + 55 + 84 = 178
    // C[2,1] = 9*2 + 10*4 + 11*6 + 12*8 = 18 + 40 + 66 + 96 = 220
    let expected = vec![50.0f32, 60.0, 114.0, 140.0, 178.0, 220.0];
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

/// Test: Branching prevents fusion
/// Pattern: a + b is used in two places
/// Should NOT fuse because of branching
#[test]
fn test_branching_prevents_fusion() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![10.into()]);
    let b = graph.input(DType::F32, vec![10.into()]);

    // Create a branch: (a + b) is used in two places
    let add = a.clone() + b.clone();
    let c = graph.input(DType::F32, vec![10.into()]);
    let d = graph.input(DType::F32, vec![10.into()]);
    let mul1 = add.clone() * c; // First use
    let mul2 = add * d; // Second use (branching!)

    let result = mul1 + mul2; // (a+b)*2 + (a+b)*3 = (a+b)*5

    graph.output(result);

    let a_data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let b_data = vec![1.0f32; 10];
    let c_data = vec![2.0f32; 10];
    let d_data = vec![3.0f32; 10];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[10], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[10], DType::F32);
    let c_buffer = harp::backend::c::CBuffer::from_slice(&c_data, &[10], DType::F32);
    let d_buffer = harp::backend::c::CBuffer::from_slice(&d_data, &[10], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer, c_buffer, d_buffer]);

    // Expected: (a[i] + b[i]) * c[i] + (a[i] + b[i]) * d[i] = (a[i] + 1) * 2 + (a[i] + 1) * 3 = (a[i] + 1) * 5
    let expected: Vec<f32> = a_data.iter().map(|&x| (x + 1.0) * 5.0).collect();
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

/// Test: Reduce max operation fusion
/// Pattern: (a + b).max(axis)
#[test]
fn test_reduce_max_fusion() {
    common::setup();

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let b = graph.input(DType::F32, vec![3.into(), 4.into()]);

    // (a + b).max(0)
    let add = a + b;
    let result = add.max(0);

    graph.output(result);

    let a_data = vec![
        1.0f32, 5.0, 2.0, 8.0, 3.0, 1.0, 7.0, 2.0, 4.0, 9.0, 1.0, 3.0,
    ];
    let b_data = vec![
        2.0f32, 1.0, 3.0, 1.0, 1.0, 4.0, 2.0, 5.0, 1.0, 1.0, 6.0, 7.0,
    ];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[3, 4], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[3, 4], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    assert_eq!(outputs[0].shape(), &[4]);

    // Expected: max along axis 0 of (a + b)
    // Column 0: max(1+2, 3+1, 4+1) = max(3, 4, 5) = 5
    // Column 1: max(5+1, 1+4, 9+1) = max(6, 5, 10) = 10
    // Column 2: max(2+3, 7+2, 1+6) = max(5, 9, 7) = 9
    // Column 3: max(8+1, 2+5, 3+7) = max(9, 7, 10) = 10
    let expected = vec![5.0f32, 10.0, 9.0, 10.0];
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
