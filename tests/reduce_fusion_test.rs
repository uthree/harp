#![cfg(feature = "backend-c")]

mod common;

use harp::ast::DType;
use harp::backend::Backend;
use harp::backend::CBackend;
use harp::graph::ops::ReduceOps;
use harp::graph::Graph;

#[test]
fn test_elementwise_reduce_fusion() {
    common::setup();

    // Create backend
    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    // Create graph: sum((a + b) * c) along axis 0
    // This should fuse the elementwise operations with the reduce
    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let b = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let c = graph.input(DType::F32, vec![3.into(), 4.into()]);

    // (a + b) * c then sum along axis 0
    let add = a + b;
    let mul = add * c;
    let result = mul.sum(0);

    graph.output(result);

    // Create input buffers
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ];
    let b_data = vec![
        1.0f32, 1.0, 1.0, 1.0, // row 0
        1.0, 1.0, 1.0, 1.0, // row 1
        1.0, 1.0, 1.0, 1.0, // row 2
    ];
    let c_data = vec![
        2.0f32, 2.0, 2.0, 2.0, // row 0
        2.0, 2.0, 2.0, 2.0, // row 1
        2.0, 2.0, 2.0, 2.0, // row 2
    ];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[3, 4], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[3, 4], DType::F32);
    let c_buffer = harp::backend::c::CBuffer::from_slice(&c_data, &[3, 4], DType::F32);

    // Execute
    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer, c_buffer]);

    // Read result
    let output_data = outputs[0].to_vec::<f32>();

    // Expected:
    // (1+1)*2 + (5+1)*2 + (9+1)*2 = 4 + 12 + 20 = 36
    // (2+1)*2 + (6+1)*2 + (10+1)*2 = 6 + 14 + 22 = 42
    // (3+1)*2 + (7+1)*2 + (11+1)*2 = 8 + 16 + 24 = 48
    // (4+1)*2 + (8+1)*2 + (12+1)*2 = 10 + 18 + 26 = 54
    let expected = [36.0f32, 42.0, 48.0, 54.0];

    // Verify
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
