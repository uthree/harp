mod common;

use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::Backend;
use harp::graph::ops::ReduceOps;
use harp::graph::Graph;

#[test]
fn test_multi_axis_reduce_fusion() {
    common::setup();

    // Create backend
    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available, skipping test");
        return;
    }

    // Create graph: sum(sum(input, axis=0), axis=0) on a 2x3x4 tensor
    // This should fuse two reduce operations into one FusedReduce with axes [0, 1]
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);

    // Sum along axis 0, then axis 0 again (which was axis 1 in the original)
    let sum1 = input.sum(0); // [2,3,4] -> [3,4]
    let sum2 = sum1.sum(0); // [3,4] -> [4]

    graph.output(sum2);

    // Create input buffer with values 1..24
    let input_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();

    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[2, 3, 4], DType::F32);

    // Execute
    let outputs = backend.execute(&graph, vec![input_buffer]);

    // Read result
    let output_data = outputs[0].to_vec::<f32>();

    // Expected: sum over first two axes
    // For each position in last axis, sum all values across first two dimensions
    // Position 0: sum of indices [0,4,8,12,16,20] = 1+5+9+13+17+21 = 66
    // Position 1: sum of indices [1,5,9,13,17,21] = 2+6+10+14+18+22 = 72
    // Position 2: sum of indices [2,6,10,14,18,22] = 3+7+11+15+19+23 = 78
    // Position 3: sum of indices [3,7,11,15,19,23] = 4+8+12+16+20+24 = 84
    let expected = vec![66.0f32, 72.0, 78.0, 84.0];

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
