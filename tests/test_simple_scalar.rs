mod common;
use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::Backend;
use harp::graph::Graph;

#[test]
fn test_scalar_add() {
    common::setup();
    let mut backend = CBackend::new();
    if !backend.is_available() {
        return;
    }
    
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![3.into()]);
    let scalar = harp::graph::GraphNode::f32(2.0);
    let result = input + scalar;
    graph.output(result);
    
    let input_data = vec![1.0f32, 2.0, 3.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[3], DType::F32);
    
    let outputs = backend.execute(&graph, vec![input_buffer]);
    let output_data = outputs[0].to_vec::<f32>();
    
    let expected = vec![3.0f32, 4.0, 5.0];
    for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!((actual - expected).abs() < 1e-5, "Mismatch at index {}", i);
    }
}
