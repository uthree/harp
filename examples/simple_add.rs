// Simple addition test to verify the pipeline works
use harp::ast::DType;
use harp::backend::c::CBuffer;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::Graph;
use harp::s;

fn main() {
    println!("=== Simple Addition Test ===\n");

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Error: C compiler not available");
        return;
    }

    let mut graph = Graph::new();

    // Create two inputs: [2, 3]
    let a = graph.input(DType::F32, s![2, 3]);
    let b = graph.input(DType::F32, s![2, 3]);

    // Simple addition
    let c = a + b;
    graph.output(c);

    // Prepare input data
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];

    println!("A: {:?}", a_data);
    println!("B: {:?}", b_data);

    // Create buffers
    let a_buffer = CBuffer::from_slice::<f32>(&a_data, &[2, 3], DType::F32);
    let b_buffer = CBuffer::from_slice::<f32>(&b_data, &[2, 3], DType::F32);

    // Execute
    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    let result: Vec<f32> = outputs[0].to_vec();
    println!("Result: {:?}", result);

    let expected = vec![11.0f32, 22.0, 33.0, 44.0, 55.0, 66.0];
    println!("Expected: {:?}", expected);

    if result == expected {
        println!("✓ Test passed!");
    } else {
        println!("✗ Test failed!");
    }
}
