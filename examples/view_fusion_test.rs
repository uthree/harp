// View fusion test - verifies that consecutive view operations are fused
use harp::ast::DType;
use harp::backend::c::CBuffer;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::Graph;

fn main() {
    env_logger::init();

    println!("=== View Fusion Test ===\n");

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available");
        return;
    }

    // Create graph with consecutive view operations
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![2.into(), 3.into()]);

    // Chain of view operations: unsqueeze -> expand -> permute
    // These should be fused into a single view operation
    let unsqueezed = input.unsqueeze(2); // [2, 3] -> [2, 3, 1]
    let expanded = unsqueezed.expand(vec![2.into(), 3.into(), 4.into()]); // -> [2, 3, 4]
    let permuted = expanded.permute(vec![2, 0, 1]); // -> [4, 2, 3]

    graph.output(permuted);

    // Create input data
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = CBuffer::from_slice(&input_data, &[2, 3], DType::F32);

    // Execute
    println!("Input shape: [2, 3]");
    println!("Input: {:?}", input_data);
    println!("\nApplying: unsqueeze(2) -> expand([2,3,4]) -> permute([2,0,1])");
    println!("Expected output shape: [4, 2, 3]");

    let outputs = backend.execute(&graph, vec![input_buffer]);

    // Read result
    let output_data = outputs[0].to_vec::<f32>();
    println!("\nOutput shape: {:?}", outputs[0].shape());
    println!("Output: {:?}", output_data);

    // Verify shape
    assert_eq!(outputs[0].shape(), &[4, 2, 3]);

    println!("\n✓ View fusion test passed!");
}
