use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::Backend;
use harp::graph::Graph;

fn main() {
    env_logger::init();

    // Create backend
    let mut backend = CBackend::new();
    if !backend.is_available() {
        println!("C compiler not available");
        return;
    }

    // Create graph: flip a 2x3 matrix along axis 0 (rows)
    // Input: [[1, 2, 3],
    //         [4, 5, 6]]
    // After flip(0):
    // Output: [[4, 5, 6],
    //          [1, 2, 3]]
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![2.into(), 3.into()]);
    let flipped = input.flip(0);
    graph.output(flipped);

    // Create input buffer
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[2, 3], DType::F32);

    // Execute
    println!("Input (2x3 matrix):");
    println!("  [1, 2, 3]");
    println!("  [4, 5, 6]");
    println!("\nFlipping along axis 0 (rows)...");

    let outputs = backend.execute(&graph, vec![input_buffer]);

    // Read result
    let output_data = outputs[0].to_vec::<f32>();
    println!("\nOutput:");
    println!(
        "  [{}, {}, {}]",
        output_data[0], output_data[1], output_data[2]
    );
    println!(
        "  [{}, {}, {}]",
        output_data[3], output_data[4], output_data[5]
    );
    println!("\nExpected:");
    println!("  [4.0, 5.0, 6.0]");
    println!("  [1.0, 2.0, 3.0]");
}
