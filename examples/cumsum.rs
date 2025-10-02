use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::Backend;
use harp::graph::ops::cumulative::CumulativeOps;
use harp::graph::Graph;

fn main() {
    env_logger::init();

    // Create backend
    let mut backend = CBackend::new();
    if !backend.is_available() {
        println!("C compiler not available");
        return;
    }

    // Create graph: cumsum([1, 2, 3, 4])
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![4.into()]);
    let cumsum = input.cumsum(0);
    graph.output(cumsum);

    // Create input buffer
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[4], DType::F32);

    // Execute
    println!("Input: [1, 2, 3, 4]");
    let outputs = backend.execute(&graph, vec![input_buffer]);

    // Read result
    let output_data = outputs[0].to_vec::<f32>();
    println!("Cumsum: {:?}", output_data);
    println!("Expected: [1.0, 3.0, 6.0, 10.0]");
}
