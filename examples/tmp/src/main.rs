use harp::{dtype, graph::Graph, shape::tracker::ShapeTracker};
use std::sync::{Arc, Mutex};

fn main() {
    // 1. Create a new computation graph
    let graph = Arc::new(Mutex::new(Graph::new()));

    // 2. Define the shape for the input tensors
    let shape: ShapeTracker = vec![2, 3].into();

    // 3. Create two input tensors
    let a = Graph::new_input(graph.clone(), shape.clone(), dtype::DType::F32);
    let b = Graph::new_input(graph.clone(), shape.clone(), dtype::DType::F32);

    // 4. Perform an addition operation
    let c = &a + &b;

    // 5. Perform an exponentiation operation
    let d = c.exp2();

    // 6. Cast the result to a different data type (e.g., I32)
    let e = d.cast(dtype::DType::I32);

    // 7. Register the final tensor as a graph output
    Graph::add_output_node(graph.clone(), &e);

    // 8. Lock the graph and generate the DOT representation
    let graph_locked = graph.lock().unwrap();
    let dot_output = graph_locked.to_dot();

    // 9. Print the DOT output to the console
    println!("{}", dot_output);
}
