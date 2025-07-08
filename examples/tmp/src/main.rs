use harp::{
    graph::{
        dtype::{DType, Scalar},
        graph::Graph,
    },
    ir::compiler::Compiler,
    shape::tracker::ShapeTracker,
};
use std::sync::{Arc, Mutex};

fn main() {
    // 1. Create a new computation graph
    let graph_arc = Arc::new(Mutex::new(Graph::new()));

    // 2. Define the shape for the input tensors
    let shape: ShapeTracker = vec![2, 3].into();

    // 3. Create two input tensors and a constant
    let a = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let b = Graph::new_input(graph_arc.clone(), shape.clone(), DType::F32);
    let c = Graph::new_const(graph_arc.clone(), Scalar::F32(2.0), shape);

    // 4. Perform some operations
    let d = &a + &b;
    let e = &d * &c;

    // 5. Register the final tensor as a graph output
    Graph::add_output_node(graph_arc.clone(), &e);

    // 6. Compile the graph into an IR function
    let mut compiler = Compiler::new();
    let graph_locked = graph_arc.lock().unwrap();
    let ir_function = compiler.compile(&graph_locked, "my_cool_function");

    // 7. Print the generated IR to the console
    println!("--- Generated IR ---");
    println!("{}", ir_function);
}
