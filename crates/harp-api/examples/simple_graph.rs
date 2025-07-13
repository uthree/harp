use harp_ir::prelude::*;

fn main() {
    // Build a simple graph: (2.0 * 3.0) + 4.0
    let a = constant(2.0f32);
    let b = constant(3.0f32);
    let c = constant(4.0f32);

    let expr = (a * b) + c;

    // Convert the graph to DOT format
    let dot_string = expr.to_dot();

    // Print the DOT string
    println!("{}", dot_string);
}
