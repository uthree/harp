//! An example of using the `Rewriter` to perform constant folding.
//! 
//! This example defines a simple rule to fold constant additions (`a + b` -> `c`)
//! and applies it to a graph. It also demonstrates how to enable logging to see
//! the rewrite process in action.
//! 
//! To see the log output, run this example with the `RUST_LOG` environment variable:
//! 
//! ```sh
//! RUST_LOG=debug cargo run --example constant_folding
//! ```

use harp::node::{self, Const};
use harp::rewriter;

fn main() {
    // Initialize the logger.
    // You can use other loggers like `fern` or `slog` as well.
    env_logger::init();

    // 1. Define a rewriter for constant folding.
    let constant_folder = rewriter!([
        (
            let x = capture("x"), y = capture("y")
            => x + y
            => |x, y| {
                // Check if both captured nodes are f32 constants.
                if let (Some(const_x), Some(const_y)) = (
                    x.op().as_any().downcast_ref::<Const>(),
                    y.op().as_any().downcast_ref::<Const>(),
                ) {
                    if let (Some(val_x), Some(val_y)) = (
                        const_x.0.as_any().downcast_ref::<f32>(),
                        const_y.0.as_any().downcast_ref::<f32>(),
                    ) {
                        // If so, return a new constant node with their sum.
                        log::info!("Folding constants: {} + {} -> {}", val_x, val_y, val_x + val_y);
                        return Some(node::constant(val_x + val_y));
                    }
                }
                // Otherwise, do not rewrite.
                None
            }
        )
    ]);

    // 2. Build a graph: (2.0 * 3.0) + (4.0 + 5.0)
    // The rewriter should first fold (4.0 + 5.0) into 9.0.
    let graph = (node::constant(2.0f32) * 3.0) + (node::constant(4.0f32) + 5.0);

    println!("--- Original Graph ---");
    println!("{}", graph.to_dot());

    // 3. Apply the rewriter.
    let rewritten_graph = constant_folder.rewrite(graph);

    println!("\n--- Rewritten Graph ---");
    println!("{}", rewritten_graph.to_dot());
}
