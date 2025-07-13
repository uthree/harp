//! # Rewrite Logging Example
//!
//! This example demonstrates how to enable and view the detailed logs of the
//! rewriting process in Harp. By initializing a logger and setting the
//! `RUST_LOG` environment variable, you can trace how the `Rewriter` applies
//! rules to simplify a computation graph.

use harp_ir::node::{self};
use harp_ir::simplify::default_rewriter;

fn main() {
    // Initialize the logger from the environment.
    // To see the debug logs, run this example with `RUST_LOG=debug`:
    //
    // RUST_LOG=debug cargo run --example rewrite_logging
    //
    env_logger::init();

    println!("--- Running Rewrite Logging Example ---");

    // 1. Build a complex computation graph.
    // This graph represents: (2.0 * 1.0) + (2.0 * 0.0)
    // It should be simplified down to just `2.0`.
    let a = || node::constant(2.0f32);
    let graph = (a() * node::constant(1.0f32)) + (a() * node::constant(0.0f32));

    println!("\nOriginal graph: {:?}", graph);

    // 2. Get the default rewriter.
    // This rewriter contains rules for algebraic simplification and constant folding.
    let rewriter = default_rewriter();

    println!("\nApplying rewriter '{}'...", rewriter.name);
    println!("(Set RUST_LOG=debug to see detailed rewrite steps)");

    // 3. Simplify the graph.
    // The rewrite engine will log each rule application at the DEBUG level.
    let simplified_graph = rewriter.rewrite(graph);

    println!("\nSimplified graph: {:?}", simplified_graph);
    println!("\n--- Example Finished ---");
}
