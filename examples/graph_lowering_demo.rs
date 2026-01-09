//! Graph and Lowering Demo
//!
//! This example demonstrates how the computation graph (GraphNode) and
//! lowering (Lowerer) modules work together to convert high-level tensor
//! operations into executable AST kernels.
//!
//! Run with: cargo run --example graph_lowering_demo

use eclat::ast::{AstNode, DType};
use eclat::graph::{
    Expr, GraphNode, collect_inputs, count_nodes, graph_to_string, input, topological_sort,
};
use eclat::lowerer::Lowerer;

fn main() {
    println!("{}", "=".repeat(70));
    println!("Harp Graph & Lowering Demo");
    println!("{}", "=".repeat(70));
    println!();

    // Demo 1: Simple elementwise operation
    demo_elementwise();

    // Demo 2: Reduction operation
    demo_reduction();

    // Demo 3: Complex computation graph
    demo_complex_graph();

    // Demo 4: View transformations
    demo_view_transforms();
}

/// Demo 1: Simple elementwise addition
fn demo_elementwise() {
    println!("{}", "-".repeat(70));
    println!("Demo 1: Elementwise Addition");
    println!("{}", "-".repeat(70));
    println!();

    // Create input tensors
    let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
    let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");

    println!("Input tensors:");
    println!("  a: shape=[32, 64], dtype=F32");
    println!("  b: shape=[32, 64], dtype=F32");
    println!();

    // Build computation: c = a + b
    let c = (&a + &b).with_name("c");

    println!("Computation: c = a + b");
    println!();

    // Analyze graph
    print_graph_info(&[c.clone()]);

    // Lower to AST
    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[c]);

    println!("Lowered AST:");
    print_program_structure(&program);
    println!();
}

/// Demo 2: Reduction (sum along axis)
fn demo_reduction() {
    println!("{}", "-".repeat(70));
    println!("Demo 2: Reduction (Sum along axis 1)");
    println!("{}", "-".repeat(70));
    println!();

    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("x");

    println!("Input tensor:");
    println!("  x: shape=[32, 64], dtype=F32");
    println!();

    // Sum along axis 1: [32, 64] -> [32, 1]
    let y = x.sum(1).with_name("sum_result");

    println!("Computation: y = x.sum(axis=1)");
    println!("Output shape: [32, 1]");
    println!();

    print_graph_info(&[y.clone()]);

    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[y]);

    println!("Lowered AST:");
    print_program_structure(&program);
    println!();
}

/// Demo 3: Complex computation graph
fn demo_complex_graph() {
    println!("{}", "-".repeat(70));
    println!("Demo 3: Complex Computation Graph");
    println!("{}", "-".repeat(70));
    println!();

    // Create inputs
    let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
    let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");

    println!("Input tensors:");
    println!("  a: shape=[32, 64], dtype=F32");
    println!("  b: shape=[32, 64], dtype=F32");
    println!();

    // Complex computation: ((a + b) * a).sum(1)
    let sum_ab = (&a + &b).with_name("sum_ab");
    let mul_result = (&sum_ab * &a).with_name("mul_result");
    let final_result = mul_result.sum(1).with_name("final");

    println!("Computation steps:");
    println!("  1. sum_ab = a + b");
    println!("  2. mul_result = sum_ab * a");
    println!("  3. final = mul_result.sum(axis=1)");
    println!();

    print_graph_info(&[final_result.clone()]);

    // Show topological order
    println!("Topological order (execution order):");
    let sorted = topological_sort(&[final_result.clone()]);
    for (i, node) in sorted.iter().enumerate() {
        let name = node.name().unwrap_or("unnamed");
        let is_input = node.is_external();
        let kind = if is_input { "input" } else { "compute" };
        println!("  {}: {} ({})", i, name, kind);
    }
    println!();

    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[final_result]);

    println!("Lowered AST:");
    print_program_structure(&program);
    println!();
}

/// Demo 4: View transformations
fn demo_view_transforms() {
    println!("{}", "-".repeat(70));
    println!("Demo 4: View Transformations");
    println!("{}", "-".repeat(70));
    println!();

    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("x");

    println!("Input tensor:");
    println!("  x: shape=[32, 64], dtype=F32");
    println!();

    // Demonstrate various view transformations
    println!("View transformations:");

    // Transpose
    let transposed = x.permute(&[1, 0]).with_name("transposed");
    println!(
        "  transposed = x.permute([1, 0])  -> shape={:?}",
        transposed.shape()
    );

    // Unsqueeze
    let unsqueezed = x.unsqueeze(0).with_name("unsqueezed");
    println!(
        "  unsqueezed = x.unsqueeze(0)     -> shape={:?}",
        unsqueezed.shape()
    );

    // Reshape
    let reshaped = x.reshape(vec![Expr::Const(2048)]).with_name("reshaped");
    println!(
        "  reshaped = x.reshape([2048])    -> shape={:?}",
        reshaped.shape()
    );

    println!();

    // Lower a transpose operation
    println!("Lowering transpose operation:");
    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[transposed]);
    print_program_structure(&program);
    println!();
}

/// Print graph information
fn print_graph_info(roots: &[GraphNode]) {
    let node_count = count_nodes(roots);
    let inputs = collect_inputs(roots);

    println!("Graph information:");
    println!("  Total nodes: {}", node_count);
    println!("  Input nodes: {}", inputs.len());
    println!();

    println!("Graph structure:");
    let graph_str = graph_to_string(roots);
    for line in graph_str.lines() {
        println!("  {}", line);
    }
    println!();
}

/// Print program structure
fn print_program_structure(program: &AstNode) {
    match program {
        AstNode::Program {
            functions,
            execution_waves,
        } => {
            println!("  Program with {} kernel(s)", functions.len());
            for (i, func) in functions.iter().enumerate() {
                match func {
                    AstNode::Kernel {
                        name,
                        params,
                        return_type,
                        ..
                    } => {
                        let kernel_name = name.as_deref().unwrap_or("unnamed");
                        println!(
                            "    Kernel {}: name={}, params={}, return_type={:?}",
                            i,
                            kernel_name,
                            params.len(),
                            return_type
                        );
                    }
                    _ => println!("    Function {}: {:?}", i, func),
                }
            }
            println!("  Execution waves: {}", execution_waves.len());
        }
        _ => println!("  Not a Program: {:?}", program),
    }
}
