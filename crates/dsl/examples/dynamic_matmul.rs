//! Dynamic Shape Matrix Multiplication Example
//!
//! This example demonstrates how to define matrix multiplication with
//! dynamic (symbolic) dimensions using the Eclat DSL.
//!
//! The matrix multiplication C = A @ B is implemented as:
//! 1. Unsqueeze A from [M, K] to [M, K, 1]
//! 2. Unsqueeze B from [K, N] to [1, K, N]
//! 3. Expand both to [M, K, N]
//! 4. Element-wise multiply
//! 5. Sum along K axis

use eclat_dsl::{parse_program, GraphBuilder};

fn main() {
    println!("=== Dynamic Shape Matrix Multiplication ===\n");

    // Define matmul with dynamic dimensions M, K, N
    let dsl_source = r#"
program {
    // Matrix multiplication with dynamic shapes
    // A: [M, K], B: [K, N] -> C: [M, N]
    graph<M, K, N> matmul(a: f32[M, K], b: f32[K, N]) -> f32[M, N] {
        // Expand A: [M, K] -> [M, K, 1] -> [M, K, N]
        let a_expanded = expand(unsqueeze(a, axis=2), [M, K, N]);
        
        // Expand B: [K, N] -> [1, K, N] -> [M, K, N]
        let b_expanded = expand(unsqueeze(b, axis=0), [M, K, N]);
        
        // Element-wise multiply and sum along K axis
        let product = a_expanded * b_expanded;
        let result = sum(product, axis=1);
        
        return squeeze(result, axis=1);
    }
}
"#;

    // Parse the DSL
    println!("Parsing DSL...");
    let program = match parse_program(dsl_source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

    println!("Parsed {} graph(s)\n", program.graphs.len());

    // Build the graph
    println!("Building computation graph...");
    let mut builder = GraphBuilder::new();
    let graphs = match builder.build_program(&program) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Build error: {}", e);
            return;
        }
    };

    // Print graph information
    for graph in &graphs {
        println!("Graph: {}", graph.name);
        println!("  Shape variables: {:?}", graph.shape_vars);
        println!("  Inputs:");
        for (name, node) in &graph.inputs {
            println!("    {}: shape={:?}, dtype={:?}", name, node.shape(), node.dtype());
        }
        println!("  Output: shape={:?}, dtype={:?}", graph.output.shape(), graph.output.dtype());
    }

    println!();

    // Lower to AST to verify
    println!("Lowering to AST...");
    let mut lowerer = eclat::lowerer::Lowerer::new();
    match lowerer.lower(&[graphs[0].output.clone()]) {
        Ok(ast) => {
            if let eclat::ast::AstNode::Program { functions, .. } = &ast {
                println!("Generated {} kernel(s)", functions.len());
                
                // Print kernel info
                for (i, func) in functions.iter().enumerate() {
                    if let eclat::ast::AstNode::Kernel { name, params, .. } = func {
                        println!("\nKernel {}: {:?}", i, name);
                        println!("  Parameters:");
                        for param in params {
                            println!("    {}: {:?}", param.name, param.dtype);
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Lowering error: {}", e);
        }
    }

    println!("\n=== Batched Matrix Multiplication ===\n");

    // Batched matmul example
    let batched_source = r#"
program {
    // Batched matrix multiplication
    // A: [batch, M, K], B: [batch, K, N] -> C: [batch, M, N]
    graph<batch, M, K, N> batched_matmul(a: f32[batch, M, K], b: f32[batch, K, N]) -> f32[batch, M, N] {
        // For each batch:
        // A[b]: [M, K] -> [M, K, 1] -> [M, K, N]
        // B[b]: [K, N] -> [1, K, N] -> [M, K, N]
        
        // Expand A: [batch, M, K] -> [batch, M, K, 1] -> [batch, M, K, N]
        let a_expanded = expand(unsqueeze(a, axis=3), [batch, M, K, N]);
        
        // Expand B: [batch, K, N] -> [batch, 1, K, N] -> [batch, M, K, N]
        let b_expanded = expand(unsqueeze(b, axis=1), [batch, M, K, N]);
        
        // Element-wise multiply and sum along K axis (axis=2)
        let product = a_expanded * b_expanded;
        let result = sum(product, axis=2);
        
        return squeeze(result, axis=2);
    }
}
"#;

    println!("Parsing batched matmul DSL...");
    let batched_program = match parse_program(batched_source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

    let mut builder = GraphBuilder::new();
    let batched_graphs = match builder.build_program(&batched_program) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Build error: {}", e);
            return;
        }
    };

    for graph in &batched_graphs {
        println!("Graph: {}", graph.name);
        println!("  Shape variables: {:?}", graph.shape_vars);
        println!("  Inputs:");
        for (name, node) in &graph.inputs {
            println!("    {}: shape={:?}", name, node.shape());
        }
        println!("  Output: shape={:?}", graph.output.shape());
    }

    println!("\n=== Done ===");
}
