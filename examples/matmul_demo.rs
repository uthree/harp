//! Matrix Multiplication Demo
//!
//! This example demonstrates matrix multiplication from graph construction
//! to lowering.
//!
//! Matrix multiplication: C[i,j] = Σ_k A[i,k] * B[k,j]
//!
//! Implementation strategy using primitives:
//! 1. A: [M, K] → unsqueeze(2) → [M, K, 1] → expand(2, N) → [M, K, N]
//! 2. B: [K, N] → permute([1,0]) → [N, K] → unsqueeze(0) → [1, K, N] → expand(0, M) → [M, K, N]
//! 3. Elementwise multiply: A * B → [M, K, N]
//! 4. Sum over axis 1 (K): [M, 1, N]
//! 5. Squeeze axis 1: [M, N]
//!
//! Run with: cargo run --example matmul_demo

use eclat::ast::{AstNode, DType};
use eclat::graph::{Expr, GraphNode, input};
use eclat::lowerer::Lowerer;

fn main() {
    println!("{}", "=".repeat(70));
    println!("Harp Matrix Multiplication Demo");
    println!("{}", "=".repeat(70));
    println!();

    // Matrix dimensions
    let m: i64 = 4; // rows of A, rows of C
    let k: i64 = 3; // cols of A, rows of B
    let n: i64 = 2; // cols of B, cols of C

    println!("Matrix dimensions:");
    println!("  A: [{}, {}]", m, k);
    println!("  B: [{}, {}]", k, n);
    println!("  C = A @ B: [{}, {}]", m, n);
    println!();

    // Demo: Graph construction and lowering
    demo_graph_lowering(m, k, n);
}

/// Build matrix multiplication graph using primitives
fn build_matmul_graph(a: &GraphNode, b: &GraphNode, m: i64, _k: i64, n: i64) -> GraphNode {
    // A: [M, K] → [M, K, 1] → [M, K, N]
    let a_expanded = a
        .unsqueeze(2) // [M, K, 1]
        .expand(2, Expr::Const(n)); // [M, K, N]

    // B: [K, N] → [N, K] → [1, N, K] → [1, K, N] → [M, K, N]
    let b_expanded = b
        .permute(&[1, 0]) // [N, K]
        .unsqueeze(0) // [1, N, K]
        .permute(&[0, 2, 1]) // [1, K, N]
        .expand(0, Expr::Const(m)); // [M, K, N]

    // Elementwise multiply and sum over K dimension
    let product = (&a_expanded * &b_expanded).with_name("product"); // [M, K, N]
    let summed = product.sum(1).with_name("summed"); // [M, 1, N]
    summed.squeeze(1).with_name("matmul_result") // [M, N]
}

/// Demo: Graph construction and lowering
fn demo_graph_lowering(m: i64, k: i64, n: i64) {
    println!("{}", "-".repeat(70));
    println!("Graph Construction and Lowering");
    println!("{}", "-".repeat(70));
    println!();

    // Create input tensors
    let a = input(vec![Expr::Const(m), Expr::Const(k)], DType::F32).with_name("A");
    let b = input(vec![Expr::Const(k), Expr::Const(n)], DType::F32).with_name("B");

    println!("Input tensors:");
    println!("  A: shape={:?}, dtype={:?}", a.shape(), a.dtype());
    println!("  B: shape={:?}, dtype={:?}", b.shape(), b.dtype());
    println!();

    // Build matmul graph
    println!("Building matmul graph...");
    println!();
    println!("Steps:");
    println!("  1. A.unsqueeze(2)           : [4,3] -> [4,3,1]");
    println!("  2. A.expand(2, N=2)         : [4,3,1] -> [4,3,2]");
    println!("  3. B.permute([1,0])         : [3,2] -> [2,3]");
    println!("  4. B.unsqueeze(0)           : [2,3] -> [1,2,3]");
    println!("  5. B.permute([0,2,1])       : [1,2,3] -> [1,3,2]");
    println!("  6. B.expand(0, M=4)         : [1,3,2] -> [4,3,2]");
    println!("  7. A * B (elementwise)      : [4,3,2]");
    println!("  8. sum(axis=1)              : [4,3,2] -> [4,1,2]");
    println!("  9. squeeze(1)               : [4,1,2] -> [4,2]");
    println!();

    let c = build_matmul_graph(&a, &b, m, k, n);

    println!("Result tensor:");
    println!("  C: shape={:?}, dtype={:?}", c.shape(), c.dtype());
    println!();

    // Lower to AST
    println!("Lowering to AST...");
    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[c]);

    println!("Generated Program:");
    print_program_structure(&program);
    println!();

    // Print expected computation for verification
    println!("{}", "-".repeat(70));
    println!("Expected Result (CPU reference)");
    println!("{}", "-".repeat(70));
    println!();

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]  (4x3)
    // B = [[1, 2], [3, 4], [5, 6]]  (3x2)
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();

    println!("Input A (4x3):");
    print_matrix(&a_data, 4, 3);
    println!();

    println!("Input B (3x2):");
    print_matrix(&b_data, 3, 2);
    println!();

    let expected = compute_matmul_cpu(&a_data, &b_data, 4, 3, 2);
    println!("Expected C = A @ B (4x2):");
    print_matrix(&expected, 4, 2);
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
                    _ => println!("    Function {}: (other)", i),
                }
            }
            println!("  Execution waves: {}", execution_waves.len());
        }
        _ => println!("  Not a Program node"),
    }
}

fn compute_matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    for i in 0..rows {
        print!("  [");
        for j in 0..cols {
            if j > 0 {
                print!(", ");
            }
            print!("{:6.1}", data[i * cols + j]);
        }
        println!("]");
    }
}
