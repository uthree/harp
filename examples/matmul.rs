// Matrix multiplication example using Harp
// Computes C = A @ B where A is [M, K] and B is [K, N]
// Using expand + elementwise multiplication + reduction

use harp::ast::DType;
use harp::backend::c::CBuffer;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::{Graph, ReduceOps};
use harp::s;

fn main() {
    let _ = env_logger::try_init();

    println!("=== Harp Matrix Multiplication Demo ===\n");

    // Initialize the C backend
    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Error: C compiler not available");
        eprintln!("Please ensure gcc or clang is installed");
        return;
    }

    // Create a graph
    let mut graph = Graph::new();

    // Define matrix dimensions: A[2, 3] @ B[3, 2] = C[2, 2]
    let m = 2isize;
    let k = 3isize;
    let n = 2isize;

    println!("Computing C = A @ B");
    println!("A: [{}, {}]", m, k);
    println!("B: [{}, {}]", k, n);
    println!("C: [{}, {}]\n", m, n);

    // Create input nodes
    let a = graph.input(DType::F32, s![m, k]); // [2, 3]
    let b = graph.input(DType::F32, s![k, n]); // [3, 2]

    // Matrix multiplication using broadcast + multiply + sum
    // A: [M, K] -> [M, K, 1]
    // B: [K, N] -> [1, K, N]
    // Multiply: [M, K, N]
    // Sum over axis 1: [M, N]

    let a_expanded = a.unsqueeze(2).expand(s![m, k, n]); // [2, 3] -> [2, 3, 1] -> [2, 3, 2]
    let b_expanded = b.unsqueeze(0).expand(s![m, k, n]); // [3, 2] -> [1, 3, 2] -> [2, 3, 2]

    let multiplied = a_expanded * b_expanded; // [2, 3, 2]
    let result = multiplied.sum(1); // [2, 2]

    graph.output(result);

    // Prepare input data
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // B = [[7,  8],
    //      [9, 10],
    //      [11, 12]]
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    println!("Matrix A:");
    println!("  [[{}, {}, {}],", a_data[0], a_data[1], a_data[2]);
    println!("   [{}, {}, {}]]", a_data[3], a_data[4], a_data[5]);
    println!();

    println!("Matrix B:");
    println!("  [[{}, {}],", b_data[0], b_data[1]);
    println!("   [{}, {}],", b_data[2], b_data[3]);
    println!("   [{}, {}]]", b_data[4], b_data[5]);
    println!();

    // Create buffers
    let a_buffer = CBuffer::from_slice::<f32>(&a_data, &[m as usize, k as usize], DType::F32);
    let b_buffer = CBuffer::from_slice::<f32>(&b_data, &[k as usize, n as usize], DType::F32);

    // Execute the graph
    println!("Executing graph...");
    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    // Get the result
    assert_eq!(outputs.len(), 1);
    let c = &outputs[0];

    println!("Result shape: {:?}", c.shape());

    let c_data: Vec<f32> = c.to_vec();

    println!("\nMatrix C = A @ B:");
    println!("  [[{}, {}],", c_data[0], c_data[1]);
    println!("   [{}, {}]]", c_data[2], c_data[3]);
    println!();

    // Verify the result
    // Expected: [[1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12],
    //            [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]]
    //         = [[7 + 18 + 33,  8 + 20 + 36],
    //            [28 + 45 + 66,  32 + 50 + 72]]
    //         = [[58, 64],
    //            [139, 154]]
    let expected = vec![58.0, 64.0, 139.0, 154.0];

    println!("Expected result:");
    println!("  [[{}, {}],", expected[0], expected[1]);
    println!("   [{}, {}]]", expected[2], expected[3]);
    println!();

    // Check if the result is correct
    let epsilon = 1e-5;
    let is_correct = c_data
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < epsilon);

    if is_correct {
        println!("✓ Result is correct!");
    } else {
        println!("✗ Result is incorrect!");
        println!("Differences:");
        for i in 0..c_data.len() {
            println!("  c[{}]: got {}, expected {}", i, c_data[i], expected[i]);
        }
    }
}
