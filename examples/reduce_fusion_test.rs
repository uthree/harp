use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::Backend;
use harp::graph::ops::ReduceOps;
use harp::graph::Graph;

fn main() {
    env_logger::init();

    // Create backend
    let mut backend = CBackend::new();
    if !backend.is_available() {
        println!("C compiler not available");
        return;
    }

    // Create graph: sum((a + b) * c) along axis 0
    // This should fuse the elementwise operations with the reduce
    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let b = graph.input(DType::F32, vec![3.into(), 4.into()]);
    let c = graph.input(DType::F32, vec![3.into(), 4.into()]);

    // (a + b) * c then sum along axis 0
    let add = a + b;
    let mul = add * c;
    let result = mul.sum(0);

    graph.output(result);

    // Create input buffers
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ];
    let b_data = vec![
        1.0f32, 1.0, 1.0, 1.0, // row 0
        1.0, 1.0, 1.0, 1.0, // row 1
        1.0, 1.0, 1.0, 1.0, // row 2
    ];
    let c_data = vec![
        2.0f32, 2.0, 2.0, 2.0, // row 0
        2.0, 2.0, 2.0, 2.0, // row 1
        2.0, 2.0, 2.0, 2.0, // row 2
    ];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[3, 4], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[3, 4], DType::F32);
    let c_buffer = harp::backend::c::CBuffer::from_slice(&c_data, &[3, 4], DType::F32);

    // Execute
    println!("Computing sum((a + b) * c) along axis 0...");
    println!("Input a:");
    println!("  [1, 2, 3, 4]");
    println!("  [5, 6, 7, 8]");
    println!("  [9, 10, 11, 12]");
    println!("\nInput b:");
    println!("  [1, 1, 1, 1]");
    println!("  [1, 1, 1, 1]");
    println!("  [1, 1, 1, 1]");
    println!("\nInput c:");
    println!("  [2, 2, 2, 2]");
    println!("  [2, 2, 2, 2]");
    println!("  [2, 2, 2, 2]");

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer, c_buffer]);

    // Read result
    let output_data = outputs[0].to_vec::<f32>();
    println!("\nResult (sum along axis 0):");
    println!("{:?}", output_data);

    // Expected:
    // (1+1)*2 + (5+1)*2 + (9+1)*2 = 4 + 12 + 20 = 36
    // (2+1)*2 + (6+1)*2 + (10+1)*2 = 6 + 14 + 22 = 42
    // (3+1)*2 + (7+1)*2 + (11+1)*2 = 8 + 16 + 24 = 48
    // (4+1)*2 + (8+1)*2 + (12+1)*2 = 10 + 18 + 26 = 54
    println!("\nExpected:");
    println!("[36.0, 42.0, 48.0, 54.0]");

    // Verify
    let expected = vec![36.0f32, 42.0, 48.0, 54.0];
    let mut all_correct = true;
    for (i, (out, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
        if (out - exp).abs() > 1e-5 {
            println!("Mismatch at index {}: got {}, expected {}", i, out, exp);
            all_correct = false;
        }
    }

    if all_correct {
        println!("\n✓ All values match!");
    } else {
        println!("\n✗ Some values don't match");
    }
}
