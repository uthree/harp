use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::Backend;
use harp::graph::Graph;

fn main() {
    env_logger::init();

    // Create backend
    let mut backend = CBackend::new();
    if !backend.is_available() {
        println!("C compiler not available");
        return;
    }

    // Create graph: ((a + b) * a) where a and b are inputs
    // This should fuse into a single kernel
    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![10.into()]);
    let b = graph.input(DType::F32, vec![10.into()]);

    // (a + b) * a - this will create a fusion opportunity
    // The add and mul should be fused into one loop
    let add = a.clone() + b;
    let result = add * a;

    graph.output(result);

    // Create input buffers
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[10], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[10], DType::F32);

    // Execute
    println!("Computing (a + b) * a...");
    println!("a = {:?}", a_data);
    println!("b = {:?}", b_data);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    // Read result
    let output_data = outputs[0].to_vec::<f32>();
    println!("\nResult:");
    println!("{:?}", output_data);

    // Expected: (1+1)*1=2, (2+1)*2=6, (3+1)*3=12, ...
    println!("\nExpected:");
    println!("[2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0]");

    // Verify
    let expected = vec![2.0f32, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0];
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
