use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::{Backend, Buffer};
use harp::graph::ops::ReduceOps;
use harp::graph::Graph;

fn main() {
    env_logger::init();

    let m = 10usize;
    let k = 1usize;
    let n = 10usize;

    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("C compiler not available");
        return;
    }

    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![m.into(), k.into()]);
    let b = graph.input(DType::F32, vec![k.into(), n.into()]);

    let a_expanded = a.unsqueeze(2).expand(vec![m.into(), k.into(), n.into()]);
    let b_expanded = b.unsqueeze(0).expand(vec![m.into(), k.into(), n.into()]);

    let multiplied = a_expanded * b_expanded;
    let result = multiplied.sum(1);

    graph.output(result);

    let a_data: Vec<f32> = (1..=(m * k)).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=(k * n)).map(|x| x as f32).collect();

    let a_buffer = harp::backend::c::CBuffer::from_slice(&a_data, &[m, k], DType::F32);
    let b_buffer = harp::backend::c::CBuffer::from_slice(&b_data, &[k, n], DType::F32);

    let outputs = backend.execute(&graph, vec![a_buffer, b_buffer]);

    println!("\nOutput shape: {:?}", outputs[0].shape());
    let output_data = outputs[0].to_vec::<f32>();

    // Expected matmul result
    let mut all_ok = true;
    for i in 0..m {
        for j in 0..n {
            let mut expected = 0.0f32;
            for l in 0..k {
                expected += a_data[i * k + l] * b_data[l * n + j];
            }
            let actual = output_data[i * n + j];
            if (actual - expected).abs() > 1e-4 {
                eprintln!(
                    "MISMATCH at [{},{}]: actual={}, expected={}",
                    i, j, actual, expected
                );
                all_ok = false;
            }
        }
    }

    if all_ok {
        println!("✓ All values correct!");
    } else {
        println!("✗ Some values incorrect!");
    }
}
