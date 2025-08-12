use harp::ast::DType;
use harp::tensor::Tensor;

fn main() {
    // Enable logging to see more details from the optimizer
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    // Define the computation graph outside the loop.
    let a = Tensor::rand(vec![512], DType::F32, true);
    let b = a.clone() * Tensor::full(vec![512], DType::F32, 1.0.into(), false); // Mul by 1
    let c = b + Tensor::full(vec![512], DType::F32, 0.0.into(), false); // Add 0

    // The backend is configured to trigger heuristic optimization on the 10th call by default.
    // We run the forward pass multiple times to trigger this.
    for i in 0..15 {
        println!("\n--- Iteration {} ---", i + 1);
        c.forward();
        // After the forward pass, we need to clear the buffer to re-run the computation.
        c.clear_buffer();
    }
}
