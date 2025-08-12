use harp::tensor::Tensor;
use harp::ast::DType;

fn main() {
    // Enable logging to see more details from the optimizer
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("--- Running End-to-End Tensor Optimization Demo ---");

    // Define a simple computation with Tensors.
    // The expression is equivalent to: (a * 2.0) + 0.0, where the optimizer
    // should be able to simplify the `* 1.0` part.
    let a = Tensor::rand(vec![3], DType::F32, true);
    let b = a.clone() * Tensor::full(vec![3], DType::F32, 2.0.into(), false);
    let c = b * Tensor::full(vec![3], DType::F32, 1.0.into(), false);
    let d = c + Tensor::full(vec![3], DType::F32, 0.0.into(), false);

    println!("\nDefined computation: (a * 2.0) * 1.0 + 0.0");
    println!("Initial Tensor 'a':\n{:?}", a);

    // The optimization and compilation happen lazily.
    // To trigger them, we need to execute the computation graph, for example by calling `forward()`.
    println!("\nTriggering forward pass to run optimization and compilation...");

    d.forward();

    println!("\n--- Optimization and Execution Complete ---");
    println!("Final Tensor 'd' (result of the computation):\n{:?}", d);
    println!("\nThe progress bar above should have shown the optimization steps.");
}

