use harp::ast::DType;
use harp::tensor::Tensor;

fn main() {
    // Enable logging to see more details from the optimizer
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    for _ in 0..50 {
        // Define a simple computation with Tensors.
        let a = Tensor::rand(vec![512], DType::F32, true);
        let b = a.clone() * Tensor::full(vec![512], DType::F32, 2.0.into(), false);
        let c = b * Tensor::full(vec![512], DType::F32, 1.0.into(), false);
        let d = c + Tensor::full(vec![512], DType::F32, 0.0.into(), false);

        // The optimization and compilation happen lazily.
        // To trigger them, we need to execute the computation graph, for example by calling `forward()`.
        // The backend is configured to trigger heuristic optimization on the 3rd call.

        d.forward();
    }
}
