use harp::ast::DType;
use harp::backend::c::CBackend;
use harp::backend::generic::GenericBackendConfig;
use harp::tensor::Tensor;
use std::sync::Arc;

/// Performs matrix multiplication using a combination of elementary tensor operations.
/// C = A @ B
fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Get shapes
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    assert_eq!(
        k,
        b.shape()[0],
        "Matrix dimensions are not compatible for multiplication"
    );

    // 1. Reshape A to (M, K, 1) and B to (1, K, N)
    let a_reshaped = a.unsqueeze(2);
    let b_reshaped = b.unsqueeze(0);

    // 2. Expand both to (M, K, N) for element-wise multiplication
    let a_expanded = a_reshaped.expand(vec![m, k, n]);
    let b_expanded = b_reshaped.expand(vec![m, k, n]);

    // 3. Element-wise multiplication and sum over the K dimension (axis=1)
    (a_expanded * b_expanded).sum(1)
}

fn main() {
    // Enable logging to see more details from the optimizer

    env_logger::Builder::from_env(env_logger::Env::default()).init();

    // Configure the backend to trigger heuristic optimization on the first call.
    let config = GenericBackendConfig {
        heuristic_optimization_threshold: 1,
    };
    let backend = Arc::new(CBackend::with_config(config));

    // Define the computation graph outside the loop.
    let a = Tensor::rand(vec![64, 128], DType::F32, true, backend.clone());
    let b = Tensor::rand(vec![128, 64], DType::F32, true, backend.clone());
    let c = Tensor::rand(vec![64, 64], DType::F32, true, backend.clone());
    let d = Tensor::rand(vec![64, 64], DType::F32, true, backend.clone());
    let x = matmul(&a, &b);
    let x = matmul(&x, &c);
    let x = matmul(&x, &d);

    // The backend is configured to trigger heuristic optimization on the 1st call.
    // We run the forward pass multiple times to trigger this.
    for _i in 0..20 {
        x.forward();
        // After the forward pass, we need to clear the buffer to re-run the computation.
        //c.clear_buffer();
    }
}
