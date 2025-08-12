use harp::ast::DType;
use harp::tensor::Tensor;

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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Define the computation graph outside the loop.
    let a = Tensor::rand(vec![256, 512], DType::F32, true);
    let b = Tensor::rand(vec![512, 1024], DType::F32, true);
    let c = matmul(&a, &b);

    // The backend is configured to trigger heuristic optimization on the 10th call by default.
    // We run the forward pass multiple times to trigger this.
    for i in 0..15 {
        println!("\n--- Iteration {} ---", i + 1);
        c.forward();
        // After the forward pass, we need to clear the buffer to re-run the computation.
        c.clear_buffer();
    }
}
