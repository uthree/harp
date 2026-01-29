//! Basic example demonstrating Eclat tensor operations.

use eclat::prelude::*;

fn main() {
    // Initialize the CPU device
    eclat::init();

    println!("=== Eclat Basic Example ===\n");

    // Create tensors from arrays
    let x = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
    let y = Tensor::new([[5.0f32, 6.0], [7.0, 8.0]]);

    println!("x = {:?}", x);
    println!("y = {:?}", y);
    println!();

    // Basic arithmetic (lazy evaluation)
    let sum = &x + &y;
    let product = &x * &y;
    let diff = &x - &y;

    // Realize and print results
    println!("x + y = {:?}", sum.to_vec::<f32>());
    println!("x * y = {:?}", product.to_vec::<f32>());
    println!("x - y = {:?}", diff.to_vec::<f32>());
    println!();

    // Reduction operations
    let total = (&x + &y).sum(None, false);
    println!("sum(x + y) = {}", total.item::<f32>());

    // Partial reduction
    let row_sum = x.sum(Some(vec![1]), false);
    println!("row sums of x = {:?}", row_sum.to_vec::<f32>());

    let col_sum = x.sum(Some(vec![0]), false);
    println!("column sums of x = {:?}", col_sum.to_vec::<f32>());
    println!();

    // Reshape and transpose
    let flat = x.reshape([4]);
    println!("x flattened = {:?}", flat.to_vec::<f32>());

    let transposed = x.transpose();
    println!("x transposed shape = {:?}", transposed.shape());
    println!("x transposed = {:?}", transposed.to_vec::<f32>());
    println!();

    // Broadcasting
    let bias = Tensor::new([10.0f32, 20.0]);
    let with_bias = &x + &bias;
    println!("x + [10, 20] (broadcast) = {:?}", with_bias.to_vec::<f32>());
    println!();

    // Unary operations
    let a = Tensor::new([1.0f32, 2.0, 3.0, 4.0]);
    println!("a = {:?}", a.to_vec::<f32>());
    println!("sqrt(a) = {:?}", a.sqrt().to_vec::<f32>());
    println!("-a = {:?}", (-&a).to_vec::<f32>());
    println!();

    // Mean
    let mean = x.mean(None, false);
    println!("mean(x) = {}", mean.item::<f32>());

    println!("\n=== Example Complete ===");
}
