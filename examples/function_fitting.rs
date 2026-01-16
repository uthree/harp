//! Function Fitting Demo
//!
//! This example demonstrates training a simple model to fit a linear function
//! y = 2*x + 1 using gradient descent.
//!
//! Run with: cargo run --example function_fitting

// Link backend crates to trigger ctor auto-initialization
#[cfg(target_os = "macos")]
extern crate eclat_backend_metal;
extern crate eclat_backend_c;

use eclat::backend::{get_default_device_kind, set_device_str};
use eclat::tensor::Tensor;
use eclat::tensor::dim::D1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Function Fitting Demo ===\n");
    println!("Learning: y = 2*x + 1\n");

    // Auto-select the best available backend (backends are auto-initialized via ctor)
    set_device_str("auto")?;
    println!("Device: {}\n", get_default_device_kind());

    // ========================================================================
    // Generate training data: y = 2*x + 1
    // ========================================================================
    let n_samples = 8;
    let x_data: Vec<f32> = (0..n_samples).map(|i| i as f32 * 0.5).collect();
    let y_target: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    println!("Training data:");
    println!("  x = {:?}", x_data);
    println!("  y = {:?}\n", y_target);

    // ========================================================================
    // Initialize learnable parameters
    // ========================================================================
    // Model: y_pred = w * x + b
    // We want to learn w ≈ 2.0 and b ≈ 1.0

    let mut w_val = 0.5f32;
    let mut b_val = 0.0f32;

    println!("Initial parameters:");
    println!("  w = {:.4}", w_val);
    println!("  b = {:.4}\n", b_val);

    // ========================================================================
    // Training loop (CPU-based gradient descent)
    // ========================================================================
    let learning_rate = 0.01f32;
    let epochs = 100;

    println!(
        "Training with learning_rate = {}, epochs = {}\n",
        learning_rate, epochs
    );

    for epoch in 0..epochs {
        // Forward pass: y_pred = w * x + b
        let y_pred_data: Vec<f32> = x_data.iter().map(|&xi| w_val * xi + b_val).collect();

        // Compute loss: MSE = sum((y_pred - y)^2)
        let diff_data: Vec<f32> = y_pred_data
            .iter()
            .zip(y_target.iter())
            .map(|(&p, &t)| p - t)
            .collect();

        let loss_val: f32 = diff_data.iter().map(|&d| d * d).sum();

        // Compute gradients
        // d(loss)/dw = sum(2 * (y_pred - y) * x)
        // d(loss)/db = sum(2 * (y_pred - y))
        let grad_w: f32 = diff_data
            .iter()
            .zip(x_data.iter())
            .map(|(&d, &xi)| 2.0 * d * xi)
            .sum();
        let grad_b: f32 = diff_data.iter().map(|&d| 2.0 * d).sum();

        // Update parameters
        w_val -= learning_rate * grad_w;
        b_val -= learning_rate * grad_b;

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {:3}: loss = {:.6}, w = {:.4}, b = {:.4}",
                epoch, loss_val, w_val, b_val
            );
        }
    }

    // ========================================================================
    // GPU-accelerated forward pass with learned parameters
    // ========================================================================
    println!("\n=== GPU Forward Pass Demo ===\n");

    let x_tensor: Tensor<D1, f32> = Tensor::input([n_samples]);
    let w_tensor: Tensor<D1, f32> = Tensor::input([n_samples]);
    let b_tensor: Tensor<D1, f32> = Tensor::input([n_samples]);

    x_tensor.set_data(&x_data)?;
    w_tensor.set_data(&vec![w_val; n_samples])?;
    b_tensor.set_data(&vec![b_val; n_samples])?;

    // GPU computation: y_pred = w * x + b
    let y_pred = &(&w_tensor * &x_tensor) + &b_tensor;
    y_pred.realize()?;

    let predictions = y_pred.to_vec()?;

    // ========================================================================
    // Final results
    // ========================================================================
    println!("=== Training Complete ===\n");
    println!("Learned parameters:");
    println!("  w = {:.4} (target: 2.0)", w_val);
    println!("  b = {:.4} (target: 1.0)", b_val);

    let final_loss: f32 = predictions
        .iter()
        .zip(y_target.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    println!("  final loss = {:.6}", final_loss);

    println!("\nPredictions vs Targets:");
    for (i, &xi) in x_data.iter().enumerate() {
        let pred = predictions[i];
        let target = y_target[i];
        println!(
            "  x = {:.1}: pred = {:.4}, target = {:.4}, error = {:.4}",
            xi,
            pred,
            target,
            (pred - target).abs()
        );
    }

    println!(
        "\nSuccess! The model learned y = {:.2}*x + {:.2}",
        w_val, b_val
    );

    Ok(())
}
