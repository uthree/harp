//! MLP Function Fitting Demo
//!
//! This example demonstrates training a Multi-Layer Perceptron to approximate sin(x).
//! Uses GPU backend for acceleration with automatic differentiation.
//!
//! # Run
//! ```bash
//! cargo run --features "opencl" --example mlp_function_fitting
//! ```

use harp::backend::{HarpDevice, set_device};
use harp::tensor::{Dim1, Dim2, DimDyn, Tensor};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;

/// Simple 2-layer MLP: input -> hidden (tanh) -> output
struct MLP {
    w1: Tensor<f32, Dim2>,
    b1: Tensor<f32, Dim1>,
    w2: Tensor<f32, Dim2>,
    b2: Tensor<f32, Dim1>,
}

impl MLP {
    /// Create a new MLP with random weights
    ///
    /// Architecture: input_dim -> hidden_dim -> output_dim
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // Initialize weights with small random values
        // Xavier-like initialization: scale by sqrt(2 / fan_in)
        let w1 = Tensor::<f32, Dim2>::rand([input_dim, hidden_dim]) * 0.5 - 0.25;
        let b1 = Tensor::<f32, Dim1>::zeros([hidden_dim]);

        let w2 = Tensor::<f32, Dim2>::rand([hidden_dim, output_dim]) * 0.5 - 0.25;
        let b2 = Tensor::<f32, Dim1>::zeros([output_dim]);

        // Enable gradient tracking
        Self {
            w1: w1.set_requires_grad(true),
            b1: b1.set_requires_grad(true),
            w2: w2.set_requires_grad(true),
            b2: b2.set_requires_grad(true),
        }
    }

    /// Forward pass: x -> hidden -> output
    fn forward(&self, x: &Tensor<f32, Dim2>) -> Tensor<f32, DimDyn> {
        // Layer 1: x @ w1 + b1
        let h = x.matmul2(&self.w1).into_dyn();
        let b1_expanded = self
            .b1
            .clone()
            .into_dyn()
            .unsqueeze(0)
            .expand(&h.shape().to_vec());
        let h = &h + &b1_expanded;

        // Activation: tanh (now works with gradient tracking for scalar ops)
        let h = h.tanh();

        // Layer 2: h @ w2 + b2
        let h_dim2 = h.into_dim2();
        let out = h_dim2.matmul2(&self.w2).into_dyn();
        let b2_expanded = self
            .b2
            .clone()
            .into_dyn()
            .unsqueeze(0)
            .expand(&out.shape().to_vec());
        let out = &out + &b2_expanded;

        out
    }

    /// Update parameters using gradients (SGD)
    fn update_params(&mut self, lr: f32) {
        // Get gradients
        let grad_w1 = self.w1.grad().expect("w1 should have gradient");
        let grad_b1 = self.b1.grad().expect("b1 should have gradient");
        let grad_w2 = self.w2.grad().expect("w2 should have gradient");
        let grad_b2 = self.b2.grad().expect("b2 should have gradient");

        // Realize gradients to get concrete values
        grad_w1.realize().expect("Failed to realize grad_w1");
        grad_b1.realize().expect("Failed to realize grad_b1");
        grad_w2.realize().expect("Failed to realize grad_w2");
        grad_b2.realize().expect("Failed to realize grad_b2");

        // Get gradient data
        let grad_w1_data = grad_w1.data().expect("Failed to get grad_w1 data");
        let grad_b1_data = grad_b1.data().expect("Failed to get grad_b1 data");
        let grad_w2_data = grad_w2.data().expect("Failed to get grad_w2 data");
        let grad_b2_data = grad_b2.data().expect("Failed to get grad_b2 data");

        // Get current parameter values
        self.w1.realize().expect("Failed to realize w1");
        self.b1.realize().expect("Failed to realize b1");
        self.w2.realize().expect("Failed to realize w2");
        self.b2.realize().expect("Failed to realize b2");

        let w1_data = self.w1.data().expect("Failed to get w1 data");
        let b1_data = self.b1.data().expect("Failed to get b1 data");
        let w2_data = self.w2.data().expect("Failed to get w2 data");
        let b2_data = self.b2.data().expect("Failed to get b2 data");

        // SGD update: param = param - lr * grad
        let new_w1: Vec<f32> = w1_data
            .iter()
            .zip(grad_w1_data.iter())
            .map(|(p, g)| p - lr * g)
            .collect();
        let new_b1: Vec<f32> = b1_data
            .iter()
            .zip(grad_b1_data.iter())
            .map(|(p, g)| p - lr * g)
            .collect();
        let new_w2: Vec<f32> = w2_data
            .iter()
            .zip(grad_w2_data.iter())
            .map(|(p, g)| p - lr * g)
            .collect();
        let new_b2: Vec<f32> = b2_data
            .iter()
            .zip(grad_b2_data.iter())
            .map(|(p, g)| p - lr * g)
            .collect();

        // Create new tensors with updated values
        let w1_shape = self.w1.shape().to_vec();
        let b1_shape = self.b1.shape().to_vec();
        let w2_shape = self.w2.shape().to_vec();
        let b2_shape = self.b2.shape().to_vec();

        // Use ndarray to create tensors
        let w1_arr = Array2::from_shape_vec((w1_shape[0], w1_shape[1]), new_w1).unwrap();
        let b1_arr = ndarray::Array1::from_shape_vec(b1_shape[0], new_b1).unwrap();
        let w2_arr = Array2::from_shape_vec((w2_shape[0], w2_shape[1]), new_w2).unwrap();
        let b2_arr = ndarray::Array1::from_shape_vec(b2_shape[0], new_b2).unwrap();

        self.w1 = Tensor::<f32, Dim2>::from_ndarray(&w1_arr).set_requires_grad(true);
        self.b1 = Tensor::<f32, Dim1>::from_ndarray(&b1_arr).set_requires_grad(true);
        self.w2 = Tensor::<f32, Dim2>::from_ndarray(&w2_arr).set_requires_grad(true);
        self.b2 = Tensor::<f32, Dim1>::from_ndarray(&b2_arr).set_requires_grad(true);
    }
}

/// Generate training data for sin(x)
fn generate_sin_data(n_samples: usize) -> (Tensor<f32, Dim2>, Tensor<f32, Dim2>) {
    let mut x_data = Vec::with_capacity(n_samples);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x = (i as f32 / n_samples as f32) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
        x_data.push(x);
        y_data.push(x.sin());
    }

    let x_arr = Array2::from_shape_vec((n_samples, 1), x_data).unwrap();
    let y_arr = Array2::from_shape_vec((n_samples, 1), y_data).unwrap();

    (
        Tensor::<f32, Dim2>::from_ndarray(&x_arr),
        Tensor::<f32, Dim2>::from_ndarray(&y_arr),
    )
}

/// Mean Squared Error loss
fn mse_loss(pred: &Tensor<f32, DimDyn>, target: &Tensor<f32, Dim2>) -> Tensor<f32, DimDyn> {
    let target_dyn = target.clone().into_dyn();
    let diff = pred - &target_dyn;
    let sq = &diff * &diff;
    // sum over all dimensions to get mean
    let n_elements = sq.shape().iter().product::<usize>() as f32;
    // Sum over all axes (reduce to scalar)
    let mut result = sq;
    while result.ndim() > 0 {
        result = result.sum(0);
    }
    // Use tensor multiplication for gradient tracking (scalar ops don't track gradients)
    let inv_n = Tensor::<f32, DimDyn>::full_dyn(&[], 1.0 / n_elements);
    &result * &inv_n
}

fn main() {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("=== MLP Function Fitting Demo ===");
    println!("Learning to approximate sin(x) using a 2-layer MLP\n");

    // Setup device (auto-select best available)
    let device = HarpDevice::auto().expect("No available device found");
    println!("Using device: {:?}", device.kind());
    set_device(device);

    // Hyperparameters
    let n_samples = 64;
    let hidden_dim = 32;
    let learning_rate = 0.1;
    let n_epochs = 1000;

    // Generate training data
    println!("Generating training data...");
    let (x_train, y_train) = generate_sin_data(n_samples);
    println!("  X shape: {:?}", x_train.shape());
    println!("  Y shape: {:?}", y_train.shape());
    println!();

    // Create MLP
    println!("Creating MLP (1 -> {} -> 1)...", hidden_dim);
    let mut mlp = MLP::new(1, hidden_dim, 1);
    println!();

    // Training loop with progress bar
    println!("Training for {} epochs...\n", n_epochs);

    let pb = ProgressBar::new(n_epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} | Loss: {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut final_loss = 0.0;

    for epoch in 0..n_epochs {
        // Forward pass
        let pred = mlp.forward(&x_train);

        // Compute loss
        let loss = mse_loss(&pred, &y_train);

        // Backward pass
        loss.backward();

        // Get and display loss value
        loss.realize().expect("Failed to realize loss");
        let loss_val = loss.data().expect("Failed to get loss data")[0];
        final_loss = loss_val;

        // Update progress bar with current loss
        pb.set_message(format!("{:.6}", loss_val));
        pb.set_position((epoch + 1) as u64);

        // Update parameters
        mlp.update_params(learning_rate);
    }

    pb.finish_and_clear();
    println!("Training complete! Final loss: {:.6}", final_loss);

    // Final evaluation
    println!("\nFinal predictions (sample):");
    println!("{:>8} {:>10} {:>10}", "x", "sin(x)", "pred");
    println!("{:-<32}", "");

    let pred = mlp.forward(&x_train);
    pred.realize().expect("Failed to realize predictions");
    let pred_data = pred.data().expect("Failed to get prediction data");

    // Show first and last few samples
    for i in [
        0,
        n_samples / 4,
        n_samples / 2,
        3 * n_samples / 4,
        n_samples - 1,
    ] {
        let x = (i as f32 / n_samples as f32) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
        let y_true = x.sin();
        let y_pred = pred_data[i];
        println!("{:>8.3} {:>10.4} {:>10.4}", x, y_true, y_pred);
    }
}
