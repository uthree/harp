//! MLP Function Fitting Demo
//!
//! This example demonstrates training a Multi-Layer Perceptron to approximate sin(x)
//! using the harp-nn module system with Linear layers and optimizers.
//!
//! # Run
//! ```bash
//! cargo run --features "opencl" --example mlp_function_fitting
//! ```

use harp::backend::{HarpDevice, set_device};
use harp::tensor::{Dim2, DimDyn, Tensor};
use harp_nn::{Adam, Linear, Module, Optimizer, Parameter, Tanh, mse_loss};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use std::collections::HashMap;

/// Simple 2-layer MLP using harp-nn Linear layers
///
/// Architecture: input -> Linear -> Tanh -> Linear -> output
struct MLP {
    fc1: Linear<f32>,
    activation: Tanh<f32>,
    fc2: Linear<f32>,
}

impl MLP {
    /// Create a new MLP
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            fc1: Linear::new(input_dim, hidden_dim),
            activation: Tanh::new(),
            fc2: Linear::new(hidden_dim, output_dim),
        }
    }

    /// Forward pass
    fn forward(&self, x: &Tensor<f32, Dim2>) -> Tensor<f32, DimDyn> {
        let h = self.fc1.forward(x);
        let h = self.activation.forward(&h);
        self.fc2.forward(&h.into_dim2()).into_dyn()
    }
}

impl Module<f32> for MLP {
    fn parameters(&mut self) -> HashMap<String, &mut Parameter<f32>> {
        let mut params = HashMap::new();
        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }
        params
    }

    fn load_parameters(&mut self, params: HashMap<String, Parameter<f32>>) {
        let mut fc1_params = HashMap::new();
        let mut fc2_params = HashMap::new();

        for (name, param) in params {
            if let Some(suffix) = name.strip_prefix("fc1.") {
                fc1_params.insert(suffix.to_string(), param);
            } else if let Some(suffix) = name.strip_prefix("fc2.") {
                fc2_params.insert(suffix.to_string(), param);
            }
        }

        self.fc1.load_parameters(fc1_params);
        self.fc2.load_parameters(fc2_params);
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

fn main() {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("=== MLP Function Fitting Demo (harp-nn) ===");
    println!("Learning to approximate sin(x) using Linear layers and Adam optimizer\n");

    // Setup device (auto-select best available)
    let device = HarpDevice::auto().expect("No available device found");
    println!("Using device: {:?}", device.kind());
    set_device(device);

    // Hyperparameters
    let n_samples = 64;
    let hidden_dim = 32;
    let learning_rate = 0.01;
    let n_epochs = 1000;

    // Generate training data
    println!("Generating training data...");
    let (x_train, y_train) = generate_sin_data(n_samples);
    println!("  X shape: {:?}", x_train.shape());
    println!("  Y shape: {:?}", y_train.shape());
    println!();

    // Create MLP using harp-nn
    println!("Creating MLP (1 -> {} -> 1)...", hidden_dim);
    println!("  Layers: Linear -> Tanh -> Linear");
    let mut mlp = MLP::new(1, hidden_dim, 1);

    // Create Adam optimizer
    let mut optimizer = Adam::<f32>::new(learning_rate);
    println!("  Optimizer: Adam (lr={})", learning_rate);
    println!(
        "  Parameters: {}",
        mlp.parameters().values().map(|p| p.numel()).sum::<usize>()
    );
    println!();

    // Training loop with progress bar
    println!("Training for {} epochs...\n", n_epochs);

    let pb = ProgressBar::new(n_epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} | Loss: {msg}",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut final_loss = 0.0;
    let y_train_dyn = y_train.clone().into_dyn();

    for epoch in 0..n_epochs {
        // Zero gradients
        mlp.zero_grad();

        // Forward pass
        let pred = mlp.forward(&x_train);

        // Compute MSE loss
        let loss = mse_loss(&pred, &y_train_dyn);

        // Backward pass
        loss.backward();

        // Get and display loss value
        loss.realize().expect("Failed to realize loss");
        let loss_val = loss.data().expect("Failed to get loss data")[0];
        final_loss = loss_val;

        // Update progress bar
        pb.set_message(format!("{:.6}", loss_val));
        pb.set_position((epoch + 1) as u64);

        // Update parameters using Adam optimizer
        optimizer.step(&mut mlp);
    }

    pb.finish_and_clear();
    println!("Training complete! Final loss: {:.6}", final_loss);

    // Final evaluation
    println!("\nFinal predictions (sample):");
    println!("{:>8} {:>10} {:>10} {:>10}", "x", "sin(x)", "pred", "error");
    println!("{:-<44}", "");

    let pred = mlp.forward(&x_train);
    pred.realize().expect("Failed to realize predictions");
    let pred_data = pred.data().expect("Failed to get prediction data");

    let mut total_error = 0.0;

    // Show samples at key points
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
        let error = (y_true - y_pred).abs();
        total_error += error;
        println!(
            "{:>8.3} {:>10.4} {:>10.4} {:>10.4}",
            x, y_true, y_pred, error
        );
    }

    println!("{:-<44}", "");
    println!("Average absolute error: {:.4}", total_error / 5.0);
}
