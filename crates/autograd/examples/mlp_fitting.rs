//! 2-Layer Perceptron (MLP) Multidimensional Function Fitting Demo
//!
//! Learns a complex function z = f(x, y) from 2D input (x, y).
//!
//! Target function:
//!   z = sin(pi*x) * cos(pi*y) + 0.3x^2 - 0.2y^2 + 0.1xy
//!
//! Network structure:
//!   Input (2) -> Hidden (32) -> Output (1)
//!
//! Run:
//! ```
//! cargo run --example mlp_fitting -p autograd --features ndarray
//! ```

use autograd::Differentiable;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use rand::Rng;

// ============================================================================
// Target function
// ============================================================================

/// Target function: z = sin(pi*x) * cos(pi*y) + 0.3x^2 - 0.2y^2 + 0.1xy
fn target_function(x: f64, y: f64) -> f64 {
    let pi = std::f64::consts::PI;
    (pi * x).sin() * (pi * y).cos() + 0.3 * x * x - 0.2 * y * y + 0.1 * x * y
}

// ============================================================================
// ReLU activation function
// ============================================================================

/// ReLU: max(x, 0)
fn relu(x: &Differentiable<Array2<f64>>) -> Differentiable<Array2<f64>> {
    x.maximum(&x.zeros_like())
}

// ============================================================================
// 2-Layer MLP
// ============================================================================

/// 2-layer perceptron (no bias, ReLU activation)
struct Mlp {
    // Layer 1: [input, hidden]
    w1: Differentiable<Array2<f64>>,
    // Layer 2: [hidden, output]
    w2: Differentiable<Array2<f64>>,
}

impl Mlp {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // He initialization (for ReLU)
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let w1_data: Vec<f64> = (0..input_dim * hidden_dim)
            .map(|_| rng.gen_range(-scale1..scale1))
            .collect();
        let w2_data: Vec<f64> = (0..hidden_dim * output_dim)
            .map(|_| rng.gen_range(-scale2..scale2))
            .collect();

        let w1 = Array2::from_shape_vec((input_dim, hidden_dim), w1_data).unwrap();
        let w2 = Array2::from_shape_vec((hidden_dim, output_dim), w2_data).unwrap();

        Self {
            w1: Differentiable::new(w1),
            w2: Differentiable::new(w2),
        }
    }

    /// Forward pass: x [batch, input] -> y [batch, output]
    fn forward(&self, x: &Differentiable<Array2<f64>>) -> Differentiable<Array2<f64>> {
        // Layer 1: z1 = x @ W1
        let z1 = x.matmul(&self.w1);

        // ReLU activation: h = max(z1, 0)
        let h = relu(&z1);

        // Layer 2: y = h @ W2
        h.matmul(&self.w2)
    }

    /// Zero gradients
    fn zero_grad(&self) {
        self.w1.zero_grad();
        self.w2.zero_grad();
    }

    /// Update parameters with gradient descent
    fn step(&mut self, lr: f64) {
        if let Some(grad) = self.w1.grad() {
            let new_w1 = self.w1.value() - &(grad.value() * lr);
            self.w1 = Differentiable::new(new_w1);
        }
        if let Some(grad) = self.w2.grad() {
            let new_w2 = self.w2.value() - &(grad.value() * lr);
            self.w2 = Differentiable::new(new_w2);
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("2-Layer Perceptron Multidimensional Function Fitting Demo");
    println!();

    // ============================================================
    // 1. Generate data
    // ============================================================
    println!("Generating data...");

    let mut rng = rand::thread_rng();
    let n_samples = 500;
    let noise_scale = 0.05;

    // Sample from [-1, 1] x [-1, 1]
    let mut x_data: Vec<f64> = Vec::with_capacity(n_samples * 2);
    let mut y_data: Vec<f64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);
        let z = target_function(x, y) + rng.gen_range(-noise_scale..noise_scale);

        x_data.push(x);
        x_data.push(y);
        y_data.push(z);
    }

    let x_train = Array2::from_shape_vec((n_samples, 2), x_data).unwrap();
    let y_train = Array2::from_shape_vec((n_samples, 1), y_data).unwrap();

    println!("  Samples: {}", n_samples);
    println!("  Input dim: 2 (x, y)");
    println!("  Output dim: 1 (z)");
    println!("  Target function: z = sin(pi*x)cos(pi*y) + 0.3x^2 - 0.2y^2 + 0.1xy");
    println!();

    // ============================================================
    // 2. Initialize MLP
    // ============================================================
    println!("Initializing network...");

    let input_dim = 2;
    let hidden_dim = 32;
    let output_dim = 1;

    let mut mlp = Mlp::new(input_dim, hidden_dim, output_dim);

    println!(
        "  Structure: {} -> {} -> {} (no bias)",
        input_dim, hidden_dim, output_dim
    );
    println!(
        "  Parameters: {} (W1: {}x{} + W2: {}x{})",
        input_dim * hidden_dim + hidden_dim * output_dim,
        input_dim,
        hidden_dim,
        hidden_dim,
        output_dim
    );
    println!();

    // ============================================================
    // 3. Training (gradient descent with autograd)
    // ============================================================
    let epochs = 1000;
    let lr = 0.01;
    let batch_size = 50;
    let n_batches = n_samples / batch_size;

    println!(
        "Starting training (epochs={}, lr={}, batch_size={})",
        epochs, lr, batch_size
    );
    println!("  Activation: ReLU (max(x, 0))");
    println!("  Gradient: autograd (backpropagation)");
    println!();

    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (loss: {msg})")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut loss_history: Vec<f64> = Vec::new();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let x_batch = x_train.slice(ndarray::s![start..end, ..]).to_owned();
            let y_batch = y_train.slice(ndarray::s![start..end, ..]).to_owned();

            // Zero gradients
            mlp.zero_grad();

            // Forward pass
            let x_var = Differentiable::new(x_batch);
            let y_var = Differentiable::new(y_batch);
            let pred = mlp.forward(&x_var);

            // MSE loss: L = mean((pred - target)^2)
            let diff = &pred - &y_var;
            let squared = &diff * &diff;
            let loss = squared.sum(0).sum(1); // Reduce to scalar

            // Record loss
            let loss_val = loss.value()[[0, 0]] / (batch_size as f64);
            epoch_loss += loss_val;

            // Backpropagation
            // Gradient scale: 1/batch_size (for MSE averaging)
            let grad_scale = 1.0 / (batch_size as f64);
            let grad = Differentiable::new(Array2::from_elem((1, 1), grad_scale));
            loss.backward_with(grad);

            // Update parameters
            mlp.step(lr);
        }

        epoch_loss /= n_batches as f64;
        loss_history.push(epoch_loss);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            pb.set_message(format!("{:.6}", epoch_loss));
        }
        pb.inc(1);
    }

    pb.finish_with_message(format!("{:.6}", loss_history.last().unwrap()));
    println!();

    // ============================================================
    // 4. Show results
    // ============================================================
    println!("Training complete!");
    println!();
    println!("Final results:");
    println!("  Final loss: {:.6}", loss_history.last().unwrap());
    println!();

    // Test: compare predictions with true values at some points
    println!("Prediction vs True (samples):");
    println!(
        "  {:>8} {:>8} | {:>10} {:>10} | {:>8}",
        "x", "y", "Pred", "True", "Error"
    );
    println!("  -----------------|-----------------------|---------");

    let test_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.7),
        (-0.8, -0.3),
    ];

    for (x, y) in test_points {
        let input = Array2::from_shape_vec((1, 2), vec![x, y]).unwrap();
        let x_var = Differentiable::new(input);
        let pred = mlp.forward(&x_var);
        let pred_val = pred.value()[[0, 0]];
        let true_val = target_function(x, y);
        let error = (pred_val - true_val).abs();

        println!(
            "  {:>8.3} {:>8.3} | {:>10.4} {:>10.4} | {:>8.4}",
            x, y, pred_val, true_val, error
        );
    }
    println!();

    // Loss history graph
    println!("Loss history:");
    use textplots::{Chart, Plot, Shape};

    let loss_points: Vec<(f32, f32)> = loss_history
        .iter()
        .enumerate()
        .step_by(10)
        .map(|(i, &l)| (i as f32, l as f32))
        .collect();

    Chart::new(100, 30, 0.0, epochs as f32)
        .lineplot(&Shape::Lines(&loss_points))
        .nice();

    println!();
    println!("This demo uses ReLU activation and autograd for gradient computation.");
}
