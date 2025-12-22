//! Curve Fitting Demo with Gradient Descent
//!
//! Fits a polynomial to a noisy sin curve.
//! Shows progress bar and terminal graphs for visualization.
//!
//! Run:
//! ```
//! cargo run --example curve_fitting -p autograd
//! ```

use harp_autograd::Differentiable;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use textplots::{Chart, Plot, Shape};

/// Polynomial model: y = a*x^3 + b*x^2 + c*x + d
fn polynomial(
    x: f64,
    a: &Differentiable<f64>,
    b: &Differentiable<f64>,
    c: &Differentiable<f64>,
    d: &Differentiable<f64>,
) -> Differentiable<f64> {
    let x_var = Differentiable::new_no_grad(x);
    let x2 = Differentiable::new_no_grad(x * x);
    let x3 = Differentiable::new_no_grad(x * x * x);

    // a*x^3 + b*x^2 + c*x + d
    &(&(a * &x3) + &(b * &x2)) + &(&(c * &x_var) + d)
}

/// Compute mean squared error
fn mse_loss(
    xs: &[f64],
    ys: &[f64],
    a: &Differentiable<f64>,
    b: &Differentiable<f64>,
    c: &Differentiable<f64>,
    d: &Differentiable<f64>,
) -> Differentiable<f64> {
    let n = xs.len() as f64;
    let mut total_loss = Differentiable::new(0.0);

    for (&x, &y_target) in xs.iter().zip(ys.iter()) {
        let y_pred = polynomial(x, a, b, c, d);
        let y_target_var = Differentiable::new_no_grad(y_target);
        let diff = &y_pred - &y_target_var;
        let squared = &diff * &diff;
        total_loss = &total_loss + &squared;
    }

    let n_var = Differentiable::new_no_grad(n);
    &total_loss / &n_var
}

/// Reset gradients to zero
fn zero_grad(params: &[&Differentiable<f64>]) {
    for p in params {
        p.zero_grad();
    }
}

/// Gradient descent step
fn sgd_step(params: &mut [Differentiable<f64>], lr: f64) {
    for p in params.iter_mut() {
        if let Some(grad) = p.grad() {
            let new_val = p.value() - lr * grad.value();
            *p = Differentiable::new(new_val);
        }
    }
}

fn main() {
    println!("Curve Fitting Demo with Gradient Descent");
    println!();

    // ============================================================
    // 1. Generate noisy data
    // ============================================================
    println!("Generating data...");

    let mut rng = rand::thread_rng();
    let n_samples = 50;
    let noise_scale = 0.3;

    // Add noise to sin curve
    let xs: Vec<f64> = (0..n_samples)
        .map(|i| (i as f64 / n_samples as f64) * 4.0 - 2.0) // [-2, 2]
        .collect();

    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| {
            let noise: f64 = rng.gen_range(-noise_scale..noise_scale);
            x.sin() + noise
        })
        .collect();

    println!("  Samples: {}", n_samples);
    println!(
        "  Input range: [{:.1}, {:.1}]",
        xs.first().unwrap(),
        xs.last().unwrap()
    );
    println!();

    // ============================================================
    // 2. Initialize parameters
    // ============================================================
    println!("Initializing parameters...");

    let mut a = Differentiable::new(rng.gen_range(-0.5..0.5));
    let mut b = Differentiable::new(rng.gen_range(-0.5..0.5));
    let mut c = Differentiable::new(rng.gen_range(-0.5..0.5));
    let mut d = Differentiable::new(rng.gen_range(-0.5..0.5));

    println!(
        "  Initial: a={:.4}, b={:.4}, c={:.4}, d={:.4}",
        a.value(),
        b.value(),
        c.value(),
        d.value()
    );
    println!();

    // ============================================================
    // 3. Optimization with gradient descent
    // ============================================================
    let epochs = 10000;
    let lr = 0.01;

    println!("Starting optimization (epochs={}, lr={})", epochs, lr);
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
        // Reset gradients
        zero_grad(&[&a, &b, &c, &d]);

        // Compute loss
        let loss = mse_loss(&xs, &ys, &a, &b, &c, &d);
        let loss_val = loss.value();
        loss_history.push(loss_val);

        // Backpropagation
        loss.backward();

        // Update parameters
        let mut params = [a.clone(), b.clone(), c.clone(), d.clone()];
        sgd_step(&mut params, lr);
        a = params[0].clone();
        b = params[1].clone();
        c = params[2].clone();
        d = params[3].clone();

        // Update progress bar
        if epoch % 10 == 0 || epoch == epochs - 1 {
            pb.set_message(format!("{:.6}", loss_val));
        }
        pb.inc(1);
    }

    pb.finish_with_message(format!("{:.6}", loss_history.last().unwrap()));
    println!();

    // ============================================================
    // 4. Show results
    // ============================================================
    println!("Optimization complete!");
    println!();
    println!("Learned parameters:");
    println!("  a = {:.6} (x^3 coefficient)", a.value());
    println!("  b = {:.6} (x^2 coefficient)", b.value());
    println!("  c = {:.6} (x coefficient)", c.value());
    println!("  d = {:.6} (constant term)", d.value());
    println!();
    println!("  Final loss: {:.6}", loss_history.last().unwrap());
    println!();

    // ============================================================
    // 5. Visualize with terminal graph
    // ============================================================
    println!("Loss history:");
    let loss_points: Vec<(f32, f32)> = loss_history
        .iter()
        .enumerate()
        .step_by(10)
        .map(|(i, &l)| (i as f32, l as f32))
        .collect();

    Chart::new(120, 40, 0.0, epochs as f32)
        .lineplot(&Shape::Lines(&loss_points))
        .nice();
    println!();

    // Fitting result graph
    println!("Fitting result (o: data, -: prediction):");

    // Data points
    let data_points: Vec<(f32, f32)> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (x as f32, y as f32))
        .collect();

    // Prediction curve
    let pred_points: Vec<(f32, f32)> = (-200..=200)
        .map(|i| {
            let x = i as f64 / 100.0;
            let y = a.value() * x.powi(3) + b.value() * x.powi(2) + c.value() * x + d.value();
            (x as f32, y as f32)
        })
        .collect();

    // True sin curve
    let true_points: Vec<(f32, f32)> = (-200..=200)
        .map(|i| {
            let x = i as f64 / 100.0;
            (x as f32, x.sin() as f32)
        })
        .collect();

    Chart::new(120, 60, -2.0, 2.0)
        .lineplot(&Shape::Points(&data_points))
        .lineplot(&Shape::Lines(&pred_points))
        .lineplot(&Shape::Lines(&true_points))
        .nice();

    println!();
    println!("Legend: scatter=observed data, solid=polynomial prediction, dashed=true sin curve");
}
