//! MNIST CNN Demo
//!
//! This example demonstrates training a Convolutional Neural Network
//! to classify MNIST handwritten digits using eclat and eclat-nn.
//!
//! Run with: cargo run --release -p mnist_cnn

// Link backend crates to trigger ctor auto-initialization
extern crate eclat_backend_c;
#[cfg(target_os = "macos")]
extern crate eclat_backend_metal;

use std::error::Error;

use eclat::backend::{get_default_device_kind, set_device_str};
use eclat::tensor::Tensor;
use eclat::tensor::dim::{D2, D4};
use eclat_nn::functional::{cross_entropy_loss, predict_classes, relu};
use eclat_nn::layers::{Module, ParameterBase};
use eclat_nn::optim::{Adam, Optimizer};
use eclat_nn::{Conv2d, Linear, MaxPool2d};
use indicatif::{ProgressBar, ProgressStyle};
use mnist::MnistBuilder;

/// Simple CNN model for MNIST classification.
///
/// Architecture:
/// - Conv2d(1, 32, 3x3, padding=1) -> ReLU -> MaxPool2d(2x2) -> [32, 14, 14]
/// - Conv2d(32, 64, 3x3, padding=1) -> ReLU -> MaxPool2d(2x2) -> [64, 7, 7]
/// - Flatten -> Linear(3136, 128) -> ReLU
/// - Linear(128, 10)
struct MnistCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    pool: MaxPool2d,
}

impl MnistCNN {
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, (3, 3)).with_padding((1, 1)).with_bias(),
            conv2: Conv2d::new(32, 64, (3, 3)).with_padding((1, 1)).with_bias(),
            fc1: Linear::new(3136, 128, true), // 64 * 7 * 7 = 3136
            fc2: Linear::new(128, 10, true),
            pool: MaxPool2d::new((2, 2)),
        }
    }

    fn forward(&self, x: &Tensor<D4, f32>) -> Tensor<D2, f32> {
        // First conv block: conv -> relu -> pool
        let x = self.conv1.forward(x);
        let x = relu(&x);
        let x = self.pool.forward(&x);

        // Second conv block: conv -> relu -> pool
        let x = self.conv2.forward(&x);
        let x = relu(&x);
        let x = self.pool.forward(&x);

        // Flatten: [N, 64, 7, 7] -> [N, 3136]
        let batch_size = x.shape()[0];
        let x: Tensor<D2, f32> = x.reshape([batch_size, 3136]);

        // Fully connected layers
        let x = self.fc1.forward(&x);
        let x = relu(&x);
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

/// Create a batch tensor from MNIST images.
fn create_batch(images: &[u8], indices: &[usize], image_size: usize) -> Tensor<D4, f32> {
    let batch_size = indices.len();
    let mut data = vec![0.0f32; batch_size * image_size];

    for (batch_idx, &sample_idx) in indices.iter().enumerate() {
        let offset = sample_idx * image_size;
        for i in 0..image_size {
            // Normalize to [0, 1]
            data[batch_idx * image_size + i] = images[offset + i] as f32 / 255.0;
        }
    }

    let tensor: Tensor<D4, f32> = Tensor::input([batch_size, 1, 28, 28]);
    tensor.set_data(&data).expect("Failed to set batch data");
    tensor
}

/// Get batch labels.
fn get_batch_labels(labels: &[u8], indices: &[usize]) -> Vec<u8> {
    indices.iter().map(|&i| labels[i]).collect()
}

/// Download and extract MNIST dataset from PyTorch's S3 mirror.
fn download_mnist(data_dir: &std::path::Path) -> Result<(), Box<dyn Error>> {
    use std::process::Command;

    std::fs::create_dir_all(data_dir)?;

    let base_url = "https://ossci-datasets.s3.amazonaws.com/mnist";
    let files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ];

    for file in &files {
        let url = format!("{}/{}", base_url, file);
        let gz_path = data_dir.join(file);
        let extracted_path = data_dir.join(file.trim_end_matches(".gz"));

        // Skip if already extracted
        if extracted_path.exists() {
            continue;
        }

        println!("  Downloading {}...", file);

        // Download using curl
        let output = Command::new("curl")
            .args(["-L", "-o", gz_path.to_str().unwrap(), &url])
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Failed to download {}: {}",
                file,
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        println!("  Extracting {}...", file);

        // Extract using gunzip
        let output = Command::new("gunzip")
            .args(["-f", gz_path.to_str().unwrap()])
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Failed to extract {}: {}",
                file,
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
    }

    println!("  Download complete!");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logger for RUST_LOG support
    env_logger::init();

    println!("=== MNIST CNN Training Demo ===\n");

    // Set up device - use C backend for debugging
    set_device_str("c")?;
    println!("Device: {}\n", get_default_device_kind());

    // ========================================================================
    // Load MNIST data
    // ========================================================================
    println!("Loading MNIST dataset...");

    // MNIST data directory
    let data_dir = std::env::current_dir()
        .unwrap()
        .join("target")
        .join("mnist_data");

    // Download MNIST data if not present
    if !data_dir.join("train-images-idx3-ubyte").exists() {
        println!("Downloading MNIST dataset...");
        download_mnist(&data_dir)?;
    }

    let mnist = MnistBuilder::new()
        .base_path(data_dir.to_str().unwrap())
        .label_format_digit()
        .training_set_length(60000)
        .test_set_length(10000)
        .finalize();

    let train_images = mnist.trn_img;
    let train_labels = mnist.trn_lbl;
    let test_images = mnist.tst_img;
    let test_labels = mnist.tst_lbl;

    println!(
        "  Training samples: {}, Test samples: {}",
        train_labels.len(),
        test_labels.len()
    );

    // ========================================================================
    // Create model and optimizer
    // ========================================================================
    let model = MnistCNN::new();
    let params = model.parameters();

    println!(
        "  Model parameters: {}\n",
        params.iter().map(|p| p.numel()).sum::<usize>()
    );

    // Create optimizer with fresh copy of parameters
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    // ========================================================================
    // Training configuration
    // ========================================================================
    let batch_size = 64;
    let epochs = 5;
    let train_size = train_labels.len();
    let batches_per_epoch = train_size / batch_size;

    println!("Training Configuration:");
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", epochs);
    println!("  Batches per epoch: {}\n", batches_per_epoch);

    // ========================================================================
    // Training loop
    // ========================================================================
    println!("Starting training (first batch may take a while due to JIT compilation)...\n");

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        // Note: Using min to limit batches for faster testing with C backend
        let test_batches = std::cmp::min(batches_per_epoch, 5);

        // Create progress bar for this epoch
        let pb = ProgressBar::new(test_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[Epoch {pos}/{len}] {bar:40.cyan/blue} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        // Simple sequential batching (no shuffling for simplicity)
        for batch_idx in 0..test_batches {
            if batch_idx == 0 && epoch == 0 {
                println!("  Batch 0: Creating batch...");
            }
            let start_idx = batch_idx * batch_size;
            let indices: Vec<usize> = (start_idx..start_idx + batch_size).collect();

            // Create batch
            let batch_images = create_batch(&train_images, &indices, 28 * 28);
            let batch_labels = get_batch_labels(&train_labels, &indices);

            // Zero gradients
            optimizer.zero_grad();

            if batch_idx == 0 && epoch == 0 {
                println!("  Batch 0: Forward pass...");
            }
            // Forward pass
            let logits = model.forward(&batch_images);
            let loss = cross_entropy_loss(&logits, &batch_labels, 10);

            if batch_idx == 0 && epoch == 0 {
                println!("  Batch 0: Realizing loss...");
            }
            // Realize loss to get value for logging
            loss.realize()?;
            let loss_val = loss.to_vec()?[0];
            epoch_loss += loss_val;

            // Get predictions for accuracy
            let predictions = predict_classes(&logits)?;
            let correct = predictions
                .iter()
                .zip(batch_labels.iter())
                .filter(|(p, t)| *p == *t)
                .count();
            epoch_correct += correct;
            epoch_total += batch_size;

            if batch_idx == 0 && epoch == 0 {
                println!("  Batch 0: Backward pass...");
            }
            // Backward pass - compute gradients for all parameters
            loss.backward_with_dyn_params(&params)?;

            if batch_idx == 0 && epoch == 0 {
                println!("  Batch 0: Optimizer step...");
            }
            // Optimizer step
            optimizer.step()?;

            if batch_idx == 0 && epoch == 0 {
                println!("  Batch 0 complete!");
            }

            // Update progress bar
            let current_acc = 100.0 * epoch_correct as f32 / epoch_total as f32;
            pb.set_message(format!("loss: {:.4} | acc: {:.2}%", loss_val, current_acc));
            pb.inc(1);
        }

        pb.finish_and_clear();

        // Epoch summary
        let avg_loss = epoch_loss / batches_per_epoch as f32;
        let train_acc = 100.0 * epoch_correct as f32 / epoch_total as f32;
        println!(
            "Epoch {}/{}: avg_loss = {:.4}, train_acc = {:.2}%",
            epoch + 1,
            epochs,
            avg_loss,
            train_acc
        );

        // Test evaluation
        let test_batch_size = 1000;
        let test_batches = test_labels.len() / test_batch_size;
        let mut test_correct = 0usize;
        let mut test_total = 0usize;

        let test_pb = ProgressBar::new(test_batches as u64);
        test_pb.set_style(
            ProgressStyle::default_bar()
                .template("[Evaluating] {bar:40.green/white} {pos}/{len}")
                .unwrap()
                .progress_chars("##-"),
        );

        for batch_idx in 0..test_batches {
            let start_idx = batch_idx * test_batch_size;
            let indices: Vec<usize> = (start_idx..start_idx + test_batch_size).collect();

            let batch_images = create_batch(&test_images, &indices, 28 * 28);
            let batch_labels = get_batch_labels(&test_labels, &indices);

            let logits = model.forward(&batch_images);
            let predictions = predict_classes(&logits)?;

            let correct = predictions
                .iter()
                .zip(batch_labels.iter())
                .filter(|(p, t)| *p == *t)
                .count();
            test_correct += correct;
            test_total += test_batch_size;
            test_pb.inc(1);
        }

        test_pb.finish_and_clear();

        let test_acc = 100.0 * test_correct as f32 / test_total as f32;
        println!("Test accuracy: {:.2}%\n", test_acc);
    }

    println!("=== Training Complete ===");

    Ok(())
}
