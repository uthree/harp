//! Loss functions
//!
//! This module provides loss functions for neural network training.

use eclat::backend::ExecutionError;
use eclat::tensor::dim::{D0, D1, D2};
use eclat::tensor::Tensor;

use super::log_softmax;

/// Cross Entropy Loss using one-hot encoding.
///
/// Computes the cross entropy loss between logits and target class indices.
/// This is equivalent to: -mean(sum(one_hot * log_softmax(logits), axis=1))
///
/// # Arguments
/// * `logits` - Predicted logits with shape [N, C] where N is batch size and C is number of classes
/// * `targets` - Target class indices as a slice of length N, values in [0, C-1]
/// * `num_classes` - Number of classes C
///
/// # Returns
/// Scalar loss tensor
///
/// # Example
/// ```ignore
/// use eclat_nn::functional::cross_entropy_loss;
///
/// let logits: Tensor<D2, f32> = Tensor::input([32, 10]); // batch of 32, 10 classes
/// let targets: Vec<u8> = vec![0, 1, 2, ...]; // 32 target labels
/// let loss = cross_entropy_loss(&logits, &targets, 10);
/// ```
pub fn cross_entropy_loss(
    logits: &Tensor<D2, f32>,
    targets: &[u8],
    num_classes: usize,
) -> Tensor<D0, f32> {
    let shape = logits.shape();
    let batch_size = shape[0];

    // 1. Compute log_softmax along class dimension (axis=1)
    let log_probs = log_softmax(logits, 1);

    // 2. Create one-hot encoding [N, C]
    let mut one_hot_data = vec![0.0f32; batch_size * num_classes];
    for (i, &label) in targets.iter().enumerate() {
        one_hot_data[i * num_classes + label as usize] = 1.0;
    }
    let one_hot: Tensor<D2, f32> = Tensor::input([batch_size, num_classes]);
    one_hot
        .set_data(&one_hot_data)
        .expect("one_hot set_data failed");

    // 3. Compute -mean(sum(one_hot * log_probs))
    // Element-wise multiplication: select log prob of correct class
    let selected = &one_hot * &log_probs;

    // Sum over classes (axis=1) to get loss per sample [N]
    let loss_per_sample: Tensor<D1, f32> = selected.sum(1);

    // Sum over batch to get scalar
    let total_loss: Tensor<D0, f32> = loss_per_sample.sum(0);

    // Negate and divide by batch size using scale operation
    total_loss.scale(-1.0 / batch_size as f32)
}

/// Compute softmax predictions from logits.
///
/// Returns the predicted class indices for each sample in the batch.
///
/// # Arguments
/// * `logits` - Predicted logits with shape [N, C]
///
/// # Returns
/// Vector of predicted class indices
pub fn predict_classes(logits: &Tensor<D2, f32>) -> Result<Vec<u8>, ExecutionError> {
    // We need to realize and extract the logits to find argmax
    logits.realize()?;
    let data = logits.to_vec()?;
    let shape = logits.shape();
    let batch_size = shape[0];
    let num_classes = shape[1];

    let mut predictions = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let start = i * num_classes;
        let end = start + num_classes;
        let sample = &data[start..end];

        // Find argmax
        let mut max_idx = 0;
        let mut max_val = sample[0];
        for (j, &val) in sample.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }
        predictions.push(max_idx as u8);
    }

    Ok(predictions)
}

/// Compute accuracy between predictions and targets.
pub fn accuracy(predictions: &[u8], targets: &[u8]) -> f32 {
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| *p == *t)
        .count();
    correct as f32 / predictions.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: cross_entropy_loss requires a device for set_data,
    // so this test is skipped. The function is tested via integration tests.

    #[test]
    fn test_accuracy() {
        let predictions = vec![0u8, 1, 2, 3, 4];
        let targets = vec![0u8, 1, 0, 3, 4];
        let acc = accuracy(&predictions, &targets);
        assert!((acc - 0.8).abs() < 1e-6); // 4/5 = 0.8
    }
}
