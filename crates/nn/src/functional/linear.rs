//! Linear (Fully Connected) operations
//!
//! Implements `y = xW^T + b` where:
//! - x: input tensor [batch, in_features]
//! - W: weight matrix [out_features, in_features]
//! - b: bias vector [out_features] (optional)

use eclat::tensor::Tensor;
use eclat::tensor::dim::{D1, D2};

/// Applies a linear transformation: `y = xW^T + b`.
///
/// # Arguments
/// * `input` - Input tensor of shape [batch, in_features]
/// * `weight` - Weight tensor of shape [out_features, in_features]
/// * `bias` - Optional bias tensor of shape [out_features]
///
/// # Returns
/// Output tensor of shape [batch, out_features]
///
/// # Implementation
/// Matrix multiplication is implemented using unsqueeze and sum:
/// 1. input: [B, K] -> [B, K, 1]
/// 2. weight^T: [K, O] -> [1, K, O]
/// 3. broadcast multiply: [B, K, 1] * [1, K, O] -> [B, K, O]
/// 4. sum over K: [B, K, O] -> [B, O]
pub fn linear(
    input: &Tensor<D2, f32>,
    weight: &Tensor<D2, f32>,
    bias: Option<&Tensor<D1, f32>>,
) -> Tensor<D2, f32> {
    // Get weight tensor and transpose it: [O, K] -> [K, O]
    let weight_t = weight.permute(&[1, 0]); // [K, O]

    // input: [B, K] -> [B, K, 1]
    let a = input.unsqueeze(2);

    // weight^T: [K, O] -> [1, K, O]
    let b = weight_t.unsqueeze(0);

    // broadcast multiply: [B, K, 1] * [1, K, O] -> [B, K, O]
    let product = &a * &b;

    // sum over K axis: [B, K, O] -> [B, O]
    let y = product.sum(1);

    // Add bias if present
    match bias {
        Some(bias_tensor) => {
            // bias: [O] -> [1, O]
            let bias_expanded = bias_tensor.unsqueeze(0);
            &y + &bias_expanded
        }
        None => y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let input: Tensor<D2, f32> = Tensor::input([4, 10]);
        let weight: Tensor<D2, f32> = Tensor::input([5, 10]);
        let output = linear(&input, &weight, None);
        assert_eq!(output.shape(), vec![4, 5]);
    }

    #[test]
    fn test_linear_with_bias() {
        let input: Tensor<D2, f32> = Tensor::input([4, 10]);
        let weight: Tensor<D2, f32> = Tensor::input([5, 10]);
        let bias: Tensor<D1, f32> = Tensor::input([5]);
        let output = linear(&input, &weight, Some(&bias));
        assert_eq!(output.shape(), vec![4, 5]);
    }
}
