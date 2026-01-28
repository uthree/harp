//! Activation functions
//!
//! This module provides pure activation functions for neural networks.
//! These are implemented using basic tensor operations.

use eclat::tensor::dim::Dimension;
use eclat::tensor::Tensor;

/// Applies the Rectified Linear Unit (ReLU) function element-wise.
///
/// `ReLU(x) = max(0, x)`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Output tensor with the same shape as input
pub fn relu<D: Dimension>(input: &Tensor<D, f32>) -> Tensor<D, f32> {
    let zeros = Tensor::<D, f32>::zeros_like(input);
    input.maximum(&zeros)
}

/// Applies the Leaky ReLU function element-wise.
///
/// `LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)`
///             = x if x >= 0, else negative_slope * x
///
/// # Arguments
/// * `input` - Input tensor of any shape
/// * `negative_slope` - Controls the angle of the negative slope (default: 0.01)
///
/// # Returns
/// Output tensor with the same shape as input
pub fn leaky_relu<D: Dimension>(input: &Tensor<D, f32>, negative_slope: f32) -> Tensor<D, f32> {
    let zeros = Tensor::<D, f32>::zeros_like(input);
    let cond = input.ge(&zeros);
    let neg_part = input.scale(negative_slope);
    input.where_cond(&cond, &neg_part)
}

/// Applies the Sigmoid function element-wise.
///
/// `Sigmoid(x) = 1 / (1 + exp(-x))`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Output tensor with the same shape as input, values in range (0, 1)
pub fn sigmoid<D: Dimension>(input: &Tensor<D, f32>) -> Tensor<D, f32> {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let neg_input = input.neg();
    let exp_neg = neg_input.exp();
    let ones = Tensor::<D, f32>::ones_like(input);
    let denom = &ones + &exp_neg;
    denom.recip()
}

/// Applies the Hyperbolic Tangent (Tanh) function element-wise.
///
/// `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
///         = (exp(2x) - 1) / (exp(2x) + 1)
///         = 2 * sigmoid(2x) - 1
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Output tensor with the same shape as input, values in range (-1, 1)
pub fn tanh<D: Dimension>(input: &Tensor<D, f32>) -> Tensor<D, f32> {
    // tanh(x) = 2 * sigmoid(2x) - 1
    let two_x = input.scale(2.0);
    let sig = sigmoid(&two_x);
    let ones = Tensor::<D, f32>::ones_like(input);
    &sig.scale(2.0) - &ones
}

/// Applies the Gaussian Error Linear Unit (GELU) function element-wise.
///
/// This implementation uses the approximation:
/// `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Output tensor with the same shape as input
pub fn gelu<D: Dimension>(input: &Tensor<D, f32>) -> Tensor<D, f32> {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
    const COEFF: f32 = 0.044715;

    let x_cubed = &input.square() * input;
    let inner = &(input.scale(1.0)) + &x_cubed.scale(COEFF);
    let tanh_arg = inner.scale(SQRT_2_OVER_PI);
    let tanh_result = tanh(&tanh_arg);
    let ones = Tensor::<D, f32>::ones_like(input);
    let factor = &ones + &tanh_result;
    (input * &factor).scale(0.5)
}

/// Applies the Sigmoid Linear Unit (SiLU) function element-wise.
///
/// Also known as Swish: `SiLU(x) = x * sigmoid(x)`
///
/// # Arguments
/// * `input` - Input tensor of any shape
///
/// # Returns
/// Output tensor with the same shape as input
pub fn silu<D: Dimension>(input: &Tensor<D, f32>) -> Tensor<D, f32> {
    let sig = sigmoid(input);
    input * &sig
}

/// Applies the Exponential Linear Unit (ELU) function element-wise.
///
/// `ELU(x) = x if x > 0, else alpha * (exp(x) - 1)`
///
/// # Arguments
/// * `input` - Input tensor of any shape
/// * `alpha` - The α value for the ELU formulation (default: 1.0)
///
/// # Returns
/// Output tensor with the same shape as input
pub fn elu<D: Dimension>(input: &Tensor<D, f32>, alpha: f32) -> Tensor<D, f32> {
    let zeros = Tensor::<D, f32>::zeros_like(input);
    let ones = Tensor::<D, f32>::ones_like(input);
    let cond = input.gt(&zeros);
    // alpha * (exp(x) - 1)
    let exp_x = input.exp();
    let neg_part = (&exp_x - &ones).scale(alpha);
    input.where_cond(&cond, &neg_part)
}

/// Applies the Softmax function along the specified axis.
///
/// `Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)`
///
/// For numerical stability, computes: `exp(x - max(x)) / Σ exp(x - max(x))`
///
/// # Arguments
/// * `input` - Input tensor of any shape
/// * `axis` - The axis along which to compute softmax
///
/// # Returns
/// Output tensor with the same shape as input, values sum to 1 along the specified axis
pub fn softmax<D: Dimension>(input: &Tensor<D, f32>, axis: usize) -> Tensor<D, f32>
where
    D::Smaller: Dimension<Larger = D>,
{
    // For numerical stability: softmax(x) = softmax(x - max(x))
    let max_val: Tensor<D::Smaller, f32> = input.max(axis);
    let max_broadcast: Tensor<D, f32> = max_val.unsqueeze(axis);
    let shifted = input - &max_broadcast;
    let exp_shifted = shifted.exp();
    let sum_exp: Tensor<D::Smaller, f32> = exp_shifted.sum(axis);
    let sum_broadcast: Tensor<D, f32> = sum_exp.unsqueeze(axis);
    &exp_shifted / &sum_broadcast
}

/// Applies the Log-Softmax function along the specified axis.
///
/// `LogSoftmax(x_i) = x_i - log(Σ_j exp(x_j))`
///
/// For numerical stability: `x_i - max(x) - log(Σ exp(x_j - max(x)))`
///
/// # Arguments
/// * `input` - Input tensor of any shape
/// * `axis` - The axis along which to compute log-softmax
///
/// # Returns
/// Output tensor with the same shape as input
pub fn log_softmax<D: Dimension>(input: &Tensor<D, f32>, axis: usize) -> Tensor<D, f32>
where
    D::Smaller: Dimension<Larger = D>,
{
    // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    let max_val: Tensor<D::Smaller, f32> = input.max(axis);
    let max_broadcast: Tensor<D, f32> = max_val.unsqueeze(axis);
    let shifted = input - &max_broadcast;
    let exp_shifted = shifted.exp();
    let sum_exp: Tensor<D::Smaller, f32> = exp_shifted.sum(axis);
    let log_sum_exp = sum_exp.ln();
    let log_sum_broadcast: Tensor<D, f32> = log_sum_exp.unsqueeze(axis);
    &shifted - &log_sum_broadcast
}

/// Applies PReLU (Parametric ReLU) function element-wise.
///
/// `PReLU(x) = max(0, x) + weight * min(0, x)`
///          = x if x >= 0, else weight * x
///
/// Unlike LeakyReLU, the weight parameter is learnable.
///
/// # Arguments
/// * `input` - Input tensor of any shape
/// * `weight` - Weight tensor with the same shape as input (pre-broadcast if needed)
///
/// # Returns
/// Output tensor with the same shape as input
pub fn prelu<D: Dimension>(input: &Tensor<D, f32>, weight: &Tensor<D, f32>) -> Tensor<D, f32> {
    // PReLU(x) = max(0, x) + weight * min(0, x)
    // = relu(x) - weight * relu(-x)
    let zeros = Tensor::<D, f32>::zeros_like(input);
    let pos = input.maximum(&zeros);
    let neg_input = input.neg();
    let neg_relu = neg_input.maximum(&zeros);
    let weighted_neg = weight * &neg_relu;
    &pos - &weighted_neg
}

#[cfg(test)]
mod tests {
    use super::*;
    use eclat::tensor::dim::D2;

    #[test]
    fn test_relu() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = relu(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_leaky_relu() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = leaky_relu(&input, 0.01);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_sigmoid() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = sigmoid(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_tanh() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = tanh(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_gelu() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = gelu(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_silu() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = silu(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_elu() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = elu(&input, 1.0);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_softmax() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = softmax(&input, 1);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_log_softmax() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = log_softmax(&input, 1);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_prelu() {
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        // Weight should be same shape as input (pre-broadcast)
        let weight: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = prelu(&input, &weight);
        assert_eq!(output.shape(), vec![2, 3]);
    }
}
