//! Activation layers with learnable parameters
//!
//! This module provides activation function layers that have learnable parameters.

use super::{Module, Parameter, ParameterBase};
use crate::functional;
use eclat::tensor::Tensor;
use eclat::tensor::dim::{D1, D2};

/// PReLU (Parametric ReLU) layer.
///
/// Applies the function: `PReLU(x) = max(0, x) + weight * min(0, x)`
///
/// Unlike LeakyReLU with a fixed negative slope, PReLU has a learnable weight parameter.
///
/// # Shape
/// - Input: `[N, C]` where N is batch size and C is number of features
/// - Output: `[N, C]` same shape as input
///
/// # Example
///
/// ```ignore
/// use eclat_nn::layers::PReLU;
/// use eclat::tensor::{Tensor, dim::D2};
///
/// // Single weight shared across all features
/// let prelu = PReLU::new(1);
///
/// // Per-feature weights
/// let prelu = PReLU::new(64);
///
/// let input: Tensor<D2, f32> = Tensor::input([32, 64]);
/// let output = prelu.forward(&input);
/// ```
pub struct PReLU {
    /// Learnable weight parameter [num_parameters]
    weight: Parameter<D1>,
    /// Number of parameters (1 for shared, or num_features for per-channel)
    num_parameters: usize,
    /// Initial value for weight
    init: f32,
    /// Training mode flag
    training: bool,
}

impl PReLU {
    /// Create a new PReLU layer.
    ///
    /// # Arguments
    /// * `num_parameters` - Number of weight parameters:
    ///   - 1: a single parameter shared across all input channels
    ///   - num_features: a parameter for each input channel
    ///
    /// The weight is initialized to 0.25 (default PyTorch initialization).
    pub fn new(num_parameters: usize) -> Self {
        Self::with_init(num_parameters, 0.25)
    }

    /// Create a new PReLU layer with custom weight initialization.
    ///
    /// # Arguments
    /// * `num_parameters` - Number of weight parameters
    /// * `init` - Initial value for all weight parameters
    pub fn with_init(num_parameters: usize, init: f32) -> Self {
        let weight_data = vec![init; num_parameters];
        let weight = Parameter::from_data("weight", &weight_data, &[num_parameters]);

        Self {
            weight,
            num_parameters,
            init,
            training: true,
        }
    }

    /// Forward pass for 2D input tensors.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, features]
    ///
    /// # Returns
    /// Output tensor of shape [batch, features]
    pub fn forward(&self, input: &Tensor<D2, f32>) -> Tensor<D2, f32> {
        // Broadcast weight [features] -> [1, features] for proper broadcasting
        let weight_broadcast: Tensor<D2, f32> = self.weight.tensor().unsqueeze(0);
        functional::prelu(input, &weight_broadcast)
    }

    /// Get the number of parameters.
    pub fn num_parameters_count(&self) -> usize {
        self.num_parameters
    }

    /// Get the initial value.
    pub fn init(&self) -> f32 {
        self.init
    }
}

impl Module for PReLU {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![Box::new(self.weight.clone())]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![("weight".to_string(), Box::new(self.weight.clone()))]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for PReLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PReLU")
            .field("num_parameters", &self.num_parameters)
            .field("init", &self.init)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelu_creation() {
        let prelu = PReLU::new(1);
        assert_eq!(prelu.num_parameters_count(), 1);
        assert_eq!(prelu.init(), 0.25);
    }

    #[test]
    fn test_prelu_with_init() {
        let prelu = PReLU::with_init(64, 0.1);
        assert_eq!(prelu.num_parameters_count(), 64);
        assert_eq!(prelu.init(), 0.1);
    }

    #[test]
    fn test_prelu_parameters() {
        let prelu = PReLU::new(64);
        let params = prelu.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(prelu.num_parameters(), 64);
    }

    #[test]
    fn test_prelu_forward() {
        let prelu = PReLU::new(3);
        let input: Tensor<D2, f32> = Tensor::input([2, 3]);
        let output = prelu.forward(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_prelu_shared_weight() {
        // Single shared weight
        let prelu = PReLU::new(1);
        let input: Tensor<D2, f32> = Tensor::input([4, 10]);
        let output = prelu.forward(&input);
        assert_eq!(output.shape(), vec![4, 10]);
    }
}
