//! Linear (Fully Connected) Layer
//!
//! Implements `y = xW^T + b` where:
//! - x: input tensor [batch, in_features]
//! - W: weight matrix [out_features, in_features]
//! - b: bias vector [out_features] (optional)

use super::{Module, Parameter, ParameterBase};
use eclat::tensor::Tensor;
use eclat::tensor::dim::{D1, D2};

/// A fully connected (linear) layer.
///
/// Computes `y = xW^T + b` using unsqueeze, broadcast multiply, and sum operations.
///
/// # Example
///
/// ```ignore
/// use eclat_nn::nn::{Module, Linear};
/// use eclat::tensor::{Tensor, dim::D2};
///
/// // Create a layer with 10 inputs, 5 outputs, and bias
/// let layer = Linear::new(10, 5, true);
///
/// // Forward pass with static dimensions
/// let input: Tensor<D2, f32> = Tensor::input([32, 10]); // batch=32, features=10
/// let output = layer.forward_d2(&input);  // [32, 5]
///
/// // Get parameters for optimization
/// let params = layer.parameters();
/// assert_eq!(params.len(), 2); // weight + bias
/// ```
pub struct Linear {
    /// Weight parameter [out_features, in_features]
    weight: Parameter<D2>,
    /// Bias parameter [out_features] (optional)
    bias: Option<Parameter<D1>>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Training mode flag
    training: bool,
}

impl Linear {
    /// Create a new Linear layer.
    ///
    /// # Arguments
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    /// * `bias` - If true, adds a learnable bias
    ///
    /// Weights are initialized using Kaiming uniform initialization.
    /// Bias is initialized to zeros.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Kaiming uniform initialization: U(-sqrt(1/in), sqrt(1/in))
        let bound = (1.0 / in_features as f32).sqrt();

        // Initialize weight with simple random values
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| {
                // Simple deterministic pseudo-random for reproducibility
                (i as f32 * 0.1).sin() * bound
            })
            .collect();
        let weight: Parameter<D2> =
            Parameter::from_data("weight", &weight_data, &[out_features, in_features]);

        // Initialize bias (zeros)
        let bias = if bias {
            let bias_data = vec![0.0f32; out_features];
            Some(Parameter::from_data("bias", &bias_data, &[out_features]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
            training: true,
        }
    }

    /// Compute y = xW^T + b with static 2D dimensions.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, in_features]
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
    pub fn forward_d2(&self, input: &Tensor<D2, f32>) -> Tensor<D2, f32> {
        // Get weight tensor and transpose it: [O, K] -> [K, O]
        let weight = self.weight.tensor();
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
        match &self.bias {
            Some(bias) => {
                let bias_tensor = bias.tensor();
                // bias: [O] -> [1, O]
                let bias_expanded = bias_tensor.unsqueeze(0);
                &y + &bias_expanded
            }
            None => y,
        }
    }

    /// Get the input features count.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get the output features count.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get a reference to the weight parameter.
    pub fn weight(&self) -> &Parameter<D2> {
        &self.weight
    }

    /// Get a reference to the bias parameter, if present.
    pub fn bias(&self) -> Option<&Parameter<D1>> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        let mut params: Vec<Box<dyn ParameterBase>> = vec![Box::new(self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(Box::new(b.clone()));
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        let mut params: Vec<(String, Box<dyn ParameterBase>)> =
            vec![("weight".to_string(), Box::new(self.weight.clone()))];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), Box::new(b.clone())));
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let layer = Linear::new(10, 5, true);
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
    }

    #[test]
    fn test_linear_parameters() {
        let layer_with_bias = Linear::new(10, 5, true);
        assert_eq!(layer_with_bias.parameters().len(), 2);

        let layer_no_bias = Linear::new(10, 5, false);
        assert_eq!(layer_no_bias.parameters().len(), 1);
    }

    #[test]
    fn test_linear_named_parameters() {
        let layer = Linear::new(10, 5, true);
        let named = layer.named_parameters();

        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_linear_num_parameters() {
        let layer = Linear::new(10, 5, true);
        // weight: 10 * 5 = 50, bias: 5
        assert_eq!(layer.num_parameters(), 55);

        let layer_no_bias = Linear::new(10, 5, false);
        assert_eq!(layer_no_bias.num_parameters(), 50);
    }

    #[test]
    fn test_linear_forward_d2() {
        let layer = Linear::new(10, 5, true);
        let input: Tensor<D2, f32> = Tensor::input([4, 10]);
        let output = layer.forward_d2(&input);
        assert_eq!(output.shape(), vec![4, 5]);
    }
}
