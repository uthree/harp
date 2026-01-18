//! Module Trait
//!
//! The `Module` trait is the base trait for all neural network layers.
//! It provides a unified interface for parameter management and forward pass.

use super::Parameter;
use eclat::tensor::{Dyn, Tensor};

/// Base trait for neural network modules.
///
/// All neural network layers implement this trait, providing:
/// - Forward pass computation
/// - Parameter enumeration
/// - Training mode control
///
/// # Example
///
/// ```ignore
/// use eclat_nn::nn::{Module, Linear};
///
/// let layer = Linear::new(10, 5, true);
///
/// // Get parameters for optimization
/// let params = layer.parameters();
/// println!("Number of parameters: {}", params.len());
///
/// // Set training mode
/// layer.train(true);
/// ```
pub trait Module {
    /// Perform the forward pass.
    ///
    /// # Arguments
    /// * `input` - Input tensor with dynamic dimensions
    ///
    /// # Returns
    /// Output tensor after applying this module's transformation.
    fn forward(&self, input: &Tensor<Dyn, f32>) -> Tensor<Dyn, f32>;

    /// Returns all learnable parameters of this module (including submodules).
    fn parameters(&self) -> Vec<Parameter>;

    /// Returns named parameters as (name, parameter) pairs.
    ///
    /// The names include hierarchical prefixes for nested modules.
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.parameters()
            .into_iter()
            .map(|p| (p.name().to_string(), p))
            .collect()
    }

    /// Set the module to training or evaluation mode.
    ///
    /// This affects layers like Dropout and BatchNorm.
    fn train(&mut self, mode: bool);

    /// Check if the module is in training mode.
    fn is_training(&self) -> bool;

    /// Set to evaluation mode. Equivalent to `train(false)`.
    fn eval(&mut self) {
        self.train(false);
    }

    /// Zero gradients of all parameters.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Get the number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eclat::tensor::{Dyn, Tensor};

    // Simple test module for testing the trait
    struct DummyModule {
        param: Parameter,
        training: bool,
    }

    impl DummyModule {
        fn new() -> Self {
            Self {
                param: Parameter::new("dummy", &[2, 3]),
                training: true,
            }
        }
    }

    impl Module for DummyModule {
        fn forward(&self, input: &Tensor<Dyn, f32>) -> Tensor<Dyn, f32> {
            // Identity function for testing
            Tensor::from_graph(input.graph().clone())
        }

        fn parameters(&self) -> Vec<Parameter> {
            vec![self.param.clone()]
        }

        fn train(&mut self, mode: bool) {
            self.training = mode;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    #[test]
    fn test_module_parameters() {
        let module = DummyModule::new();
        let params = module.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name(), "dummy");
    }

    #[test]
    fn test_module_train_eval() {
        let mut module = DummyModule::new();
        assert!(module.is_training());

        module.eval();
        assert!(!module.is_training());

        module.train(true);
        assert!(module.is_training());
    }

    #[test]
    fn test_module_num_parameters() {
        let module = DummyModule::new();
        assert_eq!(module.num_parameters(), 6); // 2 * 3 = 6
    }
}
