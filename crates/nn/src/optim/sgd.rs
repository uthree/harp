//! Stochastic Gradient Descent Optimizer
//!
//! Implements SGD with optional momentum.

use super::{OptimError, Optimizer, get_param_data};
use crate::layers::ParameterBase;

/// Stochastic Gradient Descent optimizer.
///
/// Implements the update rule:
/// - Without momentum: `param = param - lr * grad`
/// - With momentum: `v = momentum * v - lr * grad; param = param + v`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::optim::{Optimizer, SGD};
///
/// let params = model.parameters();
///
/// // Simple SGD
/// let mut sgd = SGD::new(params.clone(), 0.01);
///
/// // SGD with momentum
/// let mut sgd_momentum = SGD::with_momentum(params, 0.01, 0.9);
/// ```
pub struct SGD {
    /// Parameters to optimize
    params: Vec<Box<dyn ParameterBase>>,
    /// Learning rate
    lr: f32,
    /// Momentum factor (0 = no momentum)
    momentum: f32,
    /// Velocity buffers for momentum (one per parameter)
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer without momentum.
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    pub fn new(params: Vec<Box<dyn ParameterBase>>, lr: f32) -> Self {
        Self::with_momentum(params, lr, 0.0)
    }

    /// Create a new SGD optimizer with momentum.
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum factor (typically 0.9)
    pub fn with_momentum(params: Vec<Box<dyn ParameterBase>>, lr: f32, momentum: f32) -> Self {
        // Initialize velocity buffers to zeros
        let velocity: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();

        Self {
            params,
            lr,
            momentum,
            velocity,
        }
    }

    /// Get the momentum factor.
    pub fn momentum(&self) -> f32 {
        self.momentum
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<(), OptimError> {
        for (i, param) in self.params.iter().enumerate() {
            let data = get_param_data(param.as_ref())?;
            let param_data = data.param_data;
            let grad_data = data.grad_data;

            // Compute update
            let mut new_data = vec![0.0f32; param_data.len()];

            if self.momentum > 0.0 {
                // With momentum: v = momentum * v - lr * grad; param = param + v
                for j in 0..param_data.len() {
                    self.velocity[i][j] =
                        self.momentum * self.velocity[i][j] - self.lr * grad_data[j];
                    new_data[j] = param_data[j] + self.velocity[i][j];
                }
            } else {
                // Without momentum: param = param - lr * grad
                for j in 0..param_data.len() {
                    new_data[j] = param_data[j] - self.lr * grad_data[j];
                }
            }

            // Update parameter
            param
                .update_data(&new_data)
                .map_err(|e| OptimError::ExecutionError(e.to_string()))?;
        }

        Ok(())
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl std::fmt::Debug for SGD {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SGD")
            .field("lr", &self.lr)
            .field("momentum", &self.momentum)
            .field("num_params", &self.params.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Parameter;
    use eclat::tensor::dim::D2;

    #[test]
    fn test_sgd_creation() {
        let params: Vec<Box<dyn ParameterBase>> =
            vec![Box::new(Parameter::<D2>::new("test", &[2, 3]))];
        let sgd = SGD::new(params, 0.01);

        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum(), 0.0);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let params: Vec<Box<dyn ParameterBase>> =
            vec![Box::new(Parameter::<D2>::new("test", &[2, 3]))];
        let sgd = SGD::with_momentum(params, 0.01, 0.9);

        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum(), 0.9);
    }

    #[test]
    fn test_sgd_set_learning_rate() {
        let params: Vec<Box<dyn ParameterBase>> =
            vec![Box::new(Parameter::<D2>::new("test", &[2, 3]))];
        let mut sgd = SGD::new(params, 0.01);

        sgd.set_learning_rate(0.001);
        assert_eq!(sgd.learning_rate(), 0.001);
    }
}
