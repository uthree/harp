//! Adam Optimizer
//!
//! Implements the Adam (Adaptive Moment Estimation) algorithm.
//!
//! Reference: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.

use super::{OptimError, Optimizer, get_param_data};
use crate::layers::ParameterBase;

/// Adam optimizer.
///
/// Implements the Adam algorithm with bias correction:
/// - m = β1 * m + (1 - β1) * grad
/// - v = β2 * v + (1 - β2) * grad²
/// - m_hat = m / (1 - β1^t)
/// - v_hat = v / (1 - β2^t)
/// - param = param - lr * m_hat / (√v_hat + ε)
///
/// # Example
///
/// ```ignore
/// use eclat_nn::optim::{Optimizer, Adam};
///
/// let params = model.parameters();
///
/// // Default Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-8)
/// let mut adam = Adam::new(params.clone(), 0.001);
///
/// // Custom hyperparameters
/// let mut adam_custom = Adam::with_betas(params, 0.001, (0.9, 0.999), 1e-8);
/// ```
pub struct Adam {
    /// Parameters to optimize
    params: Vec<Box<dyn ParameterBase>>,
    /// Learning rate
    lr: f32,
    /// Exponential decay rates for moment estimates (β1, β2)
    betas: (f32, f32),
    /// Small constant for numerical stability
    eps: f32,
    /// First moment estimates (one per parameter)
    m: Vec<Vec<f32>>,
    /// Second moment estimates (one per parameter)
    v: Vec<Vec<f32>>,
    /// Time step counter
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer with default betas (0.9, 0.999) and eps (1e-8).
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (typically 0.001)
    pub fn new(params: Vec<Box<dyn ParameterBase>>, lr: f32) -> Self {
        Self::with_betas(params, lr, (0.9, 0.999), 1e-8)
    }

    /// Create a new Adam optimizer with custom hyperparameters.
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `betas` - Exponential decay rates (β1, β2) for moment estimates
    /// * `eps` - Small constant for numerical stability
    pub fn with_betas(
        params: Vec<Box<dyn ParameterBase>>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
    ) -> Self {
        // Initialize moment buffers to zeros
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();

        Self {
            params,
            lr,
            betas,
            eps,
            m,
            v,
            t: 0,
        }
    }

    /// Get the beta values (β1, β2).
    pub fn betas(&self) -> (f32, f32) {
        self.betas
    }

    /// Get the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get the current time step.
    pub fn time_step(&self) -> usize {
        self.t
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<(), OptimError> {
        // Increment time step
        self.t += 1;
        let (beta1, beta2) = self.betas;

        // Precompute bias correction terms
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);

        for (i, param) in self.params.iter().enumerate() {
            let data = get_param_data(param.as_ref())?;
            let param_data = data.param_data;
            let grad_data = data.grad_data;

            // Compute update
            let mut new_data = vec![0.0f32; param_data.len()];

            for j in 0..param_data.len() {
                let g = grad_data[j];

                // Update biased first moment estimate: m = β1 * m + (1 - β1) * g
                self.m[i][j] = beta1 * self.m[i][j] + (1.0 - beta1) * g;

                // Update biased second moment estimate: v = β2 * v + (1 - β2) * g²
                self.v[i][j] = beta2 * self.v[i][j] + (1.0 - beta2) * g * g;

                // Compute bias-corrected estimates
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;

                // Update parameter: param = param - lr * m_hat / (√v_hat + ε)
                new_data[j] = param_data[j] - self.lr * m_hat / (v_hat.sqrt() + self.eps);
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

impl std::fmt::Debug for Adam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adam")
            .field("lr", &self.lr)
            .field("betas", &self.betas)
            .field("eps", &self.eps)
            .field("t", &self.t)
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
    fn test_adam_creation() {
        let params: Vec<Box<dyn ParameterBase>> =
            vec![Box::new(Parameter::<D2>::new("test", &[2, 3]))];
        let adam = Adam::new(params, 0.001);

        assert_eq!(adam.learning_rate(), 0.001);
        assert_eq!(adam.betas(), (0.9, 0.999));
        assert_eq!(adam.eps(), 1e-8);
        assert_eq!(adam.time_step(), 0);
    }

    #[test]
    fn test_adam_with_betas() {
        let params: Vec<Box<dyn ParameterBase>> =
            vec![Box::new(Parameter::<D2>::new("test", &[2, 3]))];
        let adam = Adam::with_betas(params, 0.002, (0.8, 0.99), 1e-7);

        assert_eq!(adam.learning_rate(), 0.002);
        assert_eq!(adam.betas(), (0.8, 0.99));
        assert_eq!(adam.eps(), 1e-7);
    }

    #[test]
    fn test_adam_set_learning_rate() {
        let params: Vec<Box<dyn ParameterBase>> =
            vec![Box::new(Parameter::<D2>::new("test", &[2, 3]))];
        let mut adam = Adam::new(params, 0.001);

        adam.set_learning_rate(0.0001);
        assert_eq!(adam.learning_rate(), 0.0001);
    }
}
