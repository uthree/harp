//! Optimizer Trait
//!
//! Base trait for all optimization algorithms.

/// Base trait for optimizers.
///
/// All optimizers implement this trait, providing:
/// - `step()`: Update parameters based on gradients
/// - `zero_grad()`: Reset gradients to zero
///
/// # Example
///
/// ```ignore
/// use eclat_nn::optim::{Optimizer, SGD};
///
/// let mut optimizer = SGD::new(params, 0.01);
///
/// // Training step
/// optimizer.zero_grad();
/// // ... compute gradients ...
/// optimizer.step().unwrap();
/// ```
pub trait Optimizer {
    /// Perform a single optimization step.
    ///
    /// Updates all parameters based on their computed gradients.
    ///
    /// # Returns
    /// `Ok(())` on success, or an error if gradients are missing or invalid.
    fn step(&mut self) -> Result<(), OptimError>;

    /// Set gradients of all parameters to zero.
    ///
    /// Should be called before each backward pass to prevent gradient accumulation.
    fn zero_grad(&self);

    /// Get the current learning rate.
    fn learning_rate(&self) -> f32;

    /// Set the learning rate.
    fn set_learning_rate(&mut self, lr: f32);
}

/// Errors that can occur during optimization.
#[derive(Debug)]
pub enum OptimError {
    /// A parameter has no gradient computed.
    NoGradient(String),
    /// Shape mismatch between parameter and gradient.
    ShapeMismatch {
        param_name: String,
        expected: usize,
        got: usize,
    },
    /// Error during tensor execution.
    ExecutionError(String),
}

impl std::fmt::Display for OptimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoGradient(name) => {
                write!(f, "No gradient computed for parameter '{}'", name)
            }
            Self::ShapeMismatch {
                param_name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Shape mismatch for parameter '{}': expected {} elements, got {}",
                    param_name, expected, got
                )
            }
            Self::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
        }
    }
}

impl std::error::Error for OptimError {}

use crate::nn::Parameter;

/// Data extracted from a parameter for optimization.
pub struct ParamData {
    /// Current parameter values
    pub param_data: Vec<f32>,
    /// Gradient values
    pub grad_data: Vec<f32>,
}

/// Extract gradient and parameter data for optimization step.
///
/// This helper function:
/// 1. Gets the gradient from the parameter
/// 2. Realizes the gradient tensor
/// 3. Extracts gradient and parameter data as vectors
/// 4. Validates that shapes match
pub fn get_param_data(param: &Parameter) -> Result<ParamData, OptimError> {
    // Get gradient
    let grad = param
        .grad()
        .ok_or_else(|| OptimError::NoGradient(param.name().to_string()))?;

    // Realize gradient and get data
    grad.realize()
        .map_err(|e| OptimError::ExecutionError(e.to_string()))?;
    let grad_data = grad
        .to_vec()
        .map_err(|e| OptimError::ExecutionError(e.to_string()))?;

    // Get current parameter data
    let param_data = param
        .to_vec()
        .map_err(|e| OptimError::ExecutionError(e.to_string()))?;

    // Check shapes match
    if grad_data.len() != param_data.len() {
        return Err(OptimError::ShapeMismatch {
            param_name: param.name().to_string(),
            expected: param_data.len(),
            got: grad_data.len(),
        });
    }

    Ok(ParamData {
        param_data,
        grad_data,
    })
}