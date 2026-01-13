//! Optimizer Trait
//!
//! Base trait for all optimization algorithms.

use crate::nn::Parameter;

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

/// Helper struct for managing parameter references in optimizers.
#[derive(Clone)]
pub struct ParamGroup {
    /// Parameters in this group
    pub params: Vec<Parameter>,
    /// Learning rate for this group (overrides optimizer default if set)
    pub lr: Option<f32>,
}

impl ParamGroup {
    /// Create a new parameter group with default settings.
    pub fn new(params: Vec<Parameter>) -> Self {
        Self { params, lr: None }
    }

    /// Create a parameter group with a specific learning rate.
    pub fn with_lr(params: Vec<Parameter>, lr: f32) -> Self {
        Self {
            params,
            lr: Some(lr),
        }
    }
}
