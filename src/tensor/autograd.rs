//! PyTorch-like autograd API for Tensor
//!
//! This module provides automatic differentiation capabilities:
//! - `requires_grad_()`: Enable/disable gradient tracking
//! - `requires_grad()`: Check gradient tracking status
//! - `backward()`: Compute gradients via backpropagation
//! - `grad()`: Access stored gradient
//! - `zero_grad()`: Clear accumulated gradients
//! - `detach()`: Create gradient-free copy
//!
//! # Example
//!
//! ```ignore
//! use eclat::tensor::{Tensor, D1, D0};
//!
//! let x: Tensor<D1, f32> = Tensor::input([4]);
//! x.requires_grad_(true);
//! x.set_data(&[1.0, 2.0, 3.0, 4.0])?;
//!
//! let y = &x * &x;  // y = x^2
//! let loss = y.sum(0);  // scalar
//!
//! loss.backward_with_params(&[&x])?;
//!
//! let grad = x.grad().unwrap();  // grad = 2*x
//! ```

use crate::ast::TensorDType;
use crate::grad::backward as compute_backward;
use crate::graph::GraphNode;

use super::dim::{D0, Dimension};
use super::tensor::Tensor;

// ============================================================================
// Gradient Tracking
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Enable or disable gradient computation for this tensor.
    ///
    /// When enabled, gradients will be computed during the backward pass
    /// and stored in `.grad()`.
    ///
    /// # Arguments
    ///
    /// * `requires_grad` - Whether to track gradients for this tensor
    ///
    /// # Returns
    ///
    /// Returns `&Self` for method chaining (PyTorch-style in-place operation).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// x.requires_grad_(true);
    /// assert!(x.requires_grad());
    /// ```
    pub fn requires_grad_(&self, requires_grad: bool) -> &Self {
        self.inner.requires_grad.set(requires_grad);
        self
    }

    /// Check if this tensor requires gradient computation.
    ///
    /// # Returns
    ///
    /// `true` if gradients will be computed for this tensor during backward pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// assert!(!x.requires_grad());  // Default is false
    /// x.requires_grad_(true);
    /// assert!(x.requires_grad());
    /// ```
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad.get()
    }

    /// Get the gradient of this tensor.
    ///
    /// Returns `None` if:
    /// - The tensor doesn't require gradients
    /// - `backward()` hasn't been called yet
    /// - The tensor wasn't in the computation graph of the backward pass
    ///
    /// # Returns
    ///
    /// `Some(Tensor)` containing the gradient, or `None` if no gradient exists.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// x.requires_grad_(true);
    /// x.set_data(&data)?;
    ///
    /// let y = &x * &x;
    /// let loss = y.sum(1).sum(0);
    ///
    /// loss.backward_with_params(&[&x])?;
    ///
    /// let grad = x.grad().unwrap();
    /// ```
    pub fn grad(&self) -> Option<Tensor<D, T>> {
        self.inner.grad.borrow().clone()
    }

    /// Clear the gradient stored in this tensor.
    ///
    /// Should be called before each backward pass if you don't want
    /// gradients to accumulate (similar to `optimizer.zero_grad()` in PyTorch).
    ///
    /// # Example
    ///
    /// ```ignore
    /// for epoch in 0..epochs {
    ///     x.zero_grad();  // Clear previous gradients
    ///
    ///     let loss = compute_loss(&x);
    ///     loss.backward_with_params(&[&x])?;
    ///
    ///     // Update x using x.grad()
    /// }
    /// ```
    pub fn zero_grad(&self) {
        *self.inner.grad.borrow_mut() = None;
    }

    /// Create a new tensor that shares data but doesn't track gradients.
    ///
    /// The returned tensor:
    /// - Shares the same underlying `GraphNode`
    /// - Has `requires_grad = false`
    /// - Won't have gradients computed during backward pass
    ///
    /// Note: Unlike PyTorch, this creates a new `Tensor` wrapper rather than
    /// modifying in place, due to Rust's ownership model.
    ///
    /// # Returns
    ///
    /// A new tensor with gradient tracking disabled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// x.requires_grad_(true);
    ///
    /// let x_detached = x.detach();
    /// assert!(!x_detached.requires_grad());
    /// ```
    pub fn detach(&self) -> Tensor<D, T> {
        // Create a new tensor from the same graph node
        // This creates a fresh TensorInner with requires_grad = false
        Tensor::from_graph(self.inner.graph.clone())
    }

    /// Set the gradient for this tensor directly.
    ///
    /// This is primarily used internally during backward pass,
    /// but can also be used for custom gradient manipulation.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor to store
    pub fn set_grad(&self, grad: Tensor<D, T>) {
        *self.inner.grad.borrow_mut() = Some(grad);
    }

    /// Accumulate gradient (add to existing gradient).
    ///
    /// Used internally when a tensor is used multiple times in computation.
    pub fn accumulate_grad(&self, grad: Tensor<D, T>) {
        let mut grad_ref = self.inner.grad.borrow_mut();
        if let Some(existing) = grad_ref.as_ref() {
            // Add gradients using GraphNode addition
            let accumulated_graph = &existing.inner.graph + &grad.inner.graph;
            *grad_ref = Some(Tensor::from_graph(accumulated_graph));
        } else {
            *grad_ref = Some(grad);
        }
    }
}

// ============================================================================
// Backward Pass for Scalar Tensors
// ============================================================================

impl<T: TensorDType> Tensor<D0, T> {
    /// Compute gradients via backpropagation from this scalar tensor.
    ///
    /// This method:
    /// 1. Finds all tensors with `requires_grad = true` in the computation graph
    /// 2. Computes gradients using reverse-mode autodiff
    /// 3. Stores gradients in each tensor's `.grad()` field
    ///
    /// Note: This method requires explicit parameter list. Use `backward_with_params()`
    /// and pass all tensors that need gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if backward pass computation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// x.requires_grad_(true);
    /// x.set_data(&data)?;
    ///
    /// let y = &x * &x;
    /// let loss = y.sum(1).sum(0);
    ///
    /// loss.backward_with_params(&[&x])?;
    ///
    /// let grad = x.grad().unwrap();
    /// ```
    pub fn backward(&self) -> Result<(), BackwardError> {
        // Without explicit parameters, we can't automatically find requires_grad tensors
        // because Tensor wrappers are not tracked globally.
        // Users must use backward_with_params() to specify which tensors need gradients.
        Err(BackwardError::NoParams(
            "backward() requires explicit parameters. Use backward_with_params(&[&tensor1, &tensor2, ...]) instead.".to_string()
        ))
    }

    /// Compute gradients with explicit parameter list.
    ///
    /// This is the primary method for computing gradients. Pass all tensors
    /// that should receive gradients.
    ///
    /// # Arguments
    ///
    /// * `params` - List of tensors to compute gradients for.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D1, f32> = Tensor::input([4]);
    /// let y: Tensor<D1, f32> = Tensor::input([4]);
    /// x.requires_grad_(true);
    /// y.requires_grad_(true);
    ///
    /// let z = &x * &y;
    /// let loss = z.sum(0);
    ///
    /// loss.backward_with_params(&[&x, &y])?;
    ///
    /// let dx = x.grad().unwrap();  // dy/dx
    /// let dy = y.grad().unwrap();  // dy/dy
    /// ```
    pub fn backward_with_params<D2: Dimension>(
        &self,
        params: &[&Tensor<D2, T>],
    ) -> Result<(), BackwardError> {
        if params.is_empty() {
            return Err(BackwardError::NoParams(
                "No parameters provided for backward pass.".to_string(),
            ));
        }

        // Collect GraphNodes from params
        let param_graphs: Vec<&GraphNode> = params.iter().map(|t| t.graph()).collect();

        // Call existing backward implementation
        let grad_result = compute_backward(&self.inner.graph, &param_graphs);

        // Store gradients back into tensors
        for param in params {
            if let Some(grad_node) = grad_result.get(param.graph()) {
                let grad_tensor: Tensor<D2, T> = Tensor::from_graph(grad_node);
                param.set_grad(grad_tensor);
            }
        }

        Ok(())
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during backward pass.
#[derive(Debug, Clone)]
pub enum BackwardError {
    /// Backward was called without parameters.
    NoParams(String),
    /// Backward was called on a non-scalar tensor.
    NonScalar(String),
    /// Computation graph is invalid or disconnected.
    InvalidGraph(String),
    /// Internal error during gradient computation.
    Internal(String),
}

impl std::fmt::Display for BackwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackwardError::NoParams(msg) => write!(f, "No parameters: {}", msg),
            BackwardError::NonScalar(msg) => write!(f, "Non-scalar tensor: {}", msg),
            BackwardError::InvalidGraph(msg) => write!(f, "Invalid graph: {}", msg),
            BackwardError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for BackwardError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dim::{D1, D2};

    #[test]
    fn test_requires_grad_default() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        assert!(!x.requires_grad());
    }

    #[test]
    fn test_requires_grad_set() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        x.requires_grad_(true);
        assert!(x.requires_grad());

        x.requires_grad_(false);
        assert!(!x.requires_grad());
    }

    #[test]
    fn test_grad_none_by_default() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        assert!(x.grad().is_none());
    }

    #[test]
    fn test_zero_grad() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        x.requires_grad_(true);

        // Manually set a gradient for testing
        let grad_tensor: Tensor<D2, f32> = Tensor::ones([32, 64]);
        x.set_grad(grad_tensor);

        assert!(x.grad().is_some());

        x.zero_grad();
        assert!(x.grad().is_none());
    }

    #[test]
    fn test_detach() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        x.requires_grad_(true);

        let x_detached = x.detach();
        assert!(!x_detached.requires_grad());

        // Original should still require grad
        assert!(x.requires_grad());
    }

    #[test]
    fn test_clone_shares_requires_grad() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        x.requires_grad_(true);

        let x_clone = x.clone();

        // Clone shares the same inner, so requires_grad should be shared
        assert!(x_clone.requires_grad());

        // Modifying one affects the other
        x_clone.requires_grad_(false);
        assert!(!x.requires_grad());
    }

    #[test]
    fn test_backward_requires_params() {
        let x: Tensor<D1, f32> = Tensor::input([4]);
        x.requires_grad_(true);

        let y = &x * &x;
        let loss = y.sum(0);

        // backward() without params should return an error
        let result = loss.backward();
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_with_params() {
        // y = x * x, loss = sum(y)
        // dy/dx = 2x
        let x: Tensor<D1, f32> = Tensor::input([4]);
        x.requires_grad_(true);

        let y = &x * &x;
        let loss = y.sum(0);

        let result = loss.backward_with_params(&[&x]);
        assert!(result.is_ok());

        // Check gradient exists
        let grad = x.grad();
        assert!(grad.is_some());

        // Gradient should have same shape as x
        let grad = grad.unwrap();
        assert_eq!(grad.shape(), vec![4]);
    }

    #[test]
    fn test_backward_multiple_params() {
        // z = a * b, loss = sum(z)
        // dz/da = b, dz/db = a
        let a: Tensor<D1, f32> = Tensor::input([4]);
        let b: Tensor<D1, f32> = Tensor::input([4]);
        a.requires_grad_(true);
        b.requires_grad_(true);

        let z = &a * &b;
        let loss = z.sum(0);

        loss.backward_with_params(&[&a, &b]).unwrap();

        assert!(a.grad().is_some());
        assert!(b.grad().is_some());

        assert_eq!(a.grad().unwrap().shape(), vec![4]);
        assert_eq!(b.grad().unwrap().shape(), vec![4]);
    }

    #[test]
    fn test_gradient_accumulation() {
        let x: Tensor<D1, f32> = Tensor::input([4]);
        x.requires_grad_(true);

        // Set initial gradient
        let grad1: Tensor<D1, f32> = Tensor::ones([4]);
        x.set_grad(grad1);

        // Accumulate another gradient
        let grad2: Tensor<D1, f32> = Tensor::ones([4]);
        x.accumulate_grad(grad2);

        // Gradient should exist
        let final_grad = x.grad();
        assert!(final_grad.is_some());
        assert_eq!(final_grad.unwrap().shape(), vec![4]);
    }

    #[test]
    fn test_backward_empty_params() {
        let x: Tensor<D1, f32> = Tensor::input([4]);
        let y = &x * &x;
        let loss = y.sum(0);

        // Empty params should return an error
        let result = loss.backward_with_params::<D1>(&[]);
        assert!(result.is_err());
    }
}
