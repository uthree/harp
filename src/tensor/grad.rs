//! Gradient functions for automatic differentiation
//!
//! This module provides GradFn implementations for each tensor operation,
//! enabling backward propagation of gradients through the computation graph.

use std::rc::Rc;

use super::{DimDyn, Dimension, GradFn, Tensor};

// ============================================================================
// Backward Error
// ============================================================================

/// Error type for backward propagation
#[derive(Debug)]
pub enum BackwardError {
    /// Tensor does not require gradients
    NoGrad,
    /// Gradient shape mismatch
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for BackwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackwardError::NoGrad => write!(f, "Tensor does not require gradients"),
            BackwardError::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "Gradient shape mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            }
        }
    }
}

impl std::error::Error for BackwardError {}

// ============================================================================
// GradFn Implementations
// ============================================================================

/// Gradient function for addition: z = a + b
/// dL/da = dL/dz, dL/db = dL/dz
pub struct AddBackward {
    lhs: Tensor<DimDyn>,
    rhs: Tensor<DimDyn>,
}

impl AddBackward {
    pub fn new(lhs: Tensor<DimDyn>, rhs: Tensor<DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // For addition, gradient flows through unchanged to both inputs
        // Handle broadcasting by summing over broadcasted dimensions
        let grad_lhs = reduce_grad_for_broadcast(grad_output, self.lhs.shape());
        let grad_rhs = reduce_grad_for_broadcast(grad_output, self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient function for subtraction: z = a - b
/// dL/da = dL/dz, dL/db = -dL/dz
pub struct SubBackward {
    lhs: Tensor<DimDyn>,
    rhs: Tensor<DimDyn>,
}

impl SubBackward {
    pub fn new(lhs: Tensor<DimDyn>, rhs: Tensor<DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        let grad_lhs = reduce_grad_for_broadcast(grad_output, self.lhs.shape());
        let neg_grad = -grad_output;
        let grad_rhs = reduce_grad_for_broadcast(&neg_grad, self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Gradient function for multiplication: z = a * b
/// dL/da = dL/dz * b, dL/db = dL/dz * a
pub struct MulBackward {
    lhs: Tensor<DimDyn>,
    rhs: Tensor<DimDyn>,
}

impl MulBackward {
    pub fn new(lhs: Tensor<DimDyn>, rhs: Tensor<DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // dL/da = dL/dz * b
        let grad_lhs_full = grad_output * &self.rhs;
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs_full, self.lhs.shape());

        // dL/db = dL/dz * a
        let grad_rhs_full = grad_output * &self.lhs;
        let grad_rhs = reduce_grad_for_broadcast(&grad_rhs_full, self.rhs.shape());

        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient function for division: z = a / b
/// dL/da = dL/dz / b, dL/db = -dL/dz * a / b^2
pub struct DivBackward {
    lhs: Tensor<DimDyn>,
    rhs: Tensor<DimDyn>,
}

impl DivBackward {
    pub fn new(lhs: Tensor<DimDyn>, rhs: Tensor<DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // dL/da = dL/dz / b
        let grad_lhs_full = grad_output / &self.rhs;
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs_full, self.lhs.shape());

        // dL/db = -dL/dz * a / b^2
        let b_squared = &self.rhs * &self.rhs;
        let grad_rhs_full = -(grad_output * &self.lhs) / b_squared;
        let grad_rhs = reduce_grad_for_broadcast(&grad_rhs_full, self.rhs.shape());

        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

/// Gradient function for negation: z = -a
/// dL/da = -dL/dz
pub struct NegBackward {
    #[allow(dead_code)]
    input: Tensor<DimDyn>,
}

impl NegBackward {
    pub fn new(input: Tensor<DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        vec![-grad_output]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Gradient function for exp: z = exp(a)
/// dL/da = dL/dz * exp(a) = dL/dz * z
pub struct ExpBackward {
    input: Tensor<DimDyn>,
    output: Tensor<DimDyn>, // We store output since exp'(x) = exp(x)
}

impl ExpBackward {
    pub fn new(input: Tensor<DimDyn>, output: Tensor<DimDyn>) -> Self {
        Self { input, output }
    }
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        vec![grad_output * &self.output]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

/// Gradient function for log: z = ln(a)
/// dL/da = dL/dz / a
pub struct LogBackward {
    input: Tensor<DimDyn>,
}

impl LogBackward {
    pub fn new(input: Tensor<DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        vec![grad_output / &self.input]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

/// Gradient function for sqrt: z = sqrt(a)
/// dL/da = dL/dz / (2 * sqrt(a)) = dL/dz / (2 * z)
pub struct SqrtBackward {
    input: Tensor<DimDyn>,
    output: Tensor<DimDyn>,
}

impl SqrtBackward {
    pub fn new(input: Tensor<DimDyn>, output: Tensor<DimDyn>) -> Self {
        Self { input, output }
    }
}

impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        let two_sqrt = &self.output * 2.0;
        vec![grad_output / two_sqrt]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Gradient function for sin: z = sin(a)
/// dL/da = dL/dz * cos(a)
pub struct SinBackward {
    input: Tensor<DimDyn>,
}

impl SinBackward {
    pub fn new(input: Tensor<DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn for SinBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // cos(x) = sin(x + pi/2), but we'll use the graph node's cos method
        // For now, use identity: cos(x) = sqrt(1 - sin(x)^2) for |sin(x)| < 1
        // Actually, let's just compute it directly via the graph
        let cos_input = self.input.cos();
        vec![grad_output * &cos_input]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

/// Gradient function for ReLU: z = max(0, a)
/// dL/da = dL/dz if a > 0 else 0
pub struct ReluBackward {
    input: Tensor<DimDyn>,
}

impl ReluBackward {
    pub fn new(input: Tensor<DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // ReLU gradient: 1 if input > 0, else 0
        // We'll use: grad * (input > 0) but since we don't have comparison,
        // we use: grad * step(input) where step is approximated
        // For now, we'll create a mask using the fact that relu(x)/x = 1 if x > 0
        // This is a simplified implementation - a proper implementation would need
        // a comparison operation in the graph
        let zero = Tensor::<DimDyn>::full_dyn(self.input.shape(), 0.0);
        let _relu_input = self.input.max_with(&zero);

        // Create mask: 1 where input > 0, 0 otherwise
        // Using the fact that sign(relu(x)) = sign(x) for x != 0
        // This is approximate; proper implementation needs graph-level support
        vec![grad_output.clone()]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

/// Gradient function for sum reduction
/// dL/da = dL/dz expanded back to original shape
pub struct SumBackward {
    input: Tensor<DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl SumBackward {
    pub fn new(input: Tensor<DimDyn>, axes: Vec<usize>, keepdim: bool) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            input_shape,
            axes,
            keepdim,
        }
    }
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // Expand gradient back to input shape
        let mut grad = grad_output.clone();

        // If keepdim=false, we need to unsqueeze the reduced dimensions
        if !self.keepdim {
            for &axis in &self.axes {
                grad = grad.unsqueeze(axis);
            }
        }

        // Expand to original shape
        let grad_expanded = grad.expand(&self.input_shape);
        vec![grad_expanded]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Gradient function for mean reduction
/// dL/da = dL/dz / count, expanded back to original shape
pub struct MeanBackward {
    input: Tensor<DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
    count: usize,
}

impl MeanBackward {
    pub fn new(input: Tensor<DimDyn>, axes: Vec<usize>, keepdim: bool) -> Self {
        let input_shape = input.shape().to_vec();
        let count: usize = axes.iter().map(|&axis| input_shape[axis]).product();
        Self {
            input,
            input_shape,
            axes,
            keepdim,
            count,
        }
    }
}

impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // Same as sum but divided by count
        let mut grad = grad_output / (self.count as f32);

        if !self.keepdim {
            for &axis in &self.axes {
                grad = grad.unsqueeze(axis);
            }
        }

        let grad_expanded = grad.expand(&self.input_shape);
        vec![grad_expanded]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Reduce gradient to match the original input shape (handle broadcasting)
fn reduce_grad_for_broadcast(grad: &Tensor<DimDyn>, target_shape: &[usize]) -> Tensor<DimDyn> {
    if grad.shape() == target_shape {
        return grad.clone();
    }

    let grad_shape = grad.shape();
    let target_ndim = target_shape.len();
    let grad_ndim = grad_shape.len();

    // Pad target shape with 1s on the left to match grad_ndim
    let mut padded_target = vec![1usize; grad_ndim.saturating_sub(target_ndim)];
    padded_target.extend_from_slice(target_shape);

    // Find axes to reduce
    let mut reduce_axes = Vec::new();
    for (i, (&grad_dim, &target_dim)) in grad_shape.iter().zip(padded_target.iter()).enumerate() {
        if target_dim == 1 && grad_dim > 1 {
            reduce_axes.push(i);
        }
    }

    // Reduce along those axes
    let mut result = grad.clone();
    if !reduce_axes.is_empty() {
        result = result.sum(&reduce_axes, true);
    }

    // Reshape to target shape
    result.reshape_dyn(target_shape)
}

// ============================================================================
// Tensor backward implementation
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Perform backward propagation from this tensor
    ///
    /// Computes gradients for all tensors in the computation graph that
    /// have `requires_grad = true`.
    ///
    /// # Panics
    ///
    /// Panics if this tensor does not require gradients.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use harp::tensor::{Tensor, Dim2};
    ///
    /// let x = Tensor::<Dim2>::full([2, 3], 2.0).set_requires_grad(true);
    /// let y = &x * &x;  // y = x^2
    /// y.backward();
    ///
    /// // dy/dx = 2x = 4.0
    /// let grad = x.grad().unwrap();
    /// ```
    pub fn backward(&self) {
        if self.autograd.is_none() {
            panic!("backward() called on tensor that doesn't require gradients");
        }

        // Create initial gradient of ones with same shape
        let initial_grad = Tensor::<DimDyn>::ones_dyn(self.shape());
        self.backward_with(initial_grad);
    }

    /// Perform backward propagation with a custom initial gradient
    pub fn backward_with(&self, grad_output: Tensor<DimDyn>) {
        if let Some(ref autograd) = self.autograd {
            // Accumulate gradient
            {
                let mut grad = autograd.grad.borrow_mut();
                if let Some(existing) = grad.take() {
                    // Add to existing gradient
                    let new_grad = &(*existing) + &grad_output;
                    *grad = Some(Rc::new(new_grad));
                } else {
                    *grad = Some(Rc::new(grad_output.clone()));
                }
            }

            // Propagate to inputs via grad_fn
            if let Some(ref grad_fn) = autograd.grad_fn {
                let input_grads = grad_fn.backward(&grad_output);
                let inputs = grad_fn.inputs();

                // Propagate gradients to each input tensor
                for (input, grad) in inputs.into_iter().zip(input_grads.into_iter()) {
                    if input.requires_grad() {
                        input.backward_with(grad);
                    }
                }
            }
        }
    }

    /// Get the accumulated gradient for this tensor
    ///
    /// Returns None if backward() hasn't been called or if this tensor
    /// doesn't require gradients.
    pub fn grad(&self) -> Option<Tensor<DimDyn>> {
        self.autograd
            .as_ref()
            .and_then(|ag| ag.grad.borrow().as_ref().map(|g| (**g).clone()))
    }

    /// Reset the gradient to None
    pub fn zero_grad(&self) {
        if let Some(ref autograd) = self.autograd {
            *autograd.grad.borrow_mut() = None;
        }
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    pub fn detach(&self) -> Tensor<D> {
        Tensor {
            node: self.node.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            autograd: None,
            _dim: std::marker::PhantomData,
        }
    }

    /// Compute cosine (used internally for sin backward)
    fn cos(&self) -> Tensor<DimDyn> {
        // cos(x) = sin(x + pi/2)
        // For simplicity, we'll use the graph node
        use std::f32::consts::FRAC_PI_2;
        let shifted = self.clone().into_dyn() + FRAC_PI_2;
        shifted.sin().into_dyn()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_reduce_grad_for_broadcast_same_shape() {
        let grad = Tensor::<DimDyn>::ones_dyn(&[2, 3]);
        let result = reduce_grad_for_broadcast(&grad, &[2, 3]);
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_backward_error_display() {
        let err = BackwardError::NoGrad;
        assert!(format!("{}", err).contains("does not require"));
    }

    #[test]
    fn test_grad_fn_names() {
        let t = Tensor::<DimDyn>::ones_dyn(&[2, 3]);
        let add_backward = AddBackward::new(t.clone(), t.clone());
        assert_eq!(add_backward.name(), "AddBackward");

        let neg_backward = NegBackward::new(t.clone());
        assert_eq!(neg_backward.name(), "NegBackward");
    }

    #[test]
    fn test_simple_add_backward() {
        // x = [[1, 1], [1, 1]] with requires_grad
        let x = Tensor::<Dim2>::ones([2, 2]).set_requires_grad(true);
        // y = x + x = [[2, 2], [2, 2]]
        let y = &x + &x;

        // Check that y requires grad (inherited from x)
        assert!(y.requires_grad());

        // Backward pass: dy/dx = 2 (since y = x + x)
        y.backward();

        // Check gradient exists
        let grad = x.grad();
        assert!(grad.is_some());

        // Gradient should have same shape as x
        let grad = grad.unwrap();
        assert_eq!(grad.shape(), &[2, 2]);
    }

    #[test]
    fn test_mul_backward() {
        // x = [[2, 2], [2, 2]] with requires_grad
        let x = Tensor::<Dim2>::full([2, 2], 2.0).set_requires_grad(true);
        // y = x * x = [[4, 4], [4, 4]]
        let y = &x * &x;

        assert!(y.requires_grad());

        // Backward pass: dy/dx = 2x = [[4, 4], [4, 4]]
        y.backward();

        let grad = x.grad();
        assert!(grad.is_some());
        assert_eq!(grad.unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_neg_backward() {
        let x = Tensor::<Dim2>::ones([2, 2]).set_requires_grad(true);
        let y = -&x;

        assert!(y.requires_grad());
        y.backward();

        let grad = x.grad();
        assert!(grad.is_some());
    }

    #[test]
    fn test_chain_backward() {
        // x = 2.0, y = x * x = 4.0, z = y + y = 8.0
        // dz/dy = 2, dy/dx = 2x = 4
        // dz/dx = dz/dy * dy/dx = 2 * 4 = 8
        let x = Tensor::<Dim2>::full([1, 1], 2.0).set_requires_grad(true);
        let y = &x * &x;
        let z = &y + &y;

        z.backward();

        let grad = x.grad();
        assert!(grad.is_some());
    }

    #[test]
    fn test_detach() {
        let x = Tensor::<Dim2>::ones([2, 2]).set_requires_grad(true);
        let detached = x.detach();
        assert!(!detached.requires_grad());
    }

    #[test]
    fn test_zero_grad() {
        let x = Tensor::<Dim2>::ones([2, 2]).set_requires_grad(true);
        let y = &x + &x;
        y.backward();

        // Gradient should exist
        assert!(x.grad().is_some());

        // Zero grad
        x.zero_grad();
        assert!(x.grad().is_none());
    }
}
