//! Gradient functions for primitive operations
//!
//! Each primop has a corresponding backward function for automatic differentiation.

use crate::tensor::{DimDyn, GradFn, Tensor};

// ============================================================================
// Helper Functions
// ============================================================================

/// Reduce gradient to match the original input shape (handle broadcasting)
pub(crate) fn reduce_grad_for_broadcast(grad: &Tensor<DimDyn>, target_shape: &[usize]) -> Tensor<DimDyn> {
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
        result = result.reduce_sum(&reduce_axes, true);
    }

    // Reshape to target shape
    result.reshape_dyn(target_shape)
}

// ============================================================================
// Binary Gradients
// ============================================================================

/// Gradient for Add: z = a + b
/// ∂L/∂a = ∂L/∂z, ∂L/∂b = ∂L/∂z
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

/// Gradient for Mul: z = a * b
/// ∂L/∂a = ∂L/∂z · b, ∂L/∂b = ∂L/∂z · a
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
        let grad_lhs_full = grad_output * &self.rhs;
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs_full, self.lhs.shape());

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

/// Gradient for Max: z = max(a, b)
/// ∂L/∂a = ∂L/∂z · (a ≥ b), ∂L/∂b = ∂L/∂z · (b > a)
pub struct MaxBackward {
    lhs: Tensor<DimDyn>,
    rhs: Tensor<DimDyn>,
}

impl MaxBackward {
    pub fn new(lhs: Tensor<DimDyn>, rhs: Tensor<DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for MaxBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // Approximation: gradient flows to the larger input
        // TODO: Proper comparison operation needed
        let grad_lhs = reduce_grad_for_broadcast(grad_output, self.lhs.shape());
        let grad_rhs = Tensor::<DimDyn>::zeros_dyn(self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

// ============================================================================
// Unary Gradients
// ============================================================================

/// Gradient for Neg: z = -a
/// ∂L/∂a = -∂L/∂z
pub struct NegBackward {
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

/// Gradient for Recip: z = 1/a
/// ∂L/∂a = -∂L/∂z / a² = -∂L/∂z · z²
pub struct RecipBackward {
    input: Tensor<DimDyn>,
    output: Tensor<DimDyn>,
}

impl RecipBackward {
    pub fn new(input: Tensor<DimDyn>, output: Tensor<DimDyn>) -> Self {
        Self { input, output }
    }
}

impl GradFn for RecipBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // ∂L/∂a = -∂L/∂z · z² where z = 1/a
        let z_squared = &self.output * &self.output;
        vec![-(grad_output * &z_squared)]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "RecipBackward"
    }
}

/// Gradient for Sqrt: z = √a
/// ∂L/∂a = ∂L/∂z / (2·√a) = ∂L/∂z / (2·z)
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
        vec![grad_output / &two_sqrt]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Gradient for Log2: z = log₂(a)
/// ∂L/∂a = ∂L/∂z / (a · ln(2))
pub struct Log2Backward {
    input: Tensor<DimDyn>,
}

impl Log2Backward {
    pub fn new(input: Tensor<DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn for Log2Backward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        let ln2 = std::f32::consts::LN_2;
        let denominator = &self.input * ln2;
        vec![grad_output / &denominator]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Log2Backward"
    }
}

/// Gradient for Exp2: z = 2^a
/// ∂L/∂a = ∂L/∂z · 2^a · ln(2) = ∂L/∂z · z · ln(2)
pub struct Exp2Backward {
    input: Tensor<DimDyn>,
    output: Tensor<DimDyn>,
}

impl Exp2Backward {
    pub fn new(input: Tensor<DimDyn>, output: Tensor<DimDyn>) -> Self {
        Self { input, output }
    }
}

impl GradFn for Exp2Backward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        let ln2 = std::f32::consts::LN_2;
        let scaled_output = &self.output * ln2;
        vec![grad_output * &scaled_output]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Exp2Backward"
    }
}

/// Gradient for Sin: z = sin(a)
/// ∂L/∂a = ∂L/∂z · cos(a)
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
        // cos(x) = sin(x + π/2)
        use std::f32::consts::FRAC_PI_2;
        let shifted = &self.input + FRAC_PI_2;
        let cos_input = shifted.sin();
        vec![grad_output * &cos_input]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

// ============================================================================
// Reduce Gradients
// ============================================================================

/// Gradient for Reduce(Add): z = sum(a, axes)
/// ∂L/∂a = expand(∂L/∂z)
pub struct ReduceSumBackward {
    input: Tensor<DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl ReduceSumBackward {
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

impl GradFn for ReduceSumBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
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
        "ReduceSumBackward"
    }
}

/// Gradient for Reduce(Mul): z = prod(a, axes)
/// ∂L/∂a = ∂L/∂z · z / a
pub struct ReduceMulBackward {
    input: Tensor<DimDyn>,
    output: Tensor<DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl ReduceMulBackward {
    pub fn new(input: Tensor<DimDyn>, output: Tensor<DimDyn>, axes: Vec<usize>, keepdim: bool) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            output,
            input_shape,
            axes,
            keepdim,
        }
    }
}

impl GradFn for ReduceMulBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        let mut grad = grad_output.clone();

        if !self.keepdim {
            for &axis in &self.axes {
                grad = grad.unsqueeze(axis);
            }
        }

        // ∂L/∂a = ∂L/∂z · z / a (expanded)
        let mut output_expanded = self.output.clone();
        if !self.keepdim {
            for &axis in &self.axes {
                output_expanded = output_expanded.unsqueeze(axis);
            }
        }
        let output_expanded = output_expanded.expand(&self.input_shape);
        let grad_expanded = grad.expand(&self.input_shape);

        vec![(&grad_expanded * &output_expanded) / &self.input]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReduceMulBackward"
    }
}

/// Gradient for Reduce(Max): z = max(a, axes)
/// ∂L/∂a = ∂L/∂z · (a == max)
pub struct ReduceMaxBackward {
    input: Tensor<DimDyn>,
    #[allow(dead_code)]
    output: Tensor<DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl ReduceMaxBackward {
    pub fn new(input: Tensor<DimDyn>, output: Tensor<DimDyn>, axes: Vec<usize>, keepdim: bool) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            output,
            input_shape,
            axes,
            keepdim,
        }
    }
}

impl GradFn for ReduceMaxBackward {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>> {
        // Expand gradient to original shape
        let mut grad = grad_output.clone();
        if !self.keepdim {
            for &axis in &self.axes {
                grad = grad.unsqueeze(axis);
            }
        }

        // TODO: Proper mask where input == max
        // For now, just expand the gradient
        let grad_expanded = grad.expand(&self.input_shape);
        vec![grad_expanded]
    }

    fn inputs(&self) -> Vec<Tensor<DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReduceMaxBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward_name() {
        let t = Tensor::<DimDyn>::ones_dyn(&[2, 3]);
        let backward = AddBackward::new(t.clone(), t.clone());
        assert_eq!(backward.name(), "AddBackward");
    }

    #[test]
    fn test_reduce_grad_for_broadcast_same_shape() {
        let grad = Tensor::<DimDyn>::ones_dyn(&[2, 3]);
        let result = reduce_grad_for_broadcast(&grad, &[2, 3]);
        assert_eq!(result.shape(), &[2, 3]);
    }
}
