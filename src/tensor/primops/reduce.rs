//! Type-safe single-axis reduction primitive operations
//!
//! - sum_axis: sum reduction along a single axis
//! - prod_axis: product reduction along a single axis
//! - max_axis: max reduction along a single axis
//!
//! All operations are type-safe: reducing a Tensor<T, Dim<N>> produces Tensor<T, Dim<N-1>>.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    DimDyn, Dimension, FloatDType, GradFn, ReduceOp, Tensor, TensorInner, TensorOp,
};

use super::binary::with_grad_fn_generic;
use super::unary::Recip;
// FloatDType import is already in the main tensor use statement

// ============================================================================
// Reduce Gradients (single-axis only)
// ============================================================================

/// Gradient for sum_axis: z = sum(a, axis)
/// ∂L/∂a = expand(unsqueeze(∂L/∂z, axis))
pub(crate) struct SumBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    input_shape: Vec<usize>,
    axis: usize,
}

impl<T: FloatDType> SumBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, axis: usize) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            input_shape,
            axis,
        }
    }
}

impl<T: FloatDType> GradFn<T> for SumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Unsqueeze the reduced dimension back
        let grad = grad_output.unsqueeze(self.axis);
        // Expand to original shape
        let grad_expanded = grad.expand(&self.input_shape);
        vec![grad_expanded]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SumAxisBackward"
    }
}

/// Gradient for prod_axis: z = prod(a, axis)
/// ∂L/∂a = ∂L/∂z · z / a
pub(crate) struct ProdBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output: Tensor<T, DimDyn>,
    input_shape: Vec<usize>,
    axis: usize,
}

impl<T: FloatDType> ProdBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, output: Tensor<T, DimDyn>, axis: usize) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            output,
            input_shape,
            axis,
        }
    }
}

impl<T: FloatDType> GradFn<T> for ProdBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Unsqueeze output and grad back to input shape
        let output_expanded = self.output.unsqueeze(self.axis).expand(&self.input_shape);
        let grad_expanded = grad_output.unsqueeze(self.axis).expand(&self.input_shape);
        // ∂L/∂a = ∂L/∂z · z / a (using recip() * pattern)
        vec![&(&grad_expanded * &output_expanded) * &self.input.clone().recip()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ProdAxisBackward"
    }
}

/// Gradient for max_axis: z = max(a, axis)
/// ∂L/∂a = ∂L/∂z · (a == max) - simplified version: just expand
pub(crate) struct MaxBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    input_shape: Vec<usize>,
    axis: usize,
}

impl<T: FloatDType> MaxBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, _output: Tensor<T, DimDyn>, axis: usize) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            input_shape,
            axis,
        }
    }
}

impl<T: FloatDType> GradFn<T> for MaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // TODO: Proper mask where input == max
        // For now, just expand the gradient
        let grad = grad_output.unsqueeze(self.axis);
        let grad_expanded = grad.expand(&self.input_shape);
        vec![grad_expanded]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "MaxAxisBackward"
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

/// Compute result shape after single-axis reduction (always keepdim=false)
fn compute_single_axis_reduce_shape(input_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut result_shape = input_shape.to_vec();
    result_shape.remove(axis);
    result_shape
}

/// Create a single-axis reduce Tensor
fn create_single_axis_reduce<T: FloatDType, D: Dimension>(
    op: ReduceOp,
    input: &Tensor<T, D>,
    axis: usize,
) -> Tensor<T, D::Smaller> {
    let result_shape = compute_single_axis_reduce_shape(input.shape(), axis);
    let view = view_from_shape(&result_shape);

    // Create Reduce operation
    let input_ref = input.as_input_ref();
    let inner = TensorInner::new(
        TensorOp::reduce(input_ref, op, vec![axis], false),
        view,
        result_shape,
        T::DTYPE,
    );

    Tensor {
        inner: Arc::new(inner),
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Type-safe single-axis reductions
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Sum along a single axis with type-safe dimension tracking
    ///
    /// Returns a tensor with one fewer dimension (Dim<N> -> Dim<N-1>).
    ///
    /// # Example
    /// ```ignore
    /// let a: Tensor<f32, Dim2> = Tensor::ones([2, 3]);
    /// let s: Tensor<f32, Dim1> = a.sum(1);
    /// assert_eq!(s.shape(), &[2]);
    /// ```
    pub fn sum(&self, axis: usize) -> Tensor<T, D::Smaller> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            self.ndim()
        );

        let result = create_single_axis_reduce(ReduceOp::Sum, self, axis);

        if self.requires_grad() {
            let input = self.clone().into_dyn();
            let grad_fn = SumBackward::new(input, axis);
            // Convert result to DimDyn, add grad_fn, then convert back
            let result_dyn: Tensor<T, DimDyn> = Tensor {
                inner: result.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let result_with_grad = with_grad_fn_generic(result_dyn, Some(Arc::new(grad_fn)));
            Tensor {
                inner: result_with_grad.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            }
        } else {
            result
        }
    }

    /// Max along a single axis with type-safe dimension tracking
    ///
    /// Returns a tensor with one fewer dimension (Dim<N> -> Dim<N-1>).
    ///
    /// # Example
    /// ```ignore
    /// let a: Tensor<f32, Dim2> = Tensor::ones([2, 3]);
    /// let m: Tensor<f32, Dim1> = a.max(1);
    /// assert_eq!(m.shape(), &[2]);
    /// ```
    pub fn max(&self, axis: usize) -> Tensor<T, D::Smaller> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            self.ndim()
        );

        let result = create_single_axis_reduce(ReduceOp::Max, self, axis);

        if self.requires_grad() {
            let input = self.clone().into_dyn();
            let result_dyn: Tensor<T, DimDyn> = Tensor {
                inner: result.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_fn = MaxBackward::new(input, result_dyn.clone(), axis);
            let result_with_grad = with_grad_fn_generic(result_dyn, Some(Arc::new(grad_fn)));
            Tensor {
                inner: result_with_grad.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            }
        } else {
            result
        }
    }
}

// ============================================================================
// prod_axis - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Product along a single axis with type-safe dimension tracking
    ///
    /// Returns a tensor with one fewer dimension (Dim<N> -> Dim<N-1>).
    ///
    /// # Example
    /// ```ignore
    /// let a: Tensor<f32, Dim2> = Tensor::ones([2, 3]);
    /// let p: Tensor<f32, Dim1> = a.prod(0);
    /// assert_eq!(p.shape(), &[3]);
    /// ```
    pub fn prod(&self, axis: usize) -> Tensor<T, D::Smaller> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            self.ndim()
        );

        let result = create_single_axis_reduce(ReduceOp::Prod, self, axis);

        if self.requires_grad() {
            let input = self.clone().into_dyn();
            let result_dyn: Tensor<T, DimDyn> = Tensor {
                inner: result.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_fn = ProdBackward::new(input, result_dyn.clone(), axis);
            let result_with_grad = with_grad_fn_generic(result_dyn, Some(Arc::new(grad_fn)));
            Tensor {
                inner: result_with_grad.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            }
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim1, Dim2};

    #[test]
    fn test_sum_axis_type_safe() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let s: Tensor<f32, Dim1> = a.sum(1); // Dim2 -> Dim1
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_sum_axis_dim0() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let s: Tensor<f32, Dim1> = a.sum(0); // Dim2 -> Dim1
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_prod_axis_type_safe() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let p: Tensor<f32, Dim1> = a.prod(0); // Dim2 -> Dim1
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_max_axis_type_safe() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let m: Tensor<f32, Dim1> = a.max(1); // Dim2 -> Dim1
        assert_eq!(m.shape(), &[2]);
    }

    #[test]
    fn test_sum_axis_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let s: Tensor<f64, Dim1> = a.sum(1);
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_prod_axis_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let p: Tensor<f64, Dim1> = a.prod(0);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_max_axis_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let m: Tensor<f64, Dim1> = a.max(1);
        assert_eq!(m.shape(), &[2]);
    }

    #[test]
    #[should_panic(expected = "Axis 2 out of bounds")]
    fn test_sum_axis_out_of_bounds() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let _ = a.sum(2); // Should panic
    }
}
