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
    Dim0, DimDyn, Dimension, FloatDType, GradFn, ReduceOp, Tensor, TensorInner, TensorOp,
};

use super::unary::Recip;

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
        autograd_meta: None,
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
            let grad_fn = SumBackward::<T, D>::new(self.clone(), axis);
            result.with_grad_fn(Arc::new(grad_fn))
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
            let grad_fn = MaxReduceBackward::<T, D>::new(self.clone(), axis);
            result.with_grad_fn(Arc::new(grad_fn))
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
            let grad_fn = ProdBackward::<T, D>::new(self.clone(), result.clone(), axis);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}

// ============================================================================
// Backward Structs
// ============================================================================

// ============================================================================
// All-axes reductions (returns Dim0)
// ============================================================================

/// Create a reduce over all axes, returning Dim0
fn create_all_axes_reduce<T: FloatDType, D: Dimension>(
    op: ReduceOp,
    input: &Tensor<T, D>,
) -> Tensor<T, Dim0> {
    let ndim = input.ndim();
    let axes: Vec<usize> = (0..ndim).collect();
    let view = View::contiguous(Vec::<Expr>::new()); // Scalar has empty shape

    let input_ref = input.as_input_ref();
    let inner = TensorInner::new(
        TensorOp::reduce(input_ref, op, axes, false),
        view,
        vec![], // Scalar shape
        T::DTYPE,
    );

    Tensor {
        inner: Arc::new(inner),
        autograd_meta: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Sum over all axes, returning a scalar (Dim0)
    ///
    /// This is a type-safe all-axes reduction: any dimension D becomes Dim0.
    ///
    /// # Example
    /// ```ignore
    /// let a: Tensor<f32, Dim2> = Tensor::ones([2, 3]);
    /// let s: Tensor<f32, Dim0> = a.sum_all();
    /// assert_eq!(s.shape(), &[]);
    /// ```
    pub fn sum_all(&self) -> Tensor<T, Dim0> {
        let result = create_all_axes_reduce(ReduceOp::Sum, self);

        if self.requires_grad() {
            let grad_fn = SumAllBackward::<T, D>::new(self.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Mean over all axes, returning a scalar (Dim0)
    ///
    /// Equivalent to `sum_all() / n_elements`.
    ///
    /// # Example
    /// ```ignore
    /// let a: Tensor<f32, Dim2> = Tensor::ones([2, 3]);
    /// let m: Tensor<f32, Dim0> = a.mean();
    /// assert_eq!(m.shape(), &[]);
    /// ```
    pub fn mean(&self) -> Tensor<T, Dim0> {
        let n_elements = self.shape().iter().product::<usize>();
        let sum_result = self.sum_all();

        // Divide by number of elements
        let n_tensor = Tensor::<T, Dim0>::full([], T::from_usize(n_elements));
        &sum_result / &n_tensor
    }
}

// ============================================================================
// Backward for all-axes sum
// ============================================================================

/// Gradient for sum_all: z = sum_all(a)
/// ∂L/∂a = expand(∂L/∂z) (broadcast scalar gradient to input shape)
pub struct SumAllBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    input_shape: Vec<usize>,
}

impl<T: FloatDType, D: Dimension> SumAllBackward<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        let input_shape = input.shape().to_vec();
        Self { input, input_shape }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, Dim0> for SumAllBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, Dim0>) {
        if self.input.requires_grad() {
            // Expand scalar gradient to input shape
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_expanded = grad_dyn.expand(&self.input_shape);
            // Convert back to D
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_expanded.inner,
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "SumAllBackward"
    }
}

// ============================================================================
// Backward Structs (single-axis)
// ============================================================================

/// Gradient for sum_axis: z = sum(a, axis)
/// ∂L/∂a = expand(unsqueeze(∂L/∂z, axis))
///
/// Input has dimension D, output has dimension D::Smaller
pub struct SumBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    input_shape: Vec<usize>,
    axis: usize,
}

impl<T: FloatDType, D: Dimension> SumBackward<T, D> {
    pub fn new(input: Tensor<T, D>, axis: usize) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            input_shape,
            axis,
        }
    }
}

// Output is D::Smaller, so implements GradFn for that dimension
impl<T: FloatDType, D: Dimension> GradFn<T, D::Smaller> for SumBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Smaller>) {
        if self.input.requires_grad() {
            // Convert to DimDyn for operations, then convert back to D
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            // Unsqueeze and expand
            let grad_unsqueezed = grad_dyn.unsqueeze(self.axis);
            let grad_expanded = grad_unsqueezed.expand(&self.input_shape);
            // Convert back to D
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_expanded.inner,
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Gradient for prod_axis: z = prod(a, axis)
/// ∂L/∂a = ∂L/∂z · z / a
pub struct ProdBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    output: Tensor<T, D::Smaller>,
    input_shape: Vec<usize>,
    axis: usize,
}

impl<T: FloatDType, D: Dimension> ProdBackward<T, D> {
    pub fn new(input: Tensor<T, D>, output: Tensor<T, D::Smaller>, axis: usize) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            output,
            input_shape,
            axis,
        }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D::Smaller> for ProdBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Smaller>) {
        if self.input.requires_grad() {
            // Convert to DimDyn for operations
            let output_dyn: Tensor<T, DimDyn> = Tensor {
                inner: self.output.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let input_dyn: Tensor<T, DimDyn> = Tensor {
                inner: self.input.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };

            // Unsqueeze and expand
            let output_expanded = output_dyn.unsqueeze(self.axis).expand(&self.input_shape);
            let grad_expanded = grad_dyn.unsqueeze(self.axis).expand(&self.input_shape);
            // ∂L/∂a = ∂L/∂z · z / a
            let grad_input_dyn = &(&grad_expanded * &output_expanded) * &input_dyn.recip();
            // Convert back to D
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_input_dyn.inner,
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ProdBackward"
    }
}

/// Gradient for max_axis: z = max(a, axis)
/// ∂L/∂a = mask(a == max) · expand(unsqueeze(∂L/∂z, axis))
pub struct MaxReduceBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    input_shape: Vec<usize>,
    axis: usize,
}

impl<T: FloatDType, D: Dimension> MaxReduceBackward<T, D> {
    pub fn new(input: Tensor<T, D>, axis: usize) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            input_shape,
            axis,
        }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D::Smaller> for MaxReduceBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Smaller>) {
        if self.input.requires_grad() {
            // Convert to DimDyn for operations
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            // Approximation: expand gradient uniformly
            let grad_expanded = grad_dyn.unsqueeze(self.axis).expand(&self.input_shape);
            // Convert back to D
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_expanded.inner,
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "MaxReduceBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim0, Dim1, Dim2, Dim3};

    // ========================================================================
    // sum_all / mean tests
    // ========================================================================

    #[test]
    fn test_sum_all_dim2() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let s: Tensor<f32, Dim0> = a.sum_all();
        assert_eq!(s.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_sum_all_dim3() {
        let a = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        let s: Tensor<f32, Dim0> = a.sum_all();
        assert_eq!(s.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_mean_dim2() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let m: Tensor<f32, Dim0> = a.mean();
        assert_eq!(m.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_mean_dim3() {
        let a = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        let m: Tensor<f32, Dim0> = a.mean();
        assert_eq!(m.shape(), &[] as &[usize]);
    }

    // ========================================================================
    // Single-axis reduction tests
    // ========================================================================

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
