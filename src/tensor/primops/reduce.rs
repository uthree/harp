//! Reduction primitive operations
//!
//! - Reduce(Add): sum reduction
//! - Reduce(Mul): product reduction
//! - Reduce(Max): max reduction
//!
//! These operations support FloatDType (f32, f64).
//! Gradient tracking is only available for f32 tensors.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::ast::types::DType;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    DimDyn, Dimension, FloatDType, GradFn, ReduceOp, Tensor, TensorDType, TensorInner, TensorOp,
};

use super::binary::with_grad_fn;

/// Helper to attach gradient function for f32 tensors only
fn maybe_attach_grad<T: FloatDType, D: Dimension>(
    input: &Tensor<T, D>,
    result: Tensor<T, DimDyn>,
    grad_fn: impl FnOnce(Tensor<f32, DimDyn>) -> Arc<dyn GradFn>,
) -> Tensor<T, DimDyn> {
    // Only attach gradients for f32 tensors that require grad
    if T::DTYPE == DType::F32 && input.requires_grad() {
        // Type-erase to f32 for gradient operations
        let input_f32: Tensor<f32, DimDyn> = Tensor {
            inner: input.inner.clone(),
            _dtype: PhantomData,
            _dim: PhantomData,
        };
        let result_f32: Tensor<f32, DimDyn> = Tensor {
            inner: result.inner.clone(),
            _dtype: PhantomData,
            _dim: PhantomData,
        };
        let result_with_grad = with_grad_fn(result_f32, Some(grad_fn(input_f32)));
        // Cast back to T (which is f32 here)
        Tensor {
            inner: result_with_grad.inner,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    } else {
        result
    }
}

// ============================================================================
// Helper for type conversion
// ============================================================================

/// Convert any Tensor<T, D> to Tensor<f32, DimDyn> for graph operations.
fn to_graph_ref<T: TensorDType, D: Dimension>(tensor: &Tensor<T, D>) -> Tensor<f32, DimDyn> {
    Tensor {
        inner: tensor.inner.clone(),
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Reduce Gradients
// ============================================================================

/// Gradient for Reduce(Add): z = sum(a, axes)
/// ∂L/∂a = expand(∂L/∂z)
pub struct ReduceSumBackward {
    input: Tensor<f32, DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl ReduceSumBackward {
    pub fn new(input: Tensor<f32, DimDyn>, axes: Vec<usize>, keepdim: bool) -> Self {
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
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
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

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReduceSumBackward"
    }
}

/// Gradient for Reduce(Mul): z = prod(a, axes)
/// ∂L/∂a = ∂L/∂z · z / a
pub struct ReduceMulBackward {
    input: Tensor<f32, DimDyn>,
    output: Tensor<f32, DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl ReduceMulBackward {
    pub fn new(
        input: Tensor<f32, DimDyn>,
        output: Tensor<f32, DimDyn>,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Self {
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
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
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

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReduceMulBackward"
    }
}

/// Gradient for Reduce(Max): z = max(a, axes)
/// ∂L/∂a = ∂L/∂z · (a == max)
pub struct ReduceMaxBackward {
    input: Tensor<f32, DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl ReduceMaxBackward {
    pub fn new(
        input: Tensor<f32, DimDyn>,
        _output: Tensor<f32, DimDyn>, // TODO: Use for proper mask where input == max
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            input_shape,
            axes,
            keepdim,
        }
    }
}

impl GradFn for ReduceMaxBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
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

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReduceMaxBackward"
    }
}

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

/// Compute result shape after reduction
fn compute_reduce_shape(input_shape: &[usize], axes: &[usize], keepdim: bool) -> Vec<usize> {
    let mut result_shape = input_shape.to_vec();

    // Sort axes in descending order to handle dimension removal correctly
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_by(|a, b| b.cmp(a));

    for &axis in &sorted_axes {
        if keepdim {
            result_shape[axis] = 1;
        } else {
            result_shape.remove(axis);
        }
    }

    result_shape
}

/// Create a reduce Tensor using Compute variant
fn create_reduce<T: FloatDType, D: Dimension>(
    op: ReduceOp,
    input: &Tensor<T, D>,
    axes: &[usize],
    keepdim: bool,
) -> Tensor<T, DimDyn> {
    let result_shape = compute_reduce_shape(input.shape(), axes, keepdim);
    let view = view_from_shape(&result_shape);

    // Create Compute operation with reduce
    let input_ref = Arc::new(to_graph_ref(input));
    let inner = TensorInner::new(
        TensorOp::reduce(input_ref, op, axes.to_vec(), keepdim),
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

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    // ========================================================================
    // Type-safe single-axis reductions (recommended)
    // ========================================================================

    /// Sum along a single axis with type-safe dimension tracking
    ///
    /// Returns a tensor with one fewer dimension.
    /// For keepdim=true behavior, use `.unsqueeze(axis)` afterwards.
    pub fn sum_axis(&self, axis: usize) -> Tensor<T, D::Smaller> {
        let result_dyn = self.reduce_sum(&[axis], false);
        Tensor {
            inner: result_dyn.inner,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Product along a single axis with type-safe dimension tracking
    ///
    /// Returns a tensor with one fewer dimension.
    /// For keepdim=true behavior, use `.unsqueeze(axis)` afterwards.
    pub fn prod_axis(&self, axis: usize) -> Tensor<T, D::Smaller> {
        let result_dyn = self.reduce_mul(&[axis], false);
        Tensor {
            inner: result_dyn.inner,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Max along a single axis with type-safe dimension tracking
    ///
    /// Returns a tensor with one fewer dimension.
    /// For keepdim=true behavior, use `.unsqueeze(axis)` afterwards.
    pub fn max_axis(&self, axis: usize) -> Tensor<T, D::Smaller> {
        let result_dyn = self.reduce_max(&[axis], false);
        Tensor {
            inner: result_dyn.inner,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    // ========================================================================
    // Multi-axis reductions (returns DimDyn)
    // ========================================================================

    /// Sum reduction along specified axes
    ///
    /// For single-axis reduction, prefer `sum_axis()` for type safety.
    pub fn reduce_sum(&self, axes: &[usize], keepdim: bool) -> Tensor<T, DimDyn> {
        let result = create_reduce(ReduceOp::Sum, self, axes, keepdim);
        let axes = axes.to_vec();
        maybe_attach_grad(self, result, move |input| {
            Arc::new(ReduceSumBackward::new(input, axes, keepdim))
        })
    }

    /// Product reduction along specified axes
    ///
    /// For single-axis reduction, prefer `prod_axis()` for type safety.
    pub fn reduce_mul(&self, axes: &[usize], keepdim: bool) -> Tensor<T, DimDyn> {
        let result = create_reduce(ReduceOp::Prod, self, axes, keepdim);
        let axes = axes.to_vec();
        let result_for_grad: Tensor<f32, DimDyn> = Tensor {
            inner: result.inner.clone(),
            _dtype: PhantomData,
            _dim: PhantomData,
        };
        maybe_attach_grad(self, result, move |input| {
            Arc::new(ReduceMulBackward::new(
                input,
                result_for_grad,
                axes,
                keepdim,
            ))
        })
    }

    /// Max reduction along specified axes
    ///
    /// For single-axis reduction, prefer `max_axis()` for type safety.
    pub fn reduce_max(&self, axes: &[usize], keepdim: bool) -> Tensor<T, DimDyn> {
        let result = create_reduce(ReduceOp::Max, self, axes, keepdim);
        let axes = axes.to_vec();
        let result_for_grad: Tensor<f32, DimDyn> = Tensor {
            inner: result.inner.clone(),
            _dtype: PhantomData,
            _dim: PhantomData,
        };
        maybe_attach_grad(self, result, move |input| {
            Arc::new(ReduceMaxBackward::new(
                input,
                result_for_grad,
                axes,
                keepdim,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_reduce_sum() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let s = a.reduce_sum(&[1], false);
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_reduce_sum_keepdim() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let s = a.reduce_sum(&[1], true);
        assert_eq!(s.shape(), &[2, 1]);
    }

    #[test]
    fn test_reduce_mul() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let p = a.reduce_mul(&[1], false);
        assert_eq!(p.shape(), &[2]);
    }

    #[test]
    fn test_reduce_max() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let m = a.reduce_max(&[1], false);
        assert_eq!(m.shape(), &[2]);
    }

    // f64 tests
    #[test]
    fn test_reduce_sum_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let s = a.reduce_sum(&[1], false);
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_reduce_mul_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let p = a.reduce_mul(&[1], false);
        assert_eq!(p.shape(), &[2]);
    }

    #[test]
    fn test_reduce_max_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let m = a.reduce_max(&[1], false);
        assert_eq!(m.shape(), &[2]);
    }

    // Type-safe single-axis reduction tests
    #[test]
    fn test_sum_axis_type_safe() {
        use crate::tensor::Dim1;
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let s: Tensor<f32, Dim1> = a.sum_axis(1); // Dim2 -> Dim1
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_prod_axis_type_safe() {
        use crate::tensor::Dim1;
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let p: Tensor<f32, Dim1> = a.prod_axis(0); // Dim2 -> Dim1
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_max_axis_type_safe() {
        use crate::tensor::Dim1;
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let m: Tensor<f32, Dim1> = a.max_axis(1); // Dim2 -> Dim1
        assert_eq!(m.shape(), &[2]);
    }

    #[test]
    fn test_sum_axis_f64() {
        use crate::tensor::Dim1;
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let s: Tensor<f64, Dim1> = a.sum_axis(1);
        assert_eq!(s.shape(), &[2]);
    }
}
