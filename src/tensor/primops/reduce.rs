//! Reduction primitive operations
//!
//! - Reduce(Add): sum reduction
//! - Reduce(Mul): product reduction
//! - Reduce(Max): max reduction
//!
//! These operations support FloatDType (f32, f64).
//! Gradient tracking is currently f32-only (infrastructure for generic in place).

use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    DimDyn, Dimension, FloatDType, GradFn, ReduceOp, Tensor, TensorDType, TensorInner, TensorOp,
};

use super::binary::with_grad_fn_generic;
use super::unary::Recip;
use crate::tensor::FloatDTypeAutograd;

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
// Reduce Gradients (generic over FloatDType)
// ============================================================================

/// Gradient for Reduce(Add): z = sum(a, axes)
/// ∂L/∂a = expand(∂L/∂z)
pub struct ReduceSumBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl<T: FloatDType> ReduceSumBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, axes: Vec<usize>, keepdim: bool) -> Self {
        let input_shape = input.shape().to_vec();
        Self {
            input,
            input_shape,
            axes,
            keepdim,
        }
    }
}

// ReduceSumBackward doesn't use arithmetic ops, so we can implement generically
impl<T: FloatDType> GradFn<T> for ReduceSumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
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

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReduceSumBackward"
    }
}

/// Gradient for Reduce(Mul): z = prod(a, axes)
/// ∂L/∂a = ∂L/∂z · z / a
pub struct ReduceMulBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output: Tensor<T, DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl<T: FloatDType> ReduceMulBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        output: Tensor<T, DimDyn>,
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

// ReduceMulBackward uses * and /, need separate impls for f32/f64
impl GradFn<f32> for ReduceMulBackward<f32> {
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

impl GradFn<f64> for ReduceMulBackward<f64> {
    fn backward(&self, grad_output: &Tensor<f64, DimDyn>) -> Vec<Tensor<f64, DimDyn>> {
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

        // Use recip() * instead of / since Div not yet implemented for f64
        vec![&(&grad_expanded * &output_expanded) * &self.input.clone().recip()]
    }

    fn inputs(&self) -> Vec<Tensor<f64, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReduceMulBackward"
    }
}

/// Gradient for Reduce(Max): z = max(a, axes)
/// ∂L/∂a = ∂L/∂z · (a == max)
pub struct ReduceMaxBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdim: bool,
}

impl<T: FloatDType> ReduceMaxBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        _output: Tensor<T, DimDyn>, // TODO: Use for proper mask where input == max
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

// ReduceMaxBackward doesn't use arithmetic ops, so we can implement generically
impl<T: FloatDType> GradFn<T> for ReduceMaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
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

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
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

// ============================================================================
// Type-safe single-axis reductions (no gradient tracking needed here)
// These just delegate to multi-axis reductions
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Sum along a single axis with type-safe dimension tracking
    ///
    /// Returns a tensor with one fewer dimension.
    /// For keepdim=true behavior, use `.unsqueeze(axis)` afterwards.
    pub fn sum_axis(&self, axis: usize) -> Tensor<T, D::Smaller>
    where
        T: FloatDTypeAutograd,
    {
        let result_dyn = self.reduce_sum(&[axis], false);
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
    pub fn max_axis(&self, axis: usize) -> Tensor<T, D::Smaller>
    where
        T: FloatDTypeAutograd,
    {
        let result_dyn = self.reduce_max(&[axis], false);
        Tensor {
            inner: result_dyn.inner,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Multi-axis reductions with gradient tracking (FloatDTypeAutograd)
// ============================================================================

impl<T: FloatDTypeAutograd, D: Dimension> Tensor<T, D> {
    /// Sum reduction along specified axes
    ///
    /// For single-axis reduction, prefer `sum_axis()` for type safety.
    pub fn reduce_sum(&self, axes: &[usize], keepdim: bool) -> Tensor<T, DimDyn> {
        let result = create_reduce(ReduceOp::Sum, self, axes, keepdim);
        if self.requires_grad() {
            let axes = axes.to_vec();
            let input = self.clone().into_dyn();
            let grad_fn = ReduceSumBackward::new(input, axes, keepdim);
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Max reduction along specified axes
    ///
    /// For single-axis reduction, prefer `max_axis()` for type safety.
    pub fn reduce_max(&self, axes: &[usize], keepdim: bool) -> Tensor<T, DimDyn> {
        let result = create_reduce(ReduceOp::Max, self, axes, keepdim);
        if self.requires_grad() {
            let axes = axes.to_vec();
            let input = self.clone().into_dyn();
            let result_for_grad = result.clone();
            let grad_fn = ReduceMaxBackward::new(input, result_for_grad, axes, keepdim);
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

// ============================================================================
// reduce_mul needs separate impls because ReduceMulBackward uses arithmetic ops
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Product along a single axis with type-safe dimension tracking (f32)
    ///
    /// Returns a tensor with one fewer dimension.
    /// For keepdim=true behavior, use `.unsqueeze(axis)` afterwards.
    pub fn prod_axis(&self, axis: usize) -> Tensor<f32, D::Smaller> {
        let result_dyn = self.reduce_mul(&[axis], false);
        Tensor {
            inner: result_dyn.inner,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Product reduction along specified axes (f32)
    ///
    /// For single-axis reduction, prefer `prod_axis()` for type safety.
    pub fn reduce_mul(&self, axes: &[usize], keepdim: bool) -> Tensor<f32, DimDyn> {
        let result = create_reduce(ReduceOp::Prod, self, axes, keepdim);
        if self.requires_grad() {
            let axes = axes.to_vec();
            let input = self.clone().into_dyn();
            let result_for_grad = result.clone();
            let grad_fn = ReduceMulBackward::new(input, result_for_grad, axes, keepdim);
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Tensor<f64, D> {
    /// Product along a single axis with type-safe dimension tracking (f64)
    ///
    /// Returns a tensor with one fewer dimension.
    /// For keepdim=true behavior, use `.unsqueeze(axis)` afterwards.
    pub fn prod_axis(&self, axis: usize) -> Tensor<f64, D::Smaller> {
        let result_dyn = self.reduce_mul(&[axis], false);
        Tensor {
            inner: result_dyn.inner,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Product reduction along specified axes (f64)
    ///
    /// For single-axis reduction, prefer `prod_axis()` for type safety.
    pub fn reduce_mul(&self, axes: &[usize], keepdim: bool) -> Tensor<f64, DimDyn> {
        let result = create_reduce(ReduceOp::Prod, self, axes, keepdim);
        if self.requires_grad() {
            let axes = axes.to_vec();
            let input = self.clone().into_dyn();
            let result_for_grad = result.clone();
            let grad_fn = ReduceMulBackward::new(input, result_for_grad, axes, keepdim);
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
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
