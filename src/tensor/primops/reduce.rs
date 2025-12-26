//! Reduction primitive operations
//!
//! - Reduce(Add): sum reduction
//! - Reduce(Mul): product reduction
//! - Reduce(Max): max reduction

use std::marker::PhantomData;
use std::sync::Arc;

use crate::ast::DType;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{DimDyn, Dimension, GradFn, ReduceOp, Tensor, TensorInner, TensorOp};

use super::binary::with_grad_fn;

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
fn create_reduce<D: Dimension>(
    op: ReduceOp,
    input: &Tensor<f32, D>,
    axes: &[usize],
    keepdim: bool,
) -> Tensor<f32, DimDyn> {
    let result_shape = compute_reduce_shape(input.shape(), axes, keepdim);
    let view = view_from_shape(&result_shape);

    // Create Compute operation with reduce
    let input_ref = Arc::new(input.clone().into_dyn());
    let inner = TensorInner::new(
        TensorOp::reduce(input_ref, op, axes.to_vec(), keepdim),
        view,
        result_shape,
        DType::F32,
    );

    Tensor {
        inner: Arc::new(inner),
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

impl<D: Dimension> Tensor<f32, D> {
    /// Sum reduction along specified axes (primop)
    ///
    /// # Arguments
    /// * `axes` - Axes to reduce over
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    pub fn reduce_sum(&self, axes: &[usize], keepdim: bool) -> Tensor<f32, DimDyn> {
        let result = create_reduce(ReduceOp::Sum, self, axes, keepdim);

        if self.requires_grad() {
            let grad_fn = ReduceSumBackward::new(self.clone().into_dyn(), axes.to_vec(), keepdim);
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Product reduction along specified axes (primop)
    ///
    /// # Arguments
    /// * `axes` - Axes to reduce over
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    pub fn reduce_mul(&self, axes: &[usize], keepdim: bool) -> Tensor<f32, DimDyn> {
        let result = create_reduce(ReduceOp::Prod, self, axes, keepdim);

        if self.requires_grad() {
            let grad_fn = ReduceMulBackward::new(
                self.clone().into_dyn(),
                result.clone(),
                axes.to_vec(),
                keepdim,
            );
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Max reduction along specified axes (primop)
    ///
    /// # Arguments
    /// * `axes` - Axes to reduce over
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    pub fn reduce_max(&self, axes: &[usize], keepdim: bool) -> Tensor<f32, DimDyn> {
        let result = create_reduce(ReduceOp::Max, self, axes, keepdim);

        if self.requires_grad() {
            let grad_fn = ReduceMaxBackward::new(
                self.clone().into_dyn(),
                result.clone(),
                axes.to_vec(),
                keepdim,
            );
            with_grad_fn(result, Some(Arc::new(grad_fn)))
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
}
