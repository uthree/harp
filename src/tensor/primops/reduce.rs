//! Reduction primitive operations
//!
//! - Reduce(Add): sum reduction
//! - Reduce(Mul): product reduction
//! - Reduce(Max): max reduction

use std::marker::PhantomData;
use std::sync::Arc;

use crate::ast::DType;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{DimDyn, Dimension, ReduceOp, Tensor, TensorInner, TensorOp};

use super::binary::with_grad_fn;
use super::grad::{ReduceMaxBackward, ReduceMulBackward, ReduceSumBackward};

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
    input: &Tensor<D>,
    axes: &[usize],
    keepdim: bool,
) -> Tensor<DimDyn> {
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
        _dim: PhantomData,
    }
}

impl<D: Dimension> Tensor<D> {
    /// Sum reduction along specified axes (primop)
    ///
    /// # Arguments
    /// * `axes` - Axes to reduce over
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    pub fn reduce_sum(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
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
    pub fn reduce_mul(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
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
    pub fn reduce_max(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
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
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.reduce_sum(&[1], false);
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_reduce_sum_keepdim() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.reduce_sum(&[1], true);
        assert_eq!(s.shape(), &[2, 1]);
    }

    #[test]
    fn test_reduce_mul() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let p = a.reduce_mul(&[1], false);
        assert_eq!(p.shape(), &[2]);
    }

    #[test]
    fn test_reduce_max() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let m = a.reduce_max(&[1], false);
        assert_eq!(m.shape(), &[2]);
    }
}
