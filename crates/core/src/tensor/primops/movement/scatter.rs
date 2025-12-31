//! Scatter-Add operation
//!
//! Provides scatter_add for accumulating values at indexed positions.
//! This is the backward operation for gather.
//!
//! ## Scatter-Add semantics
//!
//! ```text
//! output[i][index[i][j][k]][k] += src[i][j][k]  (when dim=1)
//! ```
//!
//! The output shape matches the target tensor shape.

use std::sync::Arc;

use crate::tensor::ops::TensorOp;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{Dimension, Tensor, TensorDType, TensorInner};

impl<T: TensorDType, D: Dimension> Tensor<T, D> {
    /// Scatter-Add: Accumulate values at indexed positions
    ///
    /// This operation is the inverse of gather and is used for backward propagation.
    /// It atomically adds src values to target at positions specified by index.
    ///
    /// ```text
    /// output[...][index[...][j][...]][...] += src[...][j][...]  (for specified dim)
    /// ```
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to scatter
    /// * `index` - The index tensor specifying where to scatter values
    /// * `src` - The source tensor containing values to scatter
    ///
    /// # Returns
    /// A new tensor with scattered values accumulated
    ///
    /// # Panics
    /// - If dim >= ndim
    /// - If index and src have different shapes
    /// - If self, index, and src have different number of dimensions
    /// - If shapes don't match on non-dim axes (between self and index/src)
    pub fn scatter_add(
        &self,
        dim: usize,
        index: &Tensor<i64, D>,
        src: &Tensor<T, D>,
    ) -> Tensor<T, D> {
        let ndim = self.ndim();

        // Validation
        assert!(
            dim < ndim,
            "dim {} is out of bounds for tensor with {} dimensions",
            dim,
            ndim
        );
        assert_eq!(
            index.ndim(),
            ndim,
            "index tensor must have same number of dimensions as target ({} vs {})",
            index.ndim(),
            ndim
        );
        assert_eq!(
            src.ndim(),
            ndim,
            "src tensor must have same number of dimensions as target ({} vs {})",
            src.ndim(),
            ndim
        );

        // index and src must have the same shape
        assert_eq!(
            index.shape(),
            src.shape(),
            "index and src must have the same shape: {:?} vs {:?}",
            index.shape(),
            src.shape()
        );

        // Validate shapes match on non-dim axes between target and index/src
        for (axis, (&s_target, &s_src)) in self.shape().iter().zip(src.shape()).enumerate() {
            if axis != dim {
                assert_eq!(
                    s_target, s_src,
                    "target and src shapes must match on non-dim axes. \
                     Axis {}: target={}, src={}",
                    axis, s_target, s_src
                );
            }
        }

        let output_shape = self.shape().to_vec();

        // Create View for output (contiguous)
        let shape_exprs: Vec<Expr> = output_shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::contiguous(shape_exprs);

        // Create TensorOp::ScatterAdd
        let op = TensorOp::ScatterAdd {
            target: self.as_input_ref(),
            index: index.as_input_ref(),
            src: src.as_input_ref(),
            dim,
        };

        let inner = TensorInner::new(op, view, output_shape, T::DTYPE);

        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: std::marker::PhantomData,
            _dim: std::marker::PhantomData,
        }
    }
}

/// Build the output offset expression for scatter_add operation
///
/// Computes the memory offset into the output tensor using values from
/// the index tensor for the specified dimension.
///
/// # Arguments
/// * `dim` - The dimension along which to scatter
/// * `ndim` - Number of dimensions
/// * `target_shape` - Shape of the target tensor
/// * `src_shape` - Shape of the source tensor (same as index shape)
///
/// # Returns
/// An `Expr` for the output offset containing `LoadIndex` for dynamic index lookup
#[cfg(test)]
fn build_scatter_output_offset_expr(
    dim: usize,
    ndim: usize,
    target_shape: &[usize],
    src_shape: &[usize],
) -> Expr {
    assert!(dim < ndim, "dim {} must be less than ndim {}", dim, ndim);
    assert_eq!(
        target_shape.len(),
        ndim,
        "target_shape length must match ndim"
    );
    assert_eq!(src_shape.len(), ndim, "src_shape length must match ndim");

    // Build contiguous offset for reading from index buffer (same as src layout)
    let index_offset = build_contiguous_offset(ndim, src_shape);

    // LoadIndex reads from the index buffer (src_index = 1 for index tensor)
    let index_value = Expr::LoadIndex {
        src_index: 1,
        offset_expr: Box::new(index_offset),
    };

    // Compute target strides (row-major / C-contiguous)
    let target_strides = compute_strides(target_shape);

    // Build output offset expression
    // offset = sum(axis_index * stride[axis])
    // where axis_index = Idx(axis) if axis != dim, else LoadIndex value
    let mut offset = Expr::Const(0);

    for (axis, &stride_val) in target_strides.iter().enumerate().take(ndim) {
        let axis_idx: Expr = if axis == dim {
            index_value.clone()
        } else {
            Expr::Idx(axis)
        };

        let stride = Expr::from(stride_val as i64);
        offset += axis_idx * stride;
    }

    offset.simplify()
}

/// Compute strides for row-major (C-contiguous) layout
#[cfg(test)]
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Build contiguous offset expression from loop indices
///
/// offset = ridx0 * stride0 + ridx1 * stride1 + ...
#[cfg(test)]
fn build_contiguous_offset(ndim: usize, shape: &[usize]) -> Expr {
    if ndim == 0 {
        return Expr::Const(0);
    }

    let strides = compute_strides(shape);
    let mut offset = Expr::Const(0);

    for (i, &stride_val) in strides.iter().enumerate().take(ndim) {
        let idx = Expr::Idx(i);
        let stride = Expr::from(stride_val as i64);
        offset += idx * stride;
    }

    offset.simplify()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5, 10]), vec![10, 1]);
        assert_eq!(compute_strides(&[3]), vec![1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_build_contiguous_offset_1d() {
        let offset = build_contiguous_offset(1, &[4]);
        assert_eq!(offset, Expr::Idx(0));
    }

    #[test]
    fn test_build_scatter_output_offset_1d() {
        // 1D scatter: output[index[i]] += src[i]
        let expr = build_scatter_output_offset_expr(0, 1, &[4], &[3]);

        // Should contain LoadIndex
        assert!(
            expr.contains_load_index(),
            "Expression should contain LoadIndex"
        );
    }

    #[test]
    fn test_build_scatter_output_offset_2d_dim0() {
        // 2D scatter dim=0: output[index[i][j]][j] += src[i][j]
        let expr = build_scatter_output_offset_expr(0, 2, &[4, 5], &[3, 5]);

        assert!(
            expr.contains_load_index(),
            "Expression should contain LoadIndex"
        );
    }

    #[test]
    fn test_build_scatter_output_offset_2d_dim1() {
        // 2D scatter dim=1: output[i][index[i][j]] += src[i][j]
        let expr = build_scatter_output_offset_expr(1, 2, &[4, 5], &[4, 3]);

        assert!(
            expr.contains_load_index(),
            "Expression should contain LoadIndex"
        );
    }
}
