//! Gather operation helpers
//!
//! Provides index expression builders for gather operations.
//!
//! ## Gather semantics
//!
//! ```text
//! output[i][j][k] = input[i][index[i][j][k]][k]  (when dim=1)
//! ```
//!
//! The output shape matches the index tensor shape.

use crate::tensor::shape::Expr;

/// Build the index expression for gather operation
///
/// Creates an `Expr` that computes the memory offset into the input tensor
/// using values from the index tensor for the specified dimension.
///
/// # Arguments
/// * `dim` - The dimension along which to gather
/// * `ndim` - Number of dimensions
/// * `input_shape` - Shape of the input tensor
/// * `index_shape` - Shape of the index tensor
///
/// # Returns
/// An `Expr` containing `LoadIndex` for dynamic index lookup
pub fn build_gather_index_expr(
    dim: usize,
    ndim: usize,
    input_shape: &[usize],
    index_shape: &[usize],
) -> Expr {
    assert!(dim < ndim, "dim {} must be less than ndim {}", dim, ndim);
    assert_eq!(
        input_shape.len(),
        ndim,
        "input_shape length must match ndim"
    );
    assert_eq!(
        index_shape.len(),
        ndim,
        "index_shape length must match ndim"
    );

    // Build contiguous offset for reading from index buffer
    // offset = ridx0 * stride0 + ridx1 * stride1 + ...
    let index_offset = build_contiguous_offset(ndim, index_shape);

    // LoadIndex reads from the index buffer (src_index = 1)
    let index_value = Expr::LoadIndex {
        src_index: 1,
        offset_expr: Box::new(index_offset),
    };

    // Compute input strides (row-major / C-contiguous)
    let input_strides = compute_strides(input_shape);

    // Build input offset expression
    // offset = sum(axis_index * stride[axis])
    // where axis_index = Idx(axis) if axis != dim, else LoadIndex value
    let mut offset = Expr::Const(0);

    for (axis, &stride_val) in input_strides.iter().enumerate().take(ndim) {
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
        // offset = ridx0 * 1 = ridx0
        assert_eq!(offset, Expr::Idx(0));
    }

    #[test]
    fn test_build_contiguous_offset_2d() {
        let offset = build_contiguous_offset(2, &[2, 3]);
        // offset = ridx0 * 3 + ridx1 * 1
        // After simplification, should be: ridx0 * 3 + ridx1
        match offset {
            Expr::Add(_, _) => {} // Expected
            _ => panic!("Expected Add expression, got {:?}", offset),
        }
    }

    #[test]
    fn test_build_gather_index_expr_1d() {
        // 1D gather: output[i] = input[index[i]]
        let expr = build_gather_index_expr(0, 1, &[4], &[3]);

        // Should contain LoadIndex
        assert!(
            expr.contains_load_index(),
            "Expression should contain LoadIndex"
        );
    }

    #[test]
    fn test_build_gather_index_expr_2d_dim0() {
        // 2D gather dim=0: output[i][j] = input[index[i][j]][j]
        let expr = build_gather_index_expr(0, 2, &[4, 5], &[3, 5]);

        assert!(
            expr.contains_load_index(),
            "Expression should contain LoadIndex"
        );
    }

    #[test]
    fn test_build_gather_index_expr_2d_dim1() {
        // 2D gather dim=1: output[i][j] = input[i][index[i][j]]
        let expr = build_gather_index_expr(1, 2, &[4, 5], &[4, 3]);

        assert!(
            expr.contains_load_index(),
            "Expression should contain LoadIndex"
        );
    }
}
