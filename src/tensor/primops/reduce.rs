//! Reduction primitive operations
//!
//! - Reduce(Add): sum reduction
//! - Reduce(Mul): product reduction
//! - Reduce(Max): max reduction

use std::rc::Rc;

use crate::graph::{DType, ops as graph_ops};
use crate::tensor::{DimDyn, Dimension, Tensor};

use super::binary::with_grad_fn;
use super::grad::{ReduceMaxBackward, ReduceMulBackward, ReduceSumBackward};

impl<D: Dimension> Tensor<D> {
    /// Sum reduction along specified axes (primop)
    ///
    /// # Arguments
    /// * `axes` - Axes to reduce over
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    pub fn reduce_sum(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
        let mut result_shape = self.shape().to_vec();
        let mut node = self.node.clone();

        // Sort axes in descending order to handle dimension removal correctly
        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            node = graph_ops::reduce_sum(node, axis);
            if keepdim {
                result_shape[axis] = 1;
            } else {
                result_shape.remove(axis);
            }
        }

        // If keepdim, reshape to restore dimensions
        if keepdim {
            let exprs: Vec<crate::graph::shape::Expr> = result_shape
                .iter()
                .map(|&s| crate::graph::shape::Expr::from(s as i64))
                .collect();
            node = node.reshape(exprs);
        }

        let result = Tensor::from_node(node, result_shape, DType::F32);

        if self.requires_grad() {
            let grad_fn = ReduceSumBackward::new(self.clone().into_dyn(), axes.to_vec(), keepdim);
            with_grad_fn(result, Some(Rc::new(grad_fn)))
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
        let mut result_shape = self.shape().to_vec();
        let mut node = self.node.clone();

        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            node = graph_ops::reduce_mul(node, axis);
            if keepdim {
                result_shape[axis] = 1;
            } else {
                result_shape.remove(axis);
            }
        }

        // If keepdim, reshape to restore dimensions
        if keepdim {
            let exprs: Vec<crate::graph::shape::Expr> = result_shape
                .iter()
                .map(|&s| crate::graph::shape::Expr::from(s as i64))
                .collect();
            node = node.reshape(exprs);
        }

        let result = Tensor::from_node(node, result_shape, DType::F32);

        if self.requires_grad() {
            let grad_fn = ReduceMulBackward::new(
                self.clone().into_dyn(),
                result.clone(),
                axes.to_vec(),
                keepdim,
            );
            with_grad_fn(result, Some(Rc::new(grad_fn)))
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
        let mut result_shape = self.shape().to_vec();
        let mut node = self.node.clone();

        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            node = graph_ops::reduce_max(node, axis);
            if keepdim {
                result_shape[axis] = 1;
            } else {
                result_shape.remove(axis);
            }
        }

        // If keepdim, reshape to restore dimensions
        if keepdim {
            let exprs: Vec<crate::graph::shape::Expr> = result_shape
                .iter()
                .map(|&s| crate::graph::shape::Expr::from(s as i64))
                .collect();
            node = node.reshape(exprs);
        }

        let result = Tensor::from_node(node, result_shape, DType::F32);

        if self.requires_grad() {
            let grad_fn = ReduceMaxBackward::new(
                self.clone().into_dyn(),
                result.clone(),
                axes.to_vec(),
                keepdim,
            );
            with_grad_fn(result, Some(Rc::new(grad_fn)))
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
