//! Unary primitive operations
//!
//! - Neg: negation (-x)
//! - Recip: reciprocal (1/x)
//! - Sqrt: square root
//! - Log2: base-2 logarithm
//! - Exp2: base-2 exponential
//! - Sin: sine

use std::ops::Neg;
use std::rc::Rc;

use crate::graph::{DType, ops as graph_ops};
use crate::tensor::{Dimension, Tensor};

use super::binary::with_grad_fn;
use super::grad::{Exp2Backward, Log2Backward, NegBackward, RecipBackward, SinBackward, SqrtBackward};

// ============================================================================
// Neg: -Tensor
// ============================================================================

impl<D: Dimension> Neg for &Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Tensor<D> {
        let result_node = -&self.node;
        let result = Tensor::from_node(result_node, self.shape().to_vec(), DType::F32);

        if self.requires_grad() {
            let grad_fn = NegBackward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Neg for Tensor<D> {
    type Output = Tensor<D>;
    fn neg(self) -> Tensor<D> {
        -&self
    }
}

// ============================================================================
// Unary primops as methods
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Compute the reciprocal (1/x) of each element (primop)
    pub fn recip(&self) -> Tensor<D> {
        let result_node = graph_ops::recip(self.node.clone());
        let result = Tensor::from_node(result_node, self.shape().to_vec(), DType::F32);

        if self.requires_grad() {
            let grad_fn = RecipBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute sqrt(x) for each element (primop)
    pub fn sqrt(&self) -> Tensor<D> {
        let result_node = self.node.clone().sqrt();
        let result = Tensor::from_node(result_node, self.shape().to_vec(), DType::F32);

        if self.requires_grad() {
            let grad_fn = SqrtBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute log2(x) for each element (primop)
    pub fn log2(&self) -> Tensor<D> {
        let result_node = self.node.clone().log2();
        let result = Tensor::from_node(result_node, self.shape().to_vec(), DType::F32);

        if self.requires_grad() {
            let grad_fn = Log2Backward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute exp2(x) = 2^x for each element (primop)
    pub fn exp2(&self) -> Tensor<D> {
        let result_node = self.node.clone().exp2();
        let result = Tensor::from_node(result_node, self.shape().to_vec(), DType::F32);

        if self.requires_grad() {
            let grad_fn = Exp2Backward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute sin(x) for each element (primop)
    pub fn sin(&self) -> Tensor<D> {
        let result_node = self.node.clone().sin();
        let result = Tensor::from_node(result_node, self.shape().to_vec(), DType::F32);

        if self.requires_grad() {
            let grad_fn = SinBackward::new(self.clone().into_dyn());
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
    fn test_neg() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = -&a;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_recip() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.recip();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.sqrt();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_log2() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.log2();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_exp2() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.exp2();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sin() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.sin();
        assert_eq!(c.shape(), &[2, 3]);
    }
}
