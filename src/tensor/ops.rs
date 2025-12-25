//! Operator implementations for Tensor
//!
//! This module provides operator overloading for Tensor<D>, enabling lazy
//! evaluation of computations via computation graph construction.
//!
//! ## Primitive Operations
//!
//! Following tinygrad/micrograd design philosophy:
//!
//! **Binary**: Add, Mul, Max
//! **Unary**: Neg, Recip, Exp2, Log2, Sin, Sqrt
//! **Movement**: Reshape, Permute, Expand, Shrink, Pad
//! **Reduce**: ReduceSum, ReduceMax

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::graph::{DType, ops as graph_ops, shape::Expr};

use super::{Dim, DimDyn, Dimension, Tensor};

/// Helper to compute result shape for binary operations with broadcasting
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_dim = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let b_dim = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            panic!(
                "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                a, b, i
            );
        }
    }
    result
}

// ============================================================================
// Binary Operations
// ============================================================================

// Add: Tensor + Tensor
impl<D: Dimension> Add for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: Self) -> Tensor<DimDyn> {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = &self.node + &rhs.node;
        Tensor::from_node(result_node, result_shape, DType::F32)
    }
}

impl<D: Dimension> Add<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self + &rhs
    }
}

impl<D: Dimension> Add<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        &self + rhs
    }
}

impl<D: Dimension> Add for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: Self) -> Tensor<DimDyn> {
        &self + &rhs
    }
}

// Add: Tensor + f32
impl<D: Dimension> Add<f32> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: f32) -> Tensor<DimDyn> {
        let result_node = &self.node + rhs;
        Tensor::from_node(result_node, self.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Add<f32> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: f32) -> Tensor<DimDyn> {
        &self + rhs
    }
}

// Add: f32 + Tensor
impl<D: Dimension> Add<&Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        let result_node = self + &rhs.node;
        Tensor::from_node(result_node, rhs.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Add<Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn add(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self + &rhs
    }
}

// Sub: Tensor - Tensor
impl<D: Dimension> Sub for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: Self) -> Tensor<DimDyn> {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = &self.node - &rhs.node;
        Tensor::from_node(result_node, result_shape, DType::F32)
    }
}

impl<D: Dimension> Sub<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self - &rhs
    }
}

impl<D: Dimension> Sub<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        &self - rhs
    }
}

impl<D: Dimension> Sub for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: Self) -> Tensor<DimDyn> {
        &self - &rhs
    }
}

// Sub: Tensor - f32
impl<D: Dimension> Sub<f32> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: f32) -> Tensor<DimDyn> {
        let result_node = &self.node - rhs;
        Tensor::from_node(result_node, self.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Sub<f32> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: f32) -> Tensor<DimDyn> {
        &self - rhs
    }
}

// Sub: f32 - Tensor
impl<D: Dimension> Sub<&Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        let result_node = self - &rhs.node;
        Tensor::from_node(result_node, rhs.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Sub<Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn sub(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self - &rhs
    }
}

// Mul: Tensor * Tensor
impl<D: Dimension> Mul for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: Self) -> Tensor<DimDyn> {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = &self.node * &rhs.node;
        Tensor::from_node(result_node, result_shape, DType::F32)
    }
}

impl<D: Dimension> Mul<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self * &rhs
    }
}

impl<D: Dimension> Mul<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        &self * rhs
    }
}

impl<D: Dimension> Mul for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: Self) -> Tensor<DimDyn> {
        &self * &rhs
    }
}

// Mul: Tensor * f32
impl<D: Dimension> Mul<f32> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: f32) -> Tensor<DimDyn> {
        let result_node = &self.node * rhs;
        Tensor::from_node(result_node, self.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Mul<f32> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: f32) -> Tensor<DimDyn> {
        &self * rhs
    }
}

// Mul: f32 * Tensor
impl<D: Dimension> Mul<&Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        let result_node = self * &rhs.node;
        Tensor::from_node(result_node, rhs.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Mul<Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn mul(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self * &rhs
    }
}

// Div: Tensor / Tensor
impl<D: Dimension> Div for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: Self) -> Tensor<DimDyn> {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = &self.node / &rhs.node;
        Tensor::from_node(result_node, result_shape, DType::F32)
    }
}

impl<D: Dimension> Div<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self / &rhs
    }
}

impl<D: Dimension> Div<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        &self / rhs
    }
}

impl<D: Dimension> Div for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: Self) -> Tensor<DimDyn> {
        &self / &rhs
    }
}

// Div: Tensor / f32
impl<D: Dimension> Div<f32> for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: f32) -> Tensor<DimDyn> {
        let result_node = &self.node / rhs;
        Tensor::from_node(result_node, self.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Div<f32> for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: f32) -> Tensor<DimDyn> {
        &self / rhs
    }
}

// Div: f32 / Tensor
impl<D: Dimension> Div<&Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: &Tensor<D>) -> Tensor<DimDyn> {
        let result_node = self / &rhs.node;
        Tensor::from_node(result_node, rhs.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Div<Tensor<D>> for f32 {
    type Output = Tensor<DimDyn>;

    fn div(self, rhs: Tensor<D>) -> Tensor<DimDyn> {
        self / &rhs
    }
}

// Neg: -Tensor
impl<D: Dimension> Neg for &Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn neg(self) -> Tensor<DimDyn> {
        let result_node = -&self.node;
        Tensor::from_node(result_node, self.shape().to_vec(), DType::F32)
    }
}

impl<D: Dimension> Neg for Tensor<D> {
    type Output = Tensor<DimDyn>;

    fn neg(self) -> Tensor<DimDyn> {
        -&self
    }
}

// ============================================================================
// Unary Operations (as methods)
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Compute the reciprocal (1/x) of each element
    pub fn recip(&self) -> Tensor<DimDyn> {
        let result_node = graph_ops::recip(self.node.clone());
        Tensor::from_node(result_node, self.shape().to_vec(), DType::F32)
    }

    /// Compute exp2(x) = 2^x for each element
    pub fn exp2(&self) -> Tensor<DimDyn> {
        let node = self.node.clone().exp2();
        Tensor::from_node(node, self.shape().to_vec(), DType::F32)
    }

    /// Compute log2(x) for each element
    pub fn log2(&self) -> Tensor<DimDyn> {
        let node = self.node.clone().log2();
        Tensor::from_node(node, self.shape().to_vec(), DType::F32)
    }

    /// Compute sin(x) for each element
    pub fn sin(&self) -> Tensor<DimDyn> {
        let node = self.node.clone().sin();
        Tensor::from_node(node, self.shape().to_vec(), DType::F32)
    }

    /// Compute sqrt(x) for each element
    pub fn sqrt(&self) -> Tensor<DimDyn> {
        let node = self.node.clone().sqrt();
        Tensor::from_node(node, self.shape().to_vec(), DType::F32)
    }

    /// Compute exp(x) = e^x for each element
    pub fn exp(&self) -> Tensor<DimDyn> {
        let node = self.node.clone().exp();
        Tensor::from_node(node, self.shape().to_vec(), DType::F32)
    }

    /// Compute natural logarithm ln(x) for each element
    pub fn ln(&self) -> Tensor<DimDyn> {
        let node = self.node.clone().log();
        Tensor::from_node(node, self.shape().to_vec(), DType::F32)
    }

    /// Compute element-wise maximum with another tensor
    pub fn max_with<E: Dimension>(&self, other: &Tensor<E>) -> Tensor<DimDyn> {
        let result_shape = broadcast_shapes(self.shape(), other.shape());
        let result_node = graph_ops::max(self.node.clone(), other.node.clone());
        Tensor::from_node(result_node, result_shape, DType::F32)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor<DimDyn> {
        let zero = Tensor::<DimDyn>::full_dyn(self.shape(), 0.0);
        self.max_with(&zero)
    }
}

// ============================================================================
// Reduce Operations
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Sum reduction along specified axes
    ///
    /// # Arguments
    /// * `axes` - Axes to reduce over
    /// * `keepdim` - Whether to keep reduced dimensions as size 1
    pub fn sum(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
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

        Tensor::from_node(node, result_shape, DType::F32)
    }

    /// Max reduction along specified axes
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

        Tensor::from_node(node, result_shape, DType::F32)
    }

    /// Mean reduction along specified axes
    pub fn mean(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
        let mut count: usize = 1;
        for &axis in axes {
            count *= self.shape()[axis];
        }
        let sum = self.sum(axes, keepdim);
        sum / (count as f32)
    }
}

// ============================================================================
// Movement Operations
// ============================================================================

/// Convert usize slice to Expr vector
fn to_expr_vec(shape: &[usize]) -> Vec<Expr> {
    shape.iter().map(|&s| Expr::from(s as i64)).collect()
}

impl<D: Dimension> Tensor<D> {
    /// Reshape the tensor to a new shape
    ///
    /// Total number of elements must remain the same.
    pub fn reshape<const M: usize>(&self, new_shape: [usize; M]) -> Tensor<Dim<M>> {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of size {} to shape {:?} (size {})",
            self.numel(),
            new_shape,
            new_numel
        );

        let expr_shape = to_expr_vec(&new_shape);
        let node = self.node.reshape(expr_shape);
        Tensor::from_node(node, new_shape.to_vec(), self.dtype.clone())
    }

    /// Reshape to dynamic shape
    pub fn reshape_dyn(&self, new_shape: &[usize]) -> Tensor<DimDyn> {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of size {} to shape {:?} (size {})",
            self.numel(),
            new_shape,
            new_numel
        );

        let expr_shape = to_expr_vec(new_shape);
        let node = self.node.reshape(expr_shape);
        Tensor::from_node(node, new_shape.to_vec(), self.dtype.clone())
    }

    /// Permute tensor dimensions
    ///
    /// # Arguments
    /// * `axes` - New order of dimensions
    pub fn permute(&self, axes: &[usize]) -> Tensor<DimDyn> {
        assert_eq!(
            axes.len(),
            self.ndim(),
            "Permutation must have same number of axes as tensor dimensions"
        );

        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape()[i]).collect();
        let new_view = self.node.view.clone().permute(axes.to_vec());
        let node = self.node.view(new_view);
        Tensor::from_node(node, new_shape, self.dtype.clone())
    }

    /// Transpose the tensor (swap last two dimensions)
    pub fn transpose(&self) -> Tensor<DimDyn> {
        assert!(self.ndim() >= 2, "Transpose requires at least 2 dimensions");

        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        let n = axes.len();
        axes.swap(n - 2, n - 1);
        self.permute(&axes)
    }

    /// Expand tensor to a larger shape (broadcast)
    ///
    /// Dimensions of size 1 can be expanded to larger sizes.
    pub fn expand(&self, new_shape: &[usize]) -> Tensor<DimDyn> {
        assert_eq!(
            new_shape.len(),
            self.ndim(),
            "Expand shape must have same number of dimensions"
        );

        for (i, (&old, &new)) in self.shape().iter().zip(new_shape.iter()).enumerate() {
            assert!(
                old == new || old == 1,
                "Cannot expand dimension {} from {} to {}",
                i,
                old,
                new
            );
        }

        let target_expr = to_expr_vec(new_shape);
        let node = self.node.broadcast_to(target_expr);
        Tensor::from_node(node, new_shape.to_vec(), self.dtype.clone())
    }

    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Tensor<Dim<1>> {
        self.reshape([self.numel()])
    }

    /// Unsqueeze: add a dimension of size 1 at the specified position
    pub fn unsqueeze(&self, dim: usize) -> Tensor<DimDyn> {
        assert!(
            dim <= self.ndim(),
            "Dimension {} out of range for tensor with {} dimensions",
            dim,
            self.ndim()
        );

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(dim, 1);
        self.reshape_dyn(&new_shape)
    }

    /// Squeeze: remove dimensions of size 1
    pub fn squeeze(&self) -> Tensor<DimDyn> {
        let new_shape: Vec<usize> = self.shape().iter().filter(|&&s| s != 1).copied().collect();
        if new_shape.is_empty() {
            // Scalar case
            self.reshape_dyn(&[1])
        } else {
            self.reshape_dyn(&new_shape)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_add_tensors() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = Tensor::<Dim2>::ones([2, 3]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::<Dim2>::zeros([2, 3]);
        let c = &a + 1.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sub_tensors() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = Tensor::<Dim2>::ones([2, 3]);
        let c = &a - &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_mul_tensors() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = Tensor::<Dim2>::full([2, 3], 2.0);
        let c = &a * &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_tensors() {
        let a = Tensor::<Dim2>::full([2, 3], 6.0);
        let b = Tensor::<Dim2>::full([2, 3], 2.0);
        let c = &a / &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = -&a;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_broadcast_add() {
        // Broadcasting requires explicit expand operation
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = Tensor::<Dim<1>>::ones([3]);
        // First unsqueeze b to shape [1, 3], then expand to [2, 3]
        let b_expanded = b.unsqueeze(0).expand(&[2, 3]);
        let c = &a.into_dyn() + &b_expanded;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_unary_ops() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let _ = a.recip();
        let _ = a.exp2();
        let _ = a.log2();
        let _ = a.sin();
        let _ = a.sqrt();
        let _ = a.exp();
        let _ = a.ln();
    }

    #[test]
    fn test_relu() {
        let a = Tensor::<Dim2>::full([2, 3], -1.0);
        let r = a.relu();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_sum() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.sum(&[1], false);
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_sum_keepdim() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.sum(&[1], true);
        assert_eq!(s.shape(), &[2, 1]);
    }

    #[test]
    fn test_mean() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let m = a.mean(&[1], false);
        assert_eq!(m.shape(), &[2]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.reshape([6]);
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_permute() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.permute(&[1, 0]);
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.transpose();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_expand() {
        let a = Tensor::<Dim2>::ones([1, 3]);
        let b = a.expand(&[4, 3]);
        assert_eq!(b.shape(), &[4, 3]);
    }

    #[test]
    fn test_flatten() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.flatten();
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_unsqueeze() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.unsqueeze(0);
        assert_eq!(b.shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_squeeze() {
        let a = Tensor::<DimDyn>::ones_dyn(&[1, 2, 1, 3]);
        let b = a.squeeze();
        assert_eq!(b.shape(), &[2, 3]);
    }
}
