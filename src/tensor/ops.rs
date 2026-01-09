//! Tensor operations with type-safe dimension tracking
//!
//! This module provides operations on tensors that preserve
//! dimension information at compile time.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::ast::DType;
use crate::graph::Expr;

use super::dim::{DimAdd1, DimEq, DimSub1, Dimension};
use super::tensor::Tensor;

// ============================================================================
// Binary Operations (Element-wise)
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Element-wise addition.
    pub fn add<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph + &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise multiplication.
    pub fn mul<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph * &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise subtraction.
    pub fn sub<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph - &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise division.
    pub fn div<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph / &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise maximum.
    pub fn maximum<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.maximum(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise minimum.
    pub fn minimum<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.minimum(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Unary Operations (Element-wise)
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Element-wise negation.
    pub fn neg(&self) -> Tensor<D> {
        let graph = -&self.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise reciprocal (1/x).
    pub fn recip(&self) -> Tensor<D> {
        let graph = self.inner.graph.recip();
        Tensor::from_graph(graph)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Tensor<D> {
        let graph = self.inner.graph.sqrt();
        Tensor::from_graph(graph)
    }

    /// Element-wise natural logarithm.
    pub fn ln(&self) -> Tensor<D> {
        let graph = self.inner.graph.ln();
        Tensor::from_graph(graph)
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(&self) -> Tensor<D> {
        let graph = self.inner.graph.log2();
        Tensor::from_graph(graph)
    }

    /// Element-wise exponential (e^x).
    pub fn exp(&self) -> Tensor<D> {
        let graph = self.inner.graph.exp();
        Tensor::from_graph(graph)
    }

    /// Element-wise base-2 exponential (2^x).
    pub fn exp2(&self) -> Tensor<D> {
        let graph = self.inner.graph.exp2();
        Tensor::from_graph(graph)
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Tensor<D> {
        let graph = self.inner.graph.sin();
        Tensor::from_graph(graph)
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Tensor<D> {
        let graph = self.inner.graph.cos();
        Tensor::from_graph(graph)
    }

    /// Element-wise floor.
    pub fn floor(&self) -> Tensor<D> {
        let graph = self.inner.graph.floor();
        Tensor::from_graph(graph)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Tensor<D> {
        let graph = self.inner.graph.abs();
        Tensor::from_graph(graph)
    }

    /// Element-wise square (x^2).
    pub fn square(&self) -> Tensor<D> {
        self.mul(self)
    }

    /// Element-wise power.
    pub fn pow(&self, _exp: f32) -> Tensor<D> {
        // x^n = exp(n * ln(x))
        // TODO: Implement proper scalar constant multiplication
        // For now, we just return exp(ln(x)) = x
        self.ln().exp()
    }
}

// ============================================================================
// Reduce Operations
// ============================================================================

impl<D: Dimension + DimSub1> Tensor<D> {
    /// Sum over the specified axis, removing that dimension.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
    /// let y: Tensor<D1> = x.sum(1);  // [32, 64] -> [32]
    /// ```
    pub fn sum(&self, axis: usize) -> Tensor<D::Output> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        // GraphNode.sum() keeps the axis with size 1, so we need to squeeze it
        let graph = self.inner.graph.sum(axis).squeeze(axis);
        Tensor::from_graph(graph)
    }

    /// Maximum over the specified axis, removing that dimension.
    pub fn max(&self, axis: usize) -> Tensor<D::Output> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        // GraphNode.max() keeps the axis with size 1, so we need to squeeze it
        let graph = self.inner.graph.max(axis).squeeze(axis);
        Tensor::from_graph(graph)
    }

    /// Minimum over the specified axis, removing that dimension.
    pub fn min(&self, axis: usize) -> Tensor<D::Output> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        // GraphNode.min() keeps the axis with size 1, so we need to squeeze it
        let graph = self.inner.graph.min(axis).squeeze(axis);
        Tensor::from_graph(graph)
    }

    /// Product over the specified axis, removing that dimension.
    pub fn prod(&self, axis: usize) -> Tensor<D::Output> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        // GraphNode.prod() keeps the axis with size 1, so we need to squeeze it
        let graph = self.inner.graph.prod(axis).squeeze(axis);
        Tensor::from_graph(graph)
    }

    /// Mean over the specified axis, removing that dimension.
    pub fn mean(&self, axis: usize) -> Tensor<D::Output> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        let _shape = self.shape();
        let _n = _shape[axis];

        // mean = sum / n (implemented via reciprocal multiplication)

        // For now, we just return sum (proper scalar division needs backend support)
        // TODO: Implement proper scalar division
        self.sum(axis)
    }
}

// ============================================================================
// Shape Operations
// ============================================================================

impl<D: Dimension + DimAdd1> Tensor<D> {
    /// Insert a dimension of size 1 at the specified axis.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
    /// let y: Tensor<D3> = x.unsqueeze(0);  // [32, 64] -> [1, 32, 64]
    /// ```
    pub fn unsqueeze(&self, axis: usize) -> Tensor<D::Output> {
        assert!(
            axis <= self.ndim(),
            "Axis {} out of bounds for unsqueeze on {}D tensor",
            axis,
            self.ndim()
        );
        let graph = self.inner.graph.unsqueeze(axis);
        Tensor::from_graph(graph)
    }
}

impl<D: Dimension + DimSub1> Tensor<D> {
    /// Remove a dimension of size 1 at the specified axis.
    ///
    /// # Panics
    ///
    /// Panics if the dimension at `axis` is not 1.
    pub fn squeeze(&self, axis: usize) -> Tensor<D::Output> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        let graph = self.inner.graph.squeeze(axis);
        Tensor::from_graph(graph)
    }
}

impl<D: Dimension> Tensor<D> {
    /// Permute the tensor's axes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D3> = Tensor::input([2, 3, 4], DType::F32);
    /// let y: Tensor<D3> = x.permute(&[2, 0, 1]);  // [2, 3, 4] -> [4, 2, 3]
    /// ```
    pub fn permute(&self, axes: &[usize]) -> Tensor<D> {
        assert_eq!(
            axes.len(),
            self.ndim(),
            "Permutation length {} doesn't match tensor dims {}",
            axes.len(),
            self.ndim()
        );
        let graph = self.inner.graph.permute(axes);
        Tensor::from_graph(graph)
    }

    /// Transpose the last two dimensions.
    ///
    /// For a 2D tensor, this is the standard matrix transpose.
    pub fn t(&self) -> Tensor<D> {
        assert!(self.ndim() >= 2, "Transpose requires at least 2 dimensions");
        let n = self.ndim();
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 2, n - 1);
        self.permute(&axes)
    }

    /// Expand a dimension of size 1 to the specified size.
    ///
    /// This performs broadcasting.
    pub fn expand(&self, axis: usize, size: usize) -> Tensor<D> {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        let graph = self.inner.graph.expand(axis, Expr::Const(size as i64));
        Tensor::from_graph(graph)
    }

    /// Reshape the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape<D2: Dimension, const N: usize>(&self, shape: [usize; N]) -> Tensor<D2> {
        assert_eq!(
            N,
            D2::NDIM,
            "Shape length {} doesn't match target dimension {}",
            N,
            D2::NDIM
        );

        let current_numel: usize = self.shape().iter().product();
        let new_numel: usize = shape.iter().product();
        assert_eq!(
            current_numel, new_numel,
            "Cannot reshape tensor of {} elements to {} elements",
            current_numel, new_numel
        );

        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = self.inner.graph.reshape(expr_shape);
        Tensor::from_graph(graph)
    }

    /// Flatten the tensor to 1D.
    pub fn flatten(&self) -> Tensor<super::dim::D1> {
        let numel = self.numel();
        self.reshape([numel])
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Element-wise less than comparison.
    pub fn lt<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.lt(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise greater than comparison.
    pub fn gt<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.gt(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise less than or equal comparison.
    pub fn le<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.le(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise greater than or equal comparison.
    pub fn ge<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.ge(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise equality comparison.
    pub fn eq<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.eq_node(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise inequality comparison.
    pub fn ne<Rhs: Dimension>(&self, rhs: &Tensor<Rhs>) -> Tensor<D>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.ne_node(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Utility Methods
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Scale all elements by a constant.
    pub fn scale(&self, _factor: f32) -> Tensor<D> {
        // Create a scalar and multiply
        // This is a workaround since we can't easily create constants
        let ones = Self::ones_like(self);
        let factor_tensor = Self::from_graph(ones.inner.graph.clone());
        // For now, we use the graph's multiplication
        // TODO: Implement proper scalar multiplication
        self.mul(&factor_tensor)
    }

    /// Create a tensor of ones with the same shape as this tensor.
    pub fn ones_like(other: &Tensor<D>) -> Tensor<D> {
        let shape = other.shape();
        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = crate::graph::ones(expr_shape, other.dtype());
        Tensor::from_graph(graph)
    }

    /// Create a tensor of zeros with the same shape as this tensor.
    pub fn zeros_like(other: &Tensor<D>) -> Tensor<D> {
        let shape = other.shape();
        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = crate::graph::zeros(expr_shape, other.dtype());
        Tensor::from_graph(graph)
    }

    /// Conditional selection: where(cond, self, other).
    ///
    /// Returns elements from `self` where `cond` is true, otherwise from `other`.
    pub fn where_cond<C: Dimension, Rhs: Dimension>(
        &self,
        cond: &Tensor<C>,
        other: &Tensor<Rhs>,
    ) -> Tensor<D>
    where
        D: DimEq<C> + DimEq<Rhs>,
    {
        let graph = self
            .inner
            .graph
            .where_cond(&cond.inner.graph, &other.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Cast to a different data type.
    pub fn cast(&self, dtype: DType) -> Tensor<D> {
        let graph = self.inner.graph.cast(dtype);
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Operator Overloading
// ============================================================================

// Add: &Tensor + &Tensor
impl<D: Dimension> Add for &Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

// Add: Tensor + Tensor
impl<D: Dimension> Add for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add(&self, &rhs)
    }
}

// Add: &Tensor + Tensor
impl<D: Dimension> Add<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Tensor<D>) -> Self::Output {
        Tensor::add(self, &rhs)
    }
}

// Add: Tensor + &Tensor
impl<D: Dimension> Add<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::add(&self, rhs)
    }
}

// Sub: &Tensor - &Tensor
impl<D: Dimension> Sub for &Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

// Sub: Tensor - Tensor
impl<D: Dimension> Sub for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::sub(&self, &rhs)
    }
}

// Mul: &Tensor * &Tensor
impl<D: Dimension> Mul for &Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

// Mul: Tensor * Tensor
impl<D: Dimension> Mul for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::mul(&self, &rhs)
    }
}

// Div: &Tensor / &Tensor
impl<D: Dimension> Div for &Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::div(self, rhs)
    }
}

// Div: Tensor / Tensor
impl<D: Dimension> Div for Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::div(&self, &rhs)
    }
}

// Neg: -&Tensor
impl<D: Dimension> Neg for &Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

// Neg: -Tensor
impl<D: Dimension> Neg for Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dim::{D1, D2, D3};

    #[test]
    fn test_add() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let c: Tensor<D2> = &a + &b;
        assert_eq!(c.shape(), vec![32, 64]);
    }

    #[test]
    fn test_mul() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let c: Tensor<D2> = &a * &b;
        assert_eq!(c.shape(), vec![32, 64]);
    }

    #[test]
    fn test_neg() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D2> = -&a;
        assert_eq!(b.shape(), vec![32, 64]);
    }

    #[test]
    fn test_sum() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D1> = a.sum(1);
        assert_eq!(b.shape(), vec![32]);
    }

    #[test]
    fn test_unsqueeze() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D3> = a.unsqueeze(0);
        assert_eq!(b.shape(), vec![1, 32, 64]);
    }

    #[test]
    fn test_squeeze() {
        let a: Tensor<D3> = Tensor::input([1, 32, 64], DType::F32);
        let b: Tensor<D2> = a.squeeze(0);
        assert_eq!(b.shape(), vec![32, 64]);
    }

    #[test]
    fn test_permute() {
        let a: Tensor<D3> = Tensor::input([2, 3, 4], DType::F32);
        let b: Tensor<D3> = a.permute(&[2, 0, 1]);
        assert_eq!(b.shape(), vec![4, 2, 3]);
    }

    #[test]
    fn test_transpose() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D2> = a.t();
        assert_eq!(b.shape(), vec![64, 32]);
    }

    #[test]
    fn test_reshape() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D3> = a.reshape([4, 8, 64]);
        assert_eq!(b.shape(), vec![4, 8, 64]);
    }

    #[test]
    fn test_flatten() {
        let a: Tensor<D3> = Tensor::input([2, 3, 4], DType::F32);
        let b: Tensor<D1> = a.flatten();
        assert_eq!(b.shape(), vec![24]);
    }

    #[test]
    fn test_chained_operations() {
        let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let y: Tensor<D2> = Tensor::input([32, 64], DType::F32);

        // (x * y).sum(1)
        let result: Tensor<D1> = (&x * &y).sum(1);
        assert_eq!(result.shape(), vec![32]);
    }

    #[test]
    fn test_unary_ops() {
        let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);

        let _ = x.sqrt();
        let _ = x.exp();
        let _ = x.ln();
        let _ = x.sin();
        let _ = x.cos();
        let _ = x.abs();
        let _ = x.recip();
    }

    #[test]
    fn test_comparison_ops() {
        let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let b: Tensor<D2> = Tensor::input([32, 64], DType::F32);

        let _ = a.lt(&b);
        let _ = a.gt(&b);
        let _ = a.le(&b);
        let _ = a.ge(&b);
        let _ = a.eq(&b);
        let _ = a.ne(&b);
    }
}
