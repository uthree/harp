//! Tensor operations with type-safe dimension tracking
//!
//! This module provides operations on tensors that preserve
//! dimension information at compile time.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::ast::TensorDType;
use crate::graph::Expr;

use super::dim::{DimEq, Dimension, Dyn};
use super::tensor::Tensor;

// ============================================================================
// Binary Operations (Element-wise)
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Element-wise addition.
    /// Both tensors must have the same data type.
    pub fn add<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph + &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise multiplication.
    /// Both tensors must have the same data type.
    pub fn mul<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph * &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise subtraction.
    /// Both tensors must have the same data type.
    pub fn sub<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph - &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise division.
    /// Both tensors must have the same data type.
    pub fn div<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = &self.inner.graph / &rhs.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise maximum.
    /// Both tensors must have the same data type.
    pub fn maximum<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.maximum(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise minimum.
    /// Both tensors must have the same data type.
    pub fn minimum<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
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

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Element-wise negation.
    pub fn neg(&self) -> Tensor<D, T> {
        let graph = -&self.inner.graph;
        Tensor::from_graph(graph)
    }

    /// Element-wise reciprocal (1/x).
    pub fn recip(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.recip();
        Tensor::from_graph(graph)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.sqrt();
        Tensor::from_graph(graph)
    }

    /// Element-wise natural logarithm.
    pub fn ln(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.ln();
        Tensor::from_graph(graph)
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.log2();
        Tensor::from_graph(graph)
    }

    /// Element-wise exponential (e^x).
    pub fn exp(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.exp();
        Tensor::from_graph(graph)
    }

    /// Element-wise base-2 exponential (2^x).
    pub fn exp2(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.exp2();
        Tensor::from_graph(graph)
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.sin();
        Tensor::from_graph(graph)
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.cos();
        Tensor::from_graph(graph)
    }

    /// Element-wise floor.
    pub fn floor(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.floor();
        Tensor::from_graph(graph)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.abs();
        Tensor::from_graph(graph)
    }

    /// Element-wise square (x^2).
    pub fn square(&self) -> Tensor<D, T> {
        self.mul(self)
    }

    /// Element-wise power.
    pub fn pow(&self, _exp: f32) -> Tensor<D, T> {
        // x^n = exp(n * ln(x))
        // TODO: Implement proper scalar constant multiplication
        // For now, we just return exp(ln(x)) = x
        self.ln().exp()
    }
}

// ============================================================================
// Reduce Operations
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Sum over the specified axis, removing that dimension.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// let y: Tensor<D1, f32> = x.sum(1);  // [32, 64] -> [32]
    /// ```
    pub fn sum(&self, axis: usize) -> Tensor<D::Smaller, T> {
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
    pub fn max(&self, axis: usize) -> Tensor<D::Smaller, T> {
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
    pub fn min(&self, axis: usize) -> Tensor<D::Smaller, T> {
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
    pub fn prod(&self, axis: usize) -> Tensor<D::Smaller, T> {
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
    pub fn mean(&self, axis: usize) -> Tensor<D::Smaller, T> {
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
// Cumulative Operations (Scan)
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Cumulative sum along an axis.
    ///
    /// Returns a tensor of the same shape where each element is the sum of all
    /// previous elements (inclusive) along the specified axis.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // cumsum([1, 2, 3, 4]) = [1, 3, 6, 10]
    /// let x: Tensor<D1, f32> = ...;
    /// let y = x.cumsum(0);
    /// ```
    pub fn cumsum(&self, axis: usize) -> Self {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        let graph = self.inner.graph.cumsum(axis);
        Tensor::from_graph(graph)
    }

    /// Cumulative product along an axis.
    ///
    /// Returns a tensor of the same shape where each element is the product of all
    /// previous elements (inclusive) along the specified axis.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // cumprod([1, 2, 3, 4]) = [1, 2, 6, 24]
    /// let x: Tensor<D1, f32> = ...;
    /// let y = x.cumprod(0);
    /// ```
    pub fn cumprod(&self, axis: usize) -> Self {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        let graph = self.inner.graph.cumprod(axis);
        Tensor::from_graph(graph)
    }

    /// Cumulative maximum along an axis.
    ///
    /// Returns a tensor of the same shape where each element is the maximum of all
    /// previous elements (inclusive) along the specified axis.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // cummax([3, 1, 4, 1]) = [3, 3, 4, 4]
    /// let x: Tensor<D1, f32> = ...;
    /// let y = x.cummax(0);
    /// ```
    pub fn cummax(&self, axis: usize) -> Self {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );
        let graph = self.inner.graph.cummax(axis);
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Shape Operations
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Insert a dimension of size 1 at the specified axis.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// let y: Tensor<D3, f32> = x.unsqueeze(0);  // [32, 64] -> [1, 32, 64]
    /// ```
    pub fn unsqueeze(&self, axis: usize) -> Tensor<D::Larger, T> {
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

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Remove a dimension of size 1 at the specified axis.
    ///
    /// # Panics
    ///
    /// Panics if the dimension at `axis` is not 1.
    pub fn squeeze(&self, axis: usize) -> Tensor<D::Smaller, T> {
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

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Permute the tensor's axes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D3, f32> = Tensor::input([2, 3, 4]);
    /// let y: Tensor<D3, f32> = x.permute(&[2, 0, 1]);  // [2, 3, 4] -> [4, 2, 3]
    /// ```
    pub fn permute(&self, axes: &[usize]) -> Tensor<D, T> {
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
    pub fn t(&self) -> Tensor<D, T> {
        assert!(self.ndim() >= 2, "Transpose requires at least 2 dimensions");
        let n = self.ndim();
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 2, n - 1);
        self.permute(&axes)
    }

    /// Make the tensor contiguous in memory.
    ///
    /// After operations like `permute`, the tensor may have a non-contiguous
    /// memory layout. `contiguous()` creates a copy with contiguous memory,
    /// which is required before operations like `reshape`.
    pub fn contiguous(&self) -> Tensor<D, T> {
        let graph = self.inner.graph.contiguous();
        Tensor::from_graph(graph)
    }

    /// Expand a dimension of size 1 to the specified size.
    ///
    /// This performs broadcasting.
    pub fn expand(&self, axis: usize, size: usize) -> Tensor<D, T> {
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
    /// The data type is preserved.
    pub fn reshape<D2: Dimension, const N: usize>(&self, shape: [usize; N]) -> Tensor<D2, T> {
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
    pub fn flatten(&self) -> Tensor<super::dim::D1, T> {
        let numel = self.numel();
        self.reshape([numel])
    }
}

// ============================================================================
// Unfold / Fold Operations
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Unfold (sliding window extraction)
    ///
    /// Extracts sliding windows from the specified axes.
    /// The output shape is: [preserved_dims..., output_positions..., window_dims...]
    ///
    /// # Arguments
    /// * `axes` - Axes to unfold (must be sorted)
    /// * `sizes` - Window size for each axis
    /// * `strides` - Stride for each axis
    /// * `dilations` - Dilation for each axis
    ///
    /// # Example
    /// ```ignore
    /// // Input: [1, 3, 28, 28] (NCHW)
    /// // Unfold H and W with 3x3 kernel, stride 1, dilation 1
    /// let unfolded: Tensor<Dyn, f32> = x.unfold(&[2, 3], &[3, 3], &[1, 1], &[1, 1]);
    /// // Output: [1, 3, 26, 26, 3, 3]
    /// ```
    pub fn unfold(
        &self,
        axes: &[usize],
        sizes: &[usize],
        strides: &[usize],
        dilations: &[usize],
    ) -> Tensor<Dyn, T> {
        let graph = self.inner.graph.unfold(axes, sizes, strides, dilations);
        Tensor::from_graph(graph)
    }

    /// Fold (inverse of unfold)
    ///
    /// Accumulates values from unfolded windows back to the original shape.
    /// This is the inverse operation of unfold, using scatter-add semantics.
    /// Overlapping windows are summed together.
    ///
    /// # Arguments
    /// * `output_shape` - The target output shape
    /// * `axes` - Axes that were unfolded
    /// * `sizes` - Window size for each axis
    /// * `strides` - Stride for each axis
    /// * `dilations` - Dilation for each axis
    ///
    /// # Example
    /// ```ignore
    /// // Reverse of unfold: [1, 3, 26, 26, 3, 3] -> [1, 3, 28, 28]
    /// let folded: Tensor<Dyn, f32> = unfolded.fold(&[1, 3, 28, 28], &[2, 3], &[3, 3], &[1, 1], &[1, 1]);
    /// ```
    pub fn fold(
        &self,
        output_shape: &[usize],
        axes: &[usize],
        sizes: &[usize],
        strides: &[usize],
        dilations: &[usize],
    ) -> Tensor<Dyn, T> {
        let graph = self
            .inner
            .graph
            .fold(output_shape, axes, sizes, strides, dilations);
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Padding Operations
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Pad the tensor with a constant value
    ///
    /// Adds padding to each dimension. The padding is specified as a slice of
    /// (before, after) tuples, one for each dimension.
    ///
    /// # Arguments
    /// * `padding` - Padding for each dimension as (before, after)
    ///
    /// # Example
    /// ```ignore
    /// // Input: [1, 3, 28, 28]
    /// // Pad H and W with 1 on each side
    /// let padded = x.pad(&[(0, 0), (0, 0), (1, 1), (1, 1)]);
    /// // Output: [1, 3, 30, 30]
    /// ```
    pub fn pad(&self, padding: &[(usize, usize)]) -> Tensor<D, T> {
        use crate::graph::shape::PadValue;
        let graph = self.inner.graph.pad(padding, PadValue::Zero);
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Specialized Unfold Operations (Static Dimension Types)
// ============================================================================

use super::dim::{D3, D4, D5, D6, D8};

impl<T: TensorDType> Tensor<D4, T> {
    /// 2D Unfold for Conv2d (static dimension version)
    ///
    /// Extracts sliding windows from H and W axes.
    /// Input: [N, C, H, W] -> Output: [N, C, H_out, W_out, kH, kW]
    ///
    /// # Arguments
    /// * `kernel_size` - (kH, kW)
    /// * `stride` - (stride_h, stride_w)
    /// * `dilation` - (dilation_h, dilation_w)
    pub fn unfold_2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor<D6, T> {
        let graph = self.inner.graph.unfold(
            &[2, 3],
            &[kernel_size.0, kernel_size.1],
            &[stride.0, stride.1],
            &[dilation.0, dilation.1],
        );
        Tensor::from_graph(graph)
    }
}

impl<T: TensorDType> Tensor<D3, T> {
    /// 1D Unfold for Conv1d (static dimension version)
    ///
    /// Extracts sliding windows from the L axis.
    /// Input: [N, C, L] -> Output: [N, C, L_out, K]
    ///
    /// # Arguments
    /// * `kernel_size` - K
    /// * `stride` - stride
    /// * `dilation` - dilation
    pub fn unfold_1d(&self, kernel_size: usize, stride: usize, dilation: usize) -> Tensor<D4, T> {
        let graph = self
            .inner
            .graph
            .unfold(&[2], &[kernel_size], &[stride], &[dilation]);
        Tensor::from_graph(graph)
    }
}

impl<T: TensorDType> Tensor<D5, T> {
    /// 3D Unfold for Conv3d (static dimension version)
    ///
    /// Extracts sliding windows from D, H, and W axes.
    /// Input: [N, C, D, H, W] -> Output: [N, C, D_out, H_out, W_out, kD, kH, kW]
    ///
    /// # Arguments
    /// * `kernel_size` - (kD, kH, kW)
    /// * `stride` - (stride_d, stride_h, stride_w)
    /// * `dilation` - (dilation_d, dilation_h, dilation_w)
    pub fn unfold_3d(
        &self,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Tensor<D8, T> {
        let graph = self.inner.graph.unfold(
            &[2, 3, 4],
            &[kernel_size.0, kernel_size.1, kernel_size.2],
            &[stride.0, stride.1, stride.2],
            &[dilation.0, dilation.1, dilation.2],
        );
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Element-wise less than comparison.
    /// Both tensors must have the same data type.
    pub fn lt<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.lt(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise greater than comparison.
    /// Both tensors must have the same data type.
    pub fn gt<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.gt(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise less than or equal comparison.
    /// Both tensors must have the same data type.
    pub fn le<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.le(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise greater than or equal comparison.
    /// Both tensors must have the same data type.
    pub fn ge<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.ge(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise equality comparison.
    /// Both tensors must have the same data type.
    pub fn eq<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
    where
        D: DimEq<Rhs>,
    {
        let graph = self.inner.graph.eq_node(&rhs.inner.graph);
        Tensor::from_graph(graph)
    }

    /// Element-wise inequality comparison.
    /// Both tensors must have the same data type.
    pub fn ne<Rhs: Dimension>(&self, rhs: &Tensor<Rhs, T>) -> Tensor<D, T>
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

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Scale all elements by a constant.
    pub fn scale(&self, _factor: f32) -> Tensor<D, T> {
        // Create a scalar and multiply
        // This is a workaround since we can't easily create constants
        let ones = Self::ones_like(self);
        let factor_tensor = Self::from_graph(ones.inner.graph.clone());
        // For now, we use the graph's multiplication
        // TODO: Implement proper scalar multiplication
        self.mul(&factor_tensor)
    }

    /// Create a tensor of ones with the same shape as this tensor.
    pub fn ones_like(other: &Tensor<D, T>) -> Tensor<D, T> {
        let shape = other.shape();
        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = crate::graph::ones(expr_shape, T::DTYPE);
        Tensor::from_graph(graph)
    }

    /// Create a tensor of zeros with the same shape as this tensor.
    pub fn zeros_like(other: &Tensor<D, T>) -> Tensor<D, T> {
        let shape = other.shape();
        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = crate::graph::zeros(expr_shape, T::DTYPE);
        Tensor::from_graph(graph)
    }

    /// Conditional selection: where(cond, self, other).
    ///
    /// Returns elements from `self` where `cond` is true, otherwise from `other`.
    /// All tensors must have the same data type.
    pub fn where_cond<C: Dimension, Rhs: Dimension>(
        &self,
        cond: &Tensor<C, T>,
        other: &Tensor<Rhs, T>,
    ) -> Tensor<D, T>
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
    ///
    /// Returns a new tensor with the specified data type.
    pub fn cast<U: TensorDType>(&self) -> Tensor<D, U> {
        let graph = self.inner.graph.cast(U::DTYPE);
        Tensor::from_graph(graph)
    }
}

// ============================================================================
// Operator Overloading
// ============================================================================

// Add: &Tensor + &Tensor
impl<D: Dimension, T: TensorDType> Add for &Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

// Add: Tensor + Tensor
impl<D: Dimension, T: TensorDType> Add for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add(&self, &rhs)
    }
}

// Add: &Tensor + Tensor
impl<D: Dimension, T: TensorDType> Add<Tensor<D, T>> for &Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn add(self, rhs: Tensor<D, T>) -> Self::Output {
        Tensor::add(self, &rhs)
    }
}

// Add: Tensor + &Tensor
impl<D: Dimension, T: TensorDType> Add<&Tensor<D, T>> for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn add(self, rhs: &Tensor<D, T>) -> Self::Output {
        Tensor::add(&self, rhs)
    }
}

// Sub: &Tensor - &Tensor
impl<D: Dimension, T: TensorDType> Sub for &Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

// Sub: Tensor - Tensor
impl<D: Dimension, T: TensorDType> Sub for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::sub(&self, &rhs)
    }
}

// Mul: &Tensor * &Tensor
impl<D: Dimension, T: TensorDType> Mul for &Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

// Mul: Tensor * Tensor
impl<D: Dimension, T: TensorDType> Mul for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::mul(&self, &rhs)
    }
}

// Div: &Tensor / &Tensor
impl<D: Dimension, T: TensorDType> Div for &Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::div(self, rhs)
    }
}

// Div: Tensor / Tensor
impl<D: Dimension, T: TensorDType> Div for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::div(&self, &rhs)
    }
}

// Neg: -&Tensor
impl<D: Dimension, T: TensorDType> Neg for &Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

// Neg: -Tensor
impl<D: Dimension, T: TensorDType> Neg for Tensor<D, T> {
    type Output = Tensor<D, T>;

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
    use crate::ast::DType;
    use crate::tensor::dim::{D1, D2, D3};

    #[test]
    fn test_add() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D2, f32> = Tensor::input([32, 64]);
        let c: Tensor<D2, f32> = &a + &b;
        assert_eq!(c.shape(), vec![32, 64]);
    }

    #[test]
    fn test_mul() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D2, f32> = Tensor::input([32, 64]);
        let c: Tensor<D2, f32> = &a * &b;
        assert_eq!(c.shape(), vec![32, 64]);
    }

    #[test]
    fn test_neg() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D2, f32> = -&a;
        assert_eq!(b.shape(), vec![32, 64]);
    }

    #[test]
    fn test_sum() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D1, f32> = a.sum(1);
        assert_eq!(b.shape(), vec![32]);
    }

    #[test]
    fn test_unsqueeze() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D3, f32> = a.unsqueeze(0);
        assert_eq!(b.shape(), vec![1, 32, 64]);
    }

    #[test]
    fn test_squeeze() {
        let a: Tensor<D3, f32> = Tensor::input([1, 32, 64]);
        let b: Tensor<D2, f32> = a.squeeze(0);
        assert_eq!(b.shape(), vec![32, 64]);
    }

    #[test]
    fn test_permute() {
        let a: Tensor<D3, f32> = Tensor::input([2, 3, 4]);
        let b: Tensor<D3, f32> = a.permute(&[2, 0, 1]);
        assert_eq!(b.shape(), vec![4, 2, 3]);
    }

    #[test]
    fn test_transpose() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D2, f32> = a.t();
        assert_eq!(b.shape(), vec![64, 32]);
    }

    #[test]
    fn test_reshape() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D3, f32> = a.reshape([4, 8, 64]);
        assert_eq!(b.shape(), vec![4, 8, 64]);
    }

    #[test]
    fn test_flatten() {
        let a: Tensor<D3, f32> = Tensor::input([2, 3, 4]);
        let b: Tensor<D1, f32> = a.flatten();
        assert_eq!(b.shape(), vec![24]);
    }

    #[test]
    fn test_chained_operations() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        let y: Tensor<D2, f32> = Tensor::input([32, 64]);

        // (x * y).sum(1)
        let result: Tensor<D1, f32> = (&x * &y).sum(1);
        assert_eq!(result.shape(), vec![32]);
    }

    #[test]
    fn test_unary_ops() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);

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
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D2, f32> = Tensor::input([32, 64]);

        let _ = a.lt(&b);
        let _ = a.gt(&b);
        let _ = a.le(&b);
        let _ = a.ge(&b);
        let _ = a.eq(&b);
        let _ = a.ne(&b);
    }

    #[test]
    fn test_cast() {
        let a: Tensor<D2, f32> = Tensor::input([32, 64]);
        let b: Tensor<D2, i32> = a.cast();
        assert_eq!(b.shape(), vec![32, 64]);
        assert_eq!(b.dtype(), DType::I32);
    }

    #[test]
    fn test_unfold_2d() {
        use crate::tensor::dim::D4;

        // Input: [1, 3, 28, 28] (NCHW)
        let x: Tensor<D4, f32> = Tensor::input([1, 3, 28, 28]);

        // Unfold H and W axes with 3x3 kernel, stride 1, dilation 1
        // Output: [1, 3, 26, 26, 3, 3]
        let unfolded = x.unfold(&[2, 3], &[3, 3], &[1, 1], &[1, 1]);

        // Expected output shape:
        // - Preserved: N=1, C=3
        // - Output positions: (28 - 3) / 1 + 1 = 26 for both H and W
        // - Window: 3x3
        assert_eq!(unfolded.shape(), vec![1, 3, 26, 26, 3, 3]);
    }

    #[test]
    fn test_unfold_1d() {
        // Input: [1, 64, 100] (N, C, L)
        let x: Tensor<D3, f32> = Tensor::input([1, 64, 100]);

        // Unfold L axis with kernel size 5, stride 2, dilation 1
        // Output: [1, 64, 48, 5]
        let unfolded = x.unfold(&[2], &[5], &[2], &[1]);

        // Expected output shape:
        // - Preserved: N=1, C=64
        // - Output positions: (100 - 5) / 2 + 1 = 48
        // - Window: 5
        assert_eq!(unfolded.shape(), vec![1, 64, 48, 5]);
    }

    #[test]
    fn test_unfold_with_dilation() {
        use crate::tensor::dim::D4;

        // Input: [1, 3, 28, 28]
        let x: Tensor<D4, f32> = Tensor::input([1, 3, 28, 28]);

        // Unfold with dilation 2
        // effective_size = (3 - 1) * 2 + 1 = 5
        // Output positions: (28 - 5) / 1 + 1 = 24
        let unfolded = x.unfold(&[2, 3], &[3, 3], &[1, 1], &[2, 2]);
        assert_eq!(unfolded.shape(), vec![1, 3, 24, 24, 3, 3]);
    }

    #[test]
    fn test_fold() {
        use crate::tensor::dim::D6;

        // Simulating reverse of unfold
        // Unfolded shape: [1, 3, 26, 26, 3, 3] (from 28x28 with 3x3 kernel, stride 1)
        let x: Tensor<D6, f32> = Tensor::input([1, 3, 26, 26, 3, 3]);

        // Fold back to original shape
        let folded = x.fold(&[1, 3, 28, 28], &[2, 3], &[3, 3], &[1, 1], &[1, 1]);
        assert_eq!(folded.shape(), vec![1, 3, 28, 28]);
    }

    #[test]
    fn test_unfold_fold_shape_consistency() {
        use crate::tensor::dim::D4;

        // Test that unfold -> fold preserves the shape (not values, due to overlapping sums)
        let x: Tensor<D4, f32> = Tensor::input([1, 3, 10, 10]);

        // Unfold
        let unfolded = x.unfold(&[2, 3], &[3, 3], &[1, 1], &[1, 1]);
        assert_eq!(unfolded.shape(), vec![1, 3, 8, 8, 3, 3]);

        // Fold back
        let folded = unfolded.fold(&[1, 3, 10, 10], &[2, 3], &[3, 3], &[1, 1], &[1, 1]);
        assert_eq!(folded.shape(), vec![1, 3, 10, 10]);
    }

    // ========================================================================
    // Cumulative Operations Tests
    // ========================================================================

    #[test]
    fn test_cumsum() {
        use crate::tensor::dim::{D1, D2};

        // 1D cumsum
        let x: Tensor<D1, f32> = Tensor::input([5]);
        let y = x.cumsum(0);
        assert_eq!(y.shape(), vec![5]); // Same shape as input

        // 2D cumsum along axis 0
        let x2: Tensor<D2, f32> = Tensor::input([3, 4]);
        let y2 = x2.cumsum(0);
        assert_eq!(y2.shape(), vec![3, 4]);

        // 2D cumsum along axis 1
        let y3 = x2.cumsum(1);
        assert_eq!(y3.shape(), vec![3, 4]);
    }

    #[test]
    fn test_cumprod() {
        use crate::tensor::dim::{D1, D2};

        // 1D cumprod
        let x: Tensor<D1, f32> = Tensor::input([5]);
        let y = x.cumprod(0);
        assert_eq!(y.shape(), vec![5]);

        // 2D cumprod
        let x2: Tensor<D2, f32> = Tensor::input([3, 4]);
        let y2 = x2.cumprod(1);
        assert_eq!(y2.shape(), vec![3, 4]);
    }

    #[test]
    fn test_cummax() {
        use crate::tensor::dim::{D1, D2};

        // 1D cummax
        let x: Tensor<D1, f32> = Tensor::input([5]);
        let y = x.cummax(0);
        assert_eq!(y.shape(), vec![5]);

        // 2D cummax
        let x2: Tensor<D2, f32> = Tensor::input([3, 4]);
        let y2 = x2.cummax(0);
        assert_eq!(y2.shape(), vec![3, 4]);
    }

    #[test]
    #[should_panic(expected = "Axis 2 out of bounds")]
    fn test_cumsum_invalid_axis() {
        use crate::tensor::dim::D2;

        let x: Tensor<D2, f32> = Tensor::input([3, 4]);
        let _ = x.cumsum(2); // Should panic: axis 2 doesn't exist
    }
}
