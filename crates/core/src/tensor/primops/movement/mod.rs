//! Movement primitive operations
//!
//! - Squeeze: remove dimensions of size 1
//! - Unsqueeze: add dimension of size 1
//! - Repeat: repeat along dimension
//! - Reshape: change shape (same total elements)
//! - Contiguous: ensure contiguous memory layout
//! - Pad: add padding around tensor
//! - Slice: extract sub-tensor
//! - Concat: concatenate tensors along an axis
//! - Unfold: sliding window operation (im2col)
//! - Fold: inverse of unfold (col2im)
//!
//! These operations are generic over TensorDType since they only manipulate shape.
//! Gradient tracking is available for FloatDType tensors (f32, f64).

mod backward;
mod fold;
pub mod gather;
pub mod scatter;
mod unfold;

#[cfg(test)]
mod tests;

pub use backward::*;

use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::ops::{InputRef, PadValue};
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    Dim, DimDyn, Dimension, FloatDType, Tensor, TensorDType, TensorInner, TensorOp,
};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

impl<T: TensorDType, D: Dimension> Tensor<T, D> {
    // ========================================================================
    // Internal helper methods for view operations (no gradient tracking)
    // ========================================================================

    /// Internal squeeze implementation without gradient tracking
    pub(crate) fn squeeze_impl(&self, dim: usize) -> Tensor<T, D::Smaller> {
        assert!(
            dim < self.ndim(),
            "Dimension {} out of range for tensor with {} dimensions",
            dim,
            self.ndim()
        );
        assert_eq!(
            self.shape()[dim],
            1,
            "Cannot squeeze dimension {} with size {}",
            dim,
            self.shape()[dim]
        );

        let mut new_shape = self.shape().to_vec();
        new_shape.remove(dim);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let view = view_from_shape(&new_shape);
        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, view, new_shape, T::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Internal unsqueeze implementation without gradient tracking
    pub(crate) fn unsqueeze_impl(&self, dim: usize) -> Tensor<T, D::Larger> {
        assert!(
            dim <= self.ndim(),
            "Dimension {} out of range for tensor with {} dimensions",
            dim,
            self.ndim()
        );

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(dim, 1);

        let view = view_from_shape(&new_shape);
        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, view, new_shape, T::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Internal reshape implementation without gradient tracking
    pub(crate) fn reshape_impl<const M: usize>(&self, new_shape: [usize; M]) -> Tensor<T, Dim<M>>
    where
        Dim<M>: Dimension,
    {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of size {} to shape {:?} (size {})",
            self.numel(),
            new_shape,
            new_numel
        );

        let view = view_from_shape(&new_shape);
        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, view, new_shape.to_vec(), T::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Internal reshape_dyn implementation without gradient tracking
    pub(crate) fn reshape_dyn_impl(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of size {} to shape {:?} (size {})",
            self.numel(),
            new_shape,
            new_numel
        );

        let view = view_from_shape(new_shape);
        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, view, new_shape.to_vec(), T::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Internal permute implementation without gradient tracking
    pub(crate) fn permute_impl(&self, axes: &[usize]) -> Tensor<T, DimDyn> {
        assert_eq!(
            axes.len(),
            self.ndim(),
            "Permutation must have same number of axes as tensor dimensions"
        );

        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape()[i]).collect();
        let new_view = self.inner.view.clone().permute(axes.to_vec());
        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, new_view, new_shape, T::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Internal transpose implementation without gradient tracking
    pub(crate) fn transpose_impl(&self) -> Tensor<T, DimDyn> {
        assert!(self.ndim() >= 2, "Transpose requires at least 2 dimensions");

        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        let n = axes.len();
        axes.swap(n - 2, n - 1);
        self.permute_impl(&axes)
    }

    /// Repeat tensor along each dimension (primop)
    ///
    /// # Arguments
    /// * `repeats` - Number of times to repeat along each dimension
    pub fn repeat(&self, repeats: &[usize]) -> Tensor<T, DimDyn> {
        assert_eq!(
            repeats.len(),
            self.ndim(),
            "Repeats must have same length as tensor dimensions"
        );

        // New shape is old_shape * repeats
        let new_shape: Vec<usize> = self
            .shape()
            .iter()
            .zip(repeats.iter())
            .map(|(&s, &r)| s * r)
            .collect();

        // Create a view with broadcast to the new shape
        let view = view_from_shape(&new_shape);
        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, view, new_shape, T::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Ensure contiguous memory layout (primop)
    ///
    /// Returns a tensor with the same data but guaranteed contiguous memory.
    pub fn contiguous(&self) -> Tensor<T, D> {
        let view = view_from_shape(self.shape());
        let input = self.as_input_ref();
        let inner = TensorInner::new(
            TensorOp::Contiguous { input },
            view,
            self.shape().to_vec(),
            T::DTYPE,
        );
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Internal expand implementation without gradient tracking
    pub(crate) fn expand_impl(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
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

        // Build broadcast strides: stride = 0 for expanded dimensions (size 1 → size N)
        let old_shape = self.shape();
        let new_shape_exprs: Vec<Expr> = new_shape.iter().map(|&s| Expr::from(s as i64)).collect();

        // Get input strides
        let input_strides: Vec<Expr> = match &self.inner.view {
            View::Linear { strides, .. } => strides.clone(),
            View::IndexExpr { .. } => {
                // For IndexExpr, fall back to contiguous strides computation
                let mut strides = vec![Expr::from(1); old_shape.len()];
                for i in (0..old_shape.len() - 1).rev() {
                    strides[i] = Expr::from((old_shape[i + 1..].iter().product::<usize>()) as i64);
                }
                strides
            }
            View::Masked { inner, .. } => {
                // For Masked, use the inner view's strides
                match inner.as_ref() {
                    View::Linear { strides, .. } => strides.clone(),
                    _ => {
                        let mut strides = vec![Expr::from(1); old_shape.len()];
                        for i in (0..old_shape.len() - 1).rev() {
                            strides[i] =
                                Expr::from((old_shape[i + 1..].iter().product::<usize>()) as i64);
                        }
                        strides
                    }
                }
            }
        };

        // Create new strides with broadcast handling
        let mut new_strides = Vec::with_capacity(new_shape.len());
        for (i, (&old_dim, &new_dim)) in old_shape.iter().zip(new_shape.iter()).enumerate() {
            if old_dim == 1 && new_dim > 1 {
                // Broadcast dimension: stride = 0
                new_strides.push(Expr::from(0));
            } else {
                // Keep original stride
                new_strides.push(input_strides[i].clone());
            }
        }

        // Get input offset
        let input_offset = match &self.inner.view {
            View::Linear { offset, .. } => offset.clone(),
            View::IndexExpr { .. } => Expr::from(0),
            View::Masked { inner, .. } => {
                // For Masked, use the inner view's offset
                match inner.as_ref() {
                    View::Linear { offset, .. } => offset.clone(),
                    _ => Expr::from(0),
                }
            }
        };

        let view = View::Linear {
            shape: new_shape_exprs,
            strides: new_strides,
            offset: input_offset,
        };

        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, view, new_shape.to_vec(), T::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Flatten tensor to 1D (no gradient tracking version)
    ///
    /// Note: For FloatDType tensors that need gradient tracking,
    /// use the reshape method instead.
    pub fn flatten(&self) -> Tensor<T, Dim<1>> {
        self.reshape_impl([self.numel()])
    }
}

// ============================================================================
// FloatDType-only operations with gradient tracking
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Pad tensor with a specified value - type-safe version with gradient tracking
    ///
    /// Adds padding to the tensor along each dimension.
    /// The number of dimensions is preserved.
    ///
    /// # Arguments
    /// * `padding` - Slice of (before, after) padding for each dimension.
    ///   Length must match tensor's number of dimensions.
    /// * `value` - The padding value (Zero, One, or NegInf)
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f32, Dim2>::ones([2, 3]);
    /// // Pad with 1 before and 2 after on dim 0, 0 before and 1 after on dim 1
    /// let padded: Tensor<f32, Dim2> = a.pad(&[(1, 2), (0, 1)], PadValue::Zero);
    /// assert_eq!(padded.shape(), &[5, 4]); // [2+1+2, 3+0+1]
    /// ```
    pub fn pad(&self, padding: &[(usize, usize)], value: PadValue) -> Tensor<T, D> {
        assert_eq!(
            padding.len(),
            self.ndim(),
            "Padding length {} must match tensor dimensions {}",
            padding.len(),
            self.ndim()
        );

        // Calculate new shape
        let new_shape: Vec<usize> = self
            .shape()
            .iter()
            .zip(padding.iter())
            .map(|(&dim, &(before, after))| dim + before + after)
            .collect();

        // Convert padding to Expr
        let padding_exprs: Vec<(Expr, Expr)> = padding
            .iter()
            .map(|&(before, after)| (Expr::from(before as i64), Expr::from(after as i64)))
            .collect();

        // Create View::Masked wrapping the input's view (via View::padded)
        let inner_view = self.inner.view.clone();
        let padded_view = View::padded(inner_view, padding_exprs, value);

        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, padded_view, new_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = PadBackward::new(self.clone(), padding.to_vec());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Pad tensor with zeros (convenience method for sum reduction) - type-safe version
    pub fn pad_zero(&self, padding: &[(usize, usize)]) -> Tensor<T, D> {
        self.pad(padding, PadValue::Zero)
    }

    /// Slice: extract a sub-tensor by specifying ranges for each dimension (primop)
    ///
    /// Creates a view into a portion of the tensor. This is a zero-copy operation
    /// that modifies the offset and shape of the view.
    ///
    /// # Arguments
    /// * `ranges` - Slice of (start, end) for each dimension. Must have length equal
    ///   to the number of dimensions.
    ///
    /// # Panics
    /// * If `ranges.len()` doesn't match the tensor's number of dimensions
    /// * If any range is out of bounds
    /// * If any `start >= end`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f32, Dim2>::ones([4, 5]);
    /// let b = a.slice(&[(1, 3), (2, 5)]); // Extract [2, 3] sub-tensor
    /// assert_eq!(b.shape(), &[2, 3]);
    /// ```
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Tensor<T, D> {
        assert_eq!(
            ranges.len(),
            self.ndim(),
            "Slice ranges length {} must match tensor dimensions {}",
            ranges.len(),
            self.ndim()
        );

        let old_shape = self.shape();

        // Validate ranges
        for (i, &(start, end)) in ranges.iter().enumerate() {
            assert!(
                start < end,
                "Slice range start {} must be less than end {} at dimension {}",
                start,
                end,
                i
            );
            assert!(
                end <= old_shape[i],
                "Slice range end {} exceeds dimension size {} at dimension {}",
                end,
                old_shape[i],
                i
            );
        }

        // Calculate new shape
        let new_shape: Vec<usize> = ranges.iter().map(|&(start, end)| end - start).collect();

        // Get strides from current view
        let (input_strides, input_offset) = match &self.inner.view {
            View::Linear {
                strides, offset, ..
            } => (strides.clone(), offset.clone()),
            View::IndexExpr { .. } => {
                // For IndexExpr, use contiguous strides
                let mut strides = vec![Expr::from(1); old_shape.len()];
                for i in (0..old_shape.len() - 1).rev() {
                    strides[i] = Expr::from((old_shape[i + 1..].iter().product::<usize>()) as i64);
                }
                (strides, Expr::from(0))
            }
            View::Masked { inner, .. } => {
                // For Masked, use the inner view's strides
                match inner.as_ref() {
                    View::Linear {
                        strides, offset, ..
                    } => (strides.clone(), offset.clone()),
                    _ => {
                        let mut strides = vec![Expr::from(1); old_shape.len()];
                        for i in (0..old_shape.len() - 1).rev() {
                            strides[i] =
                                Expr::from((old_shape[i + 1..].iter().product::<usize>()) as i64);
                        }
                        (strides, Expr::from(0))
                    }
                }
            }
        };

        // Calculate new offset: input_offset + sum(start[i] * stride[i])
        let mut new_offset = input_offset;
        for (i, &(start, _)) in ranges.iter().enumerate() {
            if start > 0 {
                new_offset += input_strides[i].clone() * Expr::from(start as i64);
            }
        }

        // Create new view with updated shape and offset (strides remain the same)
        let new_shape_exprs: Vec<Expr> = new_shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::Linear {
            shape: new_shape_exprs,
            strides: input_strides,
            offset: new_offset,
        };

        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, view, new_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = SliceBackward::new(self.clone(), ranges.to_vec());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Concatenate multiple tensors along a specified axis (primop)
    ///
    /// All tensors must have the same shape except for the concatenation axis.
    /// Returns a new tensor with the concatenated data.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensor references to concatenate
    /// * `axis` - The axis along which to concatenate
    ///
    /// # Panics
    /// * If `tensors` is empty
    /// * If `axis` is out of bounds
    /// * If tensors have different shapes on non-axis dimensions
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f32, Dim2>::ones([2, 3]);
    /// let b = Tensor::<f32, Dim2>::ones([4, 3]);
    /// let c = Tensor::concat(&[&a, &b], 0); // [6, 3]
    /// ```
    pub fn concat(tensors: &[&Tensor<T, D>], axis: usize) -> Tensor<T, D> {
        assert!(!tensors.is_empty(), "Cannot concatenate empty tensor list");

        let first = tensors[0];
        let ndim = first.ndim();

        assert!(
            axis < ndim,
            "Axis {} is out of bounds for tensor with {} dimensions",
            axis,
            ndim
        );

        // Validate all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                tensor.ndim(),
                ndim,
                "All tensors must have the same number of dimensions. Tensor 0 has {} dims, tensor {} has {} dims",
                ndim,
                i,
                tensor.ndim()
            );

            for (dim, (&size_first, &size_other)) in
                first.shape().iter().zip(tensor.shape().iter()).enumerate()
            {
                if dim != axis {
                    assert_eq!(
                        size_first, size_other,
                        "Tensors must have same shape on non-axis dimensions. Dimension {} mismatch: {} vs {}",
                        dim, size_first, size_other
                    );
                }
            }
        }

        // Calculate output shape
        let mut new_shape = first.shape().to_vec();
        let axis_size: usize = tensors.iter().map(|t| t.shape()[axis]).sum();
        new_shape[axis] = axis_size;

        // Collect inputs
        let inputs: Vec<InputRef> = tensors.iter().map(|t| t.as_input_ref()).collect();

        let view = view_from_shape(&new_shape);
        let inner = TensorInner::new(TensorOp::Concat { inputs, axis }, view, new_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Check if any input uses typed autograd
        let any_requires_grad = tensors.iter().any(|t| t.requires_grad());

        if any_requires_grad {
            let input_tensors: Vec<Tensor<T, D>> = tensors.iter().map(|&t| t.clone()).collect();
            let grad_fn = ConcatBackward::new(input_tensors, axis);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    // ========================================================================
    // View operations with gradient tracking
    // ========================================================================

    /// Squeeze: remove a specific dimension of size 1 (with gradient tracking)
    ///
    /// Removes the dimension at position `dim` which must have size 1.
    /// Returns a tensor with one fewer dimension (`D::Smaller`).
    ///
    /// # Type Safety
    /// - `Dim<N>` → `Dim<N-1>`
    /// - `DimDyn` → `DimDyn`
    ///
    /// # Example
    /// ```ignore
    /// let a: Tensor<f32, Dim3> = Tensor::ones([2, 1, 3]);
    /// let b: Tensor<f32, Dim2> = a.squeeze(1); // Remove dim 1
    /// assert_eq!(b.shape(), &[2, 3]);
    /// ```
    pub fn squeeze(&self, dim: usize) -> Tensor<T, D::Smaller> {
        let result = self.squeeze_impl(dim);

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = SqueezeBackward::<T, D>::new(self.clone(), dim);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Unsqueeze: add a dimension of size 1 at the specified position (with gradient tracking)
    ///
    /// Adds a new dimension of size 1 at position `dim`.
    /// Returns a tensor with one more dimension (`D::Larger`).
    ///
    /// # Type Safety
    /// - `Dim<N>` → `Dim<N+1>`
    /// - `DimDyn` → `DimDyn`
    ///
    /// # Example
    /// ```ignore
    /// let a: Tensor<f32, Dim2> = Tensor::ones([2, 3]);
    /// let b: Tensor<f32, Dim3> = a.unsqueeze(0); // Add dim at position 0
    /// assert_eq!(b.shape(), &[1, 2, 3]);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Tensor<T, D::Larger> {
        let result = self.unsqueeze_impl(dim);

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = UnsqueezeBackward::<T, D>::new(self.clone(), dim);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Reshape to a new static shape (with gradient tracking)
    ///
    /// Total number of elements must remain the same.
    pub fn reshape<const M: usize>(&self, new_shape: [usize; M]) -> Tensor<T, Dim<M>>
    where
        Dim<M>: Dimension,
    {
        let result = self.reshape_impl(new_shape);

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = ReshapeBackward::<T, D, Dim<M>>::new(self.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Reshape to dynamic shape (with gradient tracking)
    pub fn reshape_dyn(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
        let original_shape = self.shape().to_vec();
        let result = self.reshape_dyn_impl(new_shape);

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = ReshapeDynBackward::<T, D>::new(self.clone(), original_shape);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Expand tensor to a larger shape (broadcast) with gradient tracking
    ///
    /// Dimensions of size 1 can be expanded to larger sizes.
    /// The stride for expanded dimensions is set to 0 to enable broadcasting.
    pub fn expand(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
        let original_shape = self.shape().to_vec();
        let result = self.expand_impl(new_shape);

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = ExpandBackward::<T, D>::new(self.clone(), original_shape);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Permute tensor dimensions (with gradient tracking)
    ///
    /// # Arguments
    /// * `axes` - New order of dimensions
    pub fn permute(&self, axes: &[usize]) -> Tensor<T, DimDyn> {
        let result = self.permute_impl(axes);

        // Use typed system if available
        // Note: permute returns DimDyn, so we use PermuteBackward<T, D>
        if self.requires_grad() {
            let grad_fn = PermuteBackward::<T, D>::new(self.clone(), axes.to_vec());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Transpose the tensor (swap last two dimensions) with gradient tracking
    pub fn transpose(&self) -> Tensor<T, DimDyn> {
        let result = self.transpose_impl();

        // Use typed system if available
        if self.requires_grad() {
            let grad_fn = TransposeBackward::<T, D>::new(self.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Interleave: insert zeros at stride intervals along specified axes
    ///
    /// Used for fold operations with stride > 1. For each axis, elements are
    /// spaced at `stride` intervals with zeros in between.
    ///
    /// # Arguments
    /// * `axes` - The axes along which to interleave
    /// * `strides` - The stride for each axis (spacing between elements)
    ///
    /// # Output Shape
    /// For each axis i in axes with size s and stride st:
    /// `new_size[i] = 1 + (s - 1) * st` if s > 0, else 0
    ///
    /// # Example
    /// ```ignore
    /// // [1, 2, 3] with stride=2 on axis 0 -> [1, 0, 2, 0, 3]
    /// let a = Tensor::<f32, Dim1>::from_data(vec![1.0, 2.0, 3.0], [3]);
    /// let b = a.interleave(&[0], &[2]);
    /// assert_eq!(b.shape(), &[5]); // 1 + (3-1) * 2 = 5
    /// ```
    pub fn interleave(&self, axes: &[usize], strides: &[usize]) -> Tensor<T, DimDyn> {
        assert_eq!(
            axes.len(),
            strides.len(),
            "axes and strides must have the same length"
        );

        if axes.is_empty() {
            return self.clone().into_dyn();
        }

        // Validate axes
        for &axis in axes {
            assert!(
                axis < self.ndim(),
                "axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.ndim()
            );
        }

        // Calculate output shape
        let mut new_shape = self.shape().to_vec();
        for (&axis, &stride) in axes.iter().zip(strides.iter()) {
            let s = self.shape()[axis];
            if s > 0 && stride > 1 {
                new_shape[axis] = 1 + (s - 1) * stride;
            }
        }

        // If all strides are 1, no interleaving needed
        if strides.iter().all(|&s| s == 1) {
            return self.reshape_dyn(&new_shape);
        }

        // Build interleaved result using reshape + pad + reshape + slice
        // This approach works for any number of axes
        let mut result = self.clone().into_dyn();

        for (&axis, &stride) in axes.iter().zip(strides.iter()) {
            if stride <= 1 {
                continue;
            }

            let s = result.shape()[axis];
            if s == 0 {
                continue;
            }

            // Step 1: Unsqueeze to add dimension after the axis
            // [... s ...] -> [... s, 1, ...]
            let unsqueezed = result.unsqueeze(axis + 1);

            // Step 2: Pad with zeros on the new dimension
            // [... s, 1, ...] -> [... s, stride, ...]
            let mut padding = vec![(0, 0); unsqueezed.ndim()];
            padding[axis + 1] = (0, stride - 1);
            let padded = unsqueezed.pad(&padding, PadValue::Zero);

            // Step 3: Reshape to merge the two dimensions
            // [... s, stride, ...] -> [... s*stride, ...]
            let mut merged_shape = result.shape().to_vec();
            merged_shape[axis] = s * stride;
            let reshaped = padded.reshape_dyn(&merged_shape);

            // Step 4: Slice to get the correct output size
            // [... s*stride, ...] -> [... 1+(s-1)*stride, ...]
            let target_size = 1 + (s - 1) * stride;
            let mut ranges: Vec<(usize, usize)> =
                reshaped.shape().iter().map(|&dim| (0, dim)).collect();
            ranges[axis] = (0, target_size);
            result = reshaped.slice(&ranges).into_dyn();
        }

        result
    }
}

// Gather with autograd support (FloatDType only)
impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Gather: select elements from input tensor using indices from index tensor
    ///
    /// PyTorch-style gather operation with automatic gradient tracking.
    /// For a 3D input tensor and dim=1:
    /// ```text
    /// output[i][j][k] = input[i][index[i][j][k]][k]
    /// ```
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to gather
    /// * `index` - Index tensor with the same number of dimensions as input.
    ///   All dimensions except `dim` must match the input shape.
    ///
    /// # Output Shape
    /// Same as the index tensor's shape.
    ///
    /// # Example
    /// ```ignore
    /// // 1D gather: output[i] = input[index[i]]
    /// let input = Tensor::<f32, Dim1>::from_data(vec![1.0, 2.0, 3.0, 4.0], [4]);
    /// let index = Tensor::<i64, Dim1>::from_data(vec![2, 0, 3, 1], [4]);
    /// let output = input.gather(0, &index);
    /// // output = [3.0, 1.0, 4.0, 2.0]
    ///
    /// // 2D gather dim=1: output[i][j] = input[i][index[i][j]]
    /// let input = Tensor::<f32, Dim2>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    /// let index = Tensor::<i64, Dim2>::from_data(vec![2, 0, 1, 2], [2, 2]);
    /// let output = input.gather(1, &index);
    /// // output = [[3.0, 1.0], [5.0, 6.0]]
    /// ```
    ///
    /// # Panics
    /// * If `dim >= input.ndim()`
    /// * If index tensor has different number of dimensions
    /// * If shapes don't match on non-dim axes
    ///
    /// # Note
    /// No bounds checking is performed on index values (matches PyTorch behavior).
    /// Gradient is propagated via scatter_add when requires_grad is true.
    pub fn gather(&self, dim: usize, index: &Tensor<i64, D>) -> Tensor<T, D> {
        let ndim = self.ndim();
        assert!(
            dim < ndim,
            "dim {} is out of bounds for tensor with {} dimensions",
            dim,
            ndim
        );
        assert_eq!(
            index.ndim(),
            ndim,
            "index tensor must have same number of dimensions as input ({} vs {})",
            index.ndim(),
            ndim
        );

        // Validate shapes match on non-dim axes
        for (axis, (&s_in, &s_idx)) in self.shape().iter().zip(index.shape()).enumerate() {
            if axis != dim {
                assert_eq!(
                    s_in, s_idx,
                    "input and index shapes must match on non-dim axes. \
                     Axis {}: input={}, index={}",
                    axis, s_in, s_idx
                );
            }
        }

        let output_shape = index.shape().to_vec();
        let input_shape: Vec<usize> = self.shape().to_vec();
        let index_shape: Vec<usize> = index.shape().to_vec();

        // Build the index expression for gather
        let index_expr = gather::build_gather_index_expr(dim, ndim, &input_shape, &index_shape);

        // Create View::IndexExpr
        let shape_exprs: Vec<Expr> = output_shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::IndexExpr {
            shape: shape_exprs,
            index_expr,
        };

        // Create TensorOp::MapReduce with both inputs
        use crate::ast::helper::wildcard;
        let inputs = vec![self.as_input_ref(), index.as_input_ref()];
        let expr = wildcard("0"); // Just pass through the data value
        let op = TensorOp::elementwise(inputs, expr);

        let inner = TensorInner::new(op, view, output_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Add autograd support
        if self.requires_grad() {
            let grad_fn = GatherBackward::new(self.clone(), index.clone(), dim, input_shape);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}
