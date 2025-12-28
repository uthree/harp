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
//!
//! These operations are generic over TensorDType since they only manipulate shape.
//! Gradient tracking is available for FloatDType tensors (f32, f64).

use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::ops::{InputRef, PadValue};
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    Dim, DimDyn, Dimension, FloatDType, GradFn, Tensor, TensorDType, TensorInner, TensorOp,
};

use super::binary::with_grad_fn_generic;

// ============================================================================
// Movement Gradients (type-safe over Dimension)
// ============================================================================

/// Gradient for Pad: y = pad(x, padding)
/// ∂L/∂x = slice(∂L/∂y, padding位置)
pub struct PadBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    padding: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> PadBackward<T, D> {
    pub fn new(input: Tensor<T, D>, padding: Vec<(usize, usize)>) -> Self {
        Self { input, padding }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T> for PadBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // padding から slice範囲を計算: (before, before + original_dim)
        let ranges: Vec<(usize, usize)> = self
            .padding
            .iter()
            .zip(self.input.shape())
            .map(|(&(before, _), &dim)| (before, before + dim))
            .collect();
        vec![grad_output.slice(&ranges)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone().into_dyn()]
    }

    fn name(&self) -> &'static str {
        "PadBackward"
    }
}

/// Gradient for Slice: y = slice(x, ranges)
/// ∂L/∂x = pad_zero(∂L/∂y, 適切なpadding)
pub struct SliceBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    ranges: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> SliceBackward<T, D> {
    pub fn new(input: Tensor<T, D>, ranges: Vec<(usize, usize)>) -> Self {
        Self { input, ranges }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T> for SliceBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // ranges から padding を計算: (start, original_dim - end)
        let padding: Vec<(usize, usize)> = self
            .ranges
            .iter()
            .zip(self.input.shape())
            .map(|(&(start, end), &dim)| (start, dim - end))
            .collect();
        vec![grad_output.pad_zero(&padding)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone().into_dyn()]
    }

    fn name(&self) -> &'static str {
        "SliceBackward"
    }
}

/// Gradient for Concat: y = concat([a, b, ...], axis)
/// ∂L/∂a, ∂L/∂b, ... = split(∂L/∂y, axis, sizes)
pub struct ConcatBackward<T: FloatDType, D: Dimension> {
    inputs: Vec<Tensor<T, D>>,
    axis: usize,
}

// ============================================================================
// View Operation Gradients
// ============================================================================

/// Gradient for Squeeze: y = squeeze(x, dim)
/// ∂L/∂x = unsqueeze(∂L/∂y, dim)
pub struct SqueezeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    dim: usize,
}

impl<T: FloatDType> SqueezeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, dim: usize) -> Self {
        Self { input, dim }
    }
}

impl<T: FloatDType> GradFn<T> for SqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // unsqueeze the gradient back at the squeezed dimension
        vec![grad_output.unsqueeze(self.dim)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SqueezeBackward"
    }
}

/// Gradient for Unsqueeze: y = unsqueeze(x, dim)
/// ∂L/∂x = squeeze(∂L/∂y, dim)
pub struct UnsqueezeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    dim: usize,
}

impl<T: FloatDType> UnsqueezeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, dim: usize) -> Self {
        Self { input, dim }
    }
}

impl<T: FloatDType> GradFn<T> for UnsqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // squeeze the gradient back at the unsqueezed dimension
        vec![grad_output.squeeze(self.dim)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }
}

/// Gradient for Reshape: y = reshape(x, new_shape)
/// ∂L/∂x = reshape(∂L/∂y, original_shape)
pub struct ReshapeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType> ReshapeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

impl<T: FloatDType> GradFn<T> for ReshapeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // reshape gradient back to original shape
        vec![grad_output.reshape_dyn(&self.original_shape)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

/// Gradient for Expand: y = expand(x, new_shape)
/// ∂L/∂x = sum(∂L/∂y) over broadcast dimensions
pub struct ExpandBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType> ExpandBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

impl<T: FloatDType> GradFn<T> for ExpandBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Sum gradients over dimensions that were broadcast
        // A dimension was broadcast if original_shape[i] == 1 and grad_output.shape()[i] > 1
        let grad_shape = grad_output.shape();
        let mut result = grad_output.clone();

        // Find axes to reduce (where original was 1 but grad is larger)
        // Process in reverse order to preserve indices
        for i in (0..self.original_shape.len()).rev() {
            if self.original_shape[i] == 1 && grad_shape[i] > 1 {
                // Sum over this dimension (keepdim=true by using sum then unsqueeze)
                result = result.sum(i).unsqueeze(i);
            }
        }

        vec![result]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}

/// Gradient for Permute: y = permute(x, axes)
/// ∂L/∂x = permute(∂L/∂y, inverse_axes)
pub struct PermuteBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    axes: Vec<usize>,
}

impl<T: FloatDType> PermuteBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, axes: Vec<usize>) -> Self {
        Self { input, axes }
    }
}

impl<T: FloatDType> GradFn<T> for PermuteBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Compute inverse permutation
        let mut inverse_axes = vec![0; self.axes.len()];
        for (i, &axis) in self.axes.iter().enumerate() {
            inverse_axes[axis] = i;
        }
        vec![grad_output.permute(&inverse_axes)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "PermuteBackward"
    }
}

/// Gradient for Transpose: y = transpose(x)
/// ∂L/∂x = transpose(∂L/∂y)
pub struct TransposeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
}

impl<T: FloatDType> TransposeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType> GradFn<T> for TransposeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Transpose is its own inverse
        vec![grad_output.transpose()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

impl<T: FloatDType, D: Dimension> ConcatBackward<T, D> {
    pub fn new(inputs: Vec<Tensor<T, D>>, axis: usize) -> Self {
        Self { inputs, axis }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T> for ConcatBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        let mut grads = Vec::new();
        let mut offset = 0;
        for input in &self.inputs {
            let size = input.shape()[self.axis];
            let ranges: Vec<(usize, usize)> = grad_output
                .shape()
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == self.axis {
                        (offset, offset + size)
                    } else {
                        (0, dim)
                    }
                })
                .collect();
            grads.push(grad_output.slice(&ranges));
            offset += size;
        }
        grads
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        self.inputs.iter().map(|t| t.clone().into_dyn()).collect()
    }

    fn name(&self) -> &'static str {
        "ConcatBackward"
    }
}

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
            View::Padded { inner, .. } => {
                // For Padded, use the inner view's strides
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
            View::Padded { inner, .. } => {
                // For Padded, use the inner view's offset
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

        // Create View::Padded wrapping the input's view
        let inner_view = self.inner.view.clone();
        let padded_view = View::padded(inner_view, padding_exprs, value);

        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, padded_view, new_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Register gradient if input requires grad
        let grad_fn = if self.requires_grad() {
            Some(Arc::new(PadBackward::new(self.clone(), padding.to_vec())) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
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
            View::Padded { inner, .. } => {
                // For Padded, use the inner view's strides
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
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Register gradient if input requires grad
        let grad_fn = if self.requires_grad() {
            Some(Arc::new(SliceBackward::new(self.clone(), ranges.to_vec())) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
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
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Register gradient if any input requires grad
        let any_requires_grad = tensors.iter().any(|t| t.requires_grad());
        let grad_fn = if any_requires_grad {
            let input_tensors: Vec<Tensor<T, D>> = tensors.iter().map(|&t| t.clone()).collect();
            Some(Arc::new(ConcatBackward::new(input_tensors, axis)) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
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

        let grad_fn = if self.requires_grad() {
            Some(Arc::new(SqueezeBackward::new(self.clone().into_dyn(), dim)) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
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

        let grad_fn = if self.requires_grad() {
            Some(
                Arc::new(UnsqueezeBackward::new(self.clone().into_dyn(), dim))
                    as Arc<dyn GradFn<T>>,
            )
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }

    /// Reshape to a new static shape (with gradient tracking)
    ///
    /// Total number of elements must remain the same.
    pub fn reshape<const M: usize>(&self, new_shape: [usize; M]) -> Tensor<T, Dim<M>>
    where
        Dim<M>: Dimension,
    {
        let original_shape = self.shape().to_vec();
        let result = self.reshape_impl(new_shape);

        let grad_fn = if self.requires_grad() {
            Some(Arc::new(ReshapeBackward::new(
                self.clone().into_dyn(),
                original_shape,
            )) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }

    /// Reshape to dynamic shape (with gradient tracking)
    pub fn reshape_dyn(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
        let original_shape = self.shape().to_vec();
        let result = self.reshape_dyn_impl(new_shape);

        let grad_fn = if self.requires_grad() {
            Some(Arc::new(ReshapeBackward::new(
                self.clone().into_dyn(),
                original_shape,
            )) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }

    /// Expand tensor to a larger shape (broadcast) with gradient tracking
    ///
    /// Dimensions of size 1 can be expanded to larger sizes.
    /// The stride for expanded dimensions is set to 0 to enable broadcasting.
    pub fn expand(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
        let original_shape = self.shape().to_vec();
        let result = self.expand_impl(new_shape);

        let grad_fn = if self.requires_grad() {
            Some(
                Arc::new(ExpandBackward::new(self.clone().into_dyn(), original_shape))
                    as Arc<dyn GradFn<T>>,
            )
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }

    /// Permute tensor dimensions (with gradient tracking)
    ///
    /// # Arguments
    /// * `axes` - New order of dimensions
    pub fn permute(&self, axes: &[usize]) -> Tensor<T, DimDyn> {
        let result = self.permute_impl(axes);

        let grad_fn = if self.requires_grad() {
            Some(
                Arc::new(PermuteBackward::new(self.clone().into_dyn(), axes.to_vec()))
                    as Arc<dyn GradFn<T>>,
            )
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }

    /// Transpose the tensor (swap last two dimensions) with gradient tracking
    pub fn transpose(&self) -> Tensor<T, DimDyn> {
        let result = self.transpose_impl();

        let grad_fn = if self.requires_grad() {
            Some(Arc::new(TransposeBackward::new(self.clone().into_dyn())) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_squeeze() {
        let a = Tensor::<f32, DimDyn>::ones_dyn(&[1, 2, 3]);
        let b = a.squeeze(0);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_unsqueeze() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.unsqueeze(0);
        assert_eq!(b.shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.reshape([6]);
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_reshape_dyn() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.reshape_dyn(&[3, 2]);
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_permute() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.permute(&[1, 0]);
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.transpose();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_expand() {
        let a = Tensor::<f32, Dim2>::ones([1, 3]);
        let b = a.expand(&[4, 3]);
        assert_eq!(b.shape(), &[4, 3]);
    }

    #[test]
    fn test_flatten() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.flatten();
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_contiguous() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.contiguous();
        assert_eq!(b.shape(), &[2, 3]);
    }

    // f64 tests
    #[test]
    fn test_unsqueeze_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = a.unsqueeze(0);
        assert_eq!(b.shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_reshape_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = a.reshape([6]);
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_permute_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = a.permute(&[1, 0]);
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = a.transpose();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_expand_f64() {
        let a = Tensor::<f64, Dim2>::ones([1, 3]);
        let b = a.expand(&[4, 3]);
        assert_eq!(b.shape(), &[4, 3]);
    }

    #[test]
    fn test_flatten_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = a.flatten();
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_contiguous_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = a.contiguous();
        assert_eq!(b.shape(), &[2, 3]);
    }

    // Pad tests
    #[test]
    fn test_pad_shape() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.pad(&[(1, 2), (0, 1)], PadValue::Zero);
        assert_eq!(b.shape(), &[5, 4]); // [2+1+2, 3+0+1]
    }

    #[test]
    fn test_pad_zero_shape() {
        let a = Tensor::<f32, Dim2>::ones([3, 4]);
        let b = a.pad_zero(&[(2, 1), (1, 3)]);
        assert_eq!(b.shape(), &[6, 8]); // [3+2+1, 4+1+3]
    }

    #[test]
    fn test_pad_no_padding() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.pad(&[(0, 0), (0, 0)], PadValue::Zero);
        assert_eq!(b.shape(), &[2, 3]); // No change
    }

    #[test]
    fn test_pad_1d() {
        let a = Tensor::<f32, crate::tensor::Dim1>::ones([5]);
        let b = a.pad(&[(2, 3)], PadValue::One);
        assert_eq!(b.shape(), &[10]); // 5+2+3
    }

    // Type-safe dimension tests
    #[test]
    fn test_squeeze_type_safe() {
        use crate::tensor::{Dim1, Dim3};

        // Dim3 -> Dim2 (squeeze one dimension)
        let a = Tensor::<f32, Dim3>::ones([2, 1, 3]);
        let b: Tensor<f32, Dim2> = a.squeeze(1);
        assert_eq!(b.shape(), &[2, 3]);

        // Dim2 -> Dim1 (squeeze one dimension)
        let c = Tensor::<f32, Dim2>::ones([1, 5]);
        let d: Tensor<f32, Dim1> = c.squeeze(0);
        assert_eq!(d.shape(), &[5]);
    }

    #[test]
    fn test_unsqueeze_type_safe() {
        use crate::tensor::{Dim1, Dim3};

        // Dim2 -> Dim3 (unsqueeze adds one dimension)
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b: Tensor<f32, Dim3> = a.unsqueeze(0);
        assert_eq!(b.shape(), &[1, 2, 3]);

        // Dim1 -> Dim2 (unsqueeze adds one dimension)
        let c = Tensor::<f32, Dim1>::ones([5]);
        let d: Tensor<f32, Dim2> = c.unsqueeze(1);
        assert_eq!(d.shape(), &[5, 1]);
    }

    #[test]
    fn test_pad_type_safe() {
        use crate::tensor::Dim3;

        // Dim2 -> Dim2 (pad preserves dimension)
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b: Tensor<f32, Dim2> = a.pad(&[(1, 1), (2, 2)], PadValue::Zero);
        assert_eq!(b.shape(), &[4, 7]);

        // Dim3 -> Dim3 (pad preserves dimension)
        let c = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        let d: Tensor<f32, Dim3> = c.pad_zero(&[(0, 1), (1, 0), (2, 2)]);
        assert_eq!(d.shape(), &[3, 4, 8]);
    }

    #[test]
    fn test_chained_squeeze_unsqueeze() {
        use crate::tensor::{Dim1, Dim3};

        // Dim2 -> Dim3 -> Dim2
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b: Tensor<f32, Dim3> = a.unsqueeze(0);
        let c: Tensor<f32, Dim2> = b.squeeze(0);
        assert_eq!(c.shape(), &[2, 3]);

        // Dim1 -> Dim2 -> Dim1
        let d = Tensor::<f32, Dim1>::ones([5]);
        let e: Tensor<f32, Dim2> = d.unsqueeze(0);
        let f: Tensor<f32, Dim1> = e.squeeze(0);
        assert_eq!(f.shape(), &[5]);
    }

    // Slice tests
    #[test]
    fn test_slice_basic() {
        let a = Tensor::<f32, Dim2>::ones([4, 5]);
        let b = a.slice(&[(1, 3), (2, 5)]);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_slice_full_range() {
        let a = Tensor::<f32, Dim2>::ones([4, 5]);
        let b = a.slice(&[(0, 4), (0, 5)]);
        assert_eq!(b.shape(), &[4, 5]);
    }

    #[test]
    fn test_slice_single_element() {
        use crate::tensor::Dim1;
        let a = Tensor::<f32, Dim1>::ones([10]);
        let b = a.slice(&[(5, 6)]);
        assert_eq!(b.shape(), &[1]);
    }

    #[test]
    fn test_slice_3d() {
        use crate::tensor::Dim3;
        let a = Tensor::<f32, Dim3>::ones([4, 5, 6]);
        let b = a.slice(&[(1, 3), (0, 5), (2, 4)]);
        assert_eq!(b.shape(), &[2, 5, 2]);
    }

    #[test]
    fn test_slice_type_safe() {
        // Slice preserves dimension type
        let a = Tensor::<f32, Dim2>::ones([4, 5]);
        let b: Tensor<f32, Dim2> = a.slice(&[(1, 3), (2, 5)]);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_slice_f64() {
        let a = Tensor::<f64, Dim2>::ones([4, 5]);
        let b = a.slice(&[(0, 2), (1, 4)]);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "must be less than end")]
    fn test_slice_invalid_range() {
        let a = Tensor::<f32, Dim2>::ones([4, 5]);
        let _ = a.slice(&[(3, 1), (0, 5)]); // start > end
    }

    #[test]
    #[should_panic(expected = "exceeds dimension size")]
    fn test_slice_out_of_bounds() {
        let a = Tensor::<f32, Dim2>::ones([4, 5]);
        let _ = a.slice(&[(0, 4), (0, 10)]); // end > dim size
    }

    // Concat tests
    #[test]
    fn test_concat_axis0() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([4, 3]);
        let c = Tensor::concat(&[&a, &b], 0);
        assert_eq!(c.shape(), &[6, 3]);
    }

    #[test]
    fn test_concat_axis1() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 5]);
        let c = Tensor::concat(&[&a, &b], 1);
        assert_eq!(c.shape(), &[2, 8]);
    }

    #[test]
    fn test_concat_multiple() {
        let a = Tensor::<f32, Dim2>::ones([1, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = Tensor::<f32, Dim2>::ones([3, 3]);
        let d = Tensor::concat(&[&a, &b, &c], 0);
        assert_eq!(d.shape(), &[6, 3]);
    }

    #[test]
    fn test_concat_3d() {
        use crate::tensor::Dim3;
        let a = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        let b = Tensor::<f32, Dim3>::ones([2, 5, 4]);
        let c = Tensor::concat(&[&a, &b], 1);
        assert_eq!(c.shape(), &[2, 8, 4]);
    }

    #[test]
    fn test_concat_type_safe() {
        // Concat preserves dimension type
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([4, 3]);
        let c: Tensor<f32, Dim2> = Tensor::concat(&[&a, &b], 0);
        assert_eq!(c.shape(), &[6, 3]);
    }

    #[test]
    fn test_concat_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = Tensor::<f64, Dim2>::ones([4, 3]);
        let c = Tensor::concat(&[&a, &b], 0);
        assert_eq!(c.shape(), &[6, 3]);
    }

    #[test]
    #[should_panic(expected = "Cannot concatenate empty")]
    fn test_concat_empty() {
        let empty: &[&Tensor<f32, Dim2>] = &[];
        let _ = Tensor::concat(empty, 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_concat_invalid_axis() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([4, 3]);
        let _ = Tensor::concat(&[&a, &b], 5);
    }

    #[test]
    #[should_panic(expected = "same shape on non-axis")]
    fn test_concat_shape_mismatch() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([4, 5]); // Different non-axis dimension
        let _ = Tensor::concat(&[&a, &b], 0);
    }

    // ========================================================================
    // Gradient tests for pad, slice, concat
    // ========================================================================

    #[test]
    fn test_pad_backward_shape() {
        // Test that PadBackward produces correct gradient shape
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let padded = a.pad(&[(1, 2), (0, 1)], PadValue::Zero);

        assert!(padded.requires_grad());
        assert_eq!(padded.shape(), &[5, 4]); // [2+1+2, 3+0+1]

        // Backward
        padded.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
    }

    #[test]
    fn test_pad_backward_f64() {
        let a = Tensor::<f64, Dim2>::ones([3, 4]).set_requires_grad(true);
        let padded = a.pad_zero(&[(2, 1), (1, 3)]);

        assert!(padded.requires_grad());
        padded.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[3, 4]);
    }

    #[test]
    fn test_slice_backward_shape() {
        // Test that SliceBackward produces correct gradient shape
        let a = Tensor::<f32, Dim2>::ones([4, 5]).set_requires_grad(true);
        let sliced = a.slice(&[(1, 3), (2, 5)]);

        assert!(sliced.requires_grad());
        assert_eq!(sliced.shape(), &[2, 3]);

        // Backward
        sliced.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[4, 5]); // Same as original input
    }

    #[test]
    fn test_slice_backward_f64() {
        let a = Tensor::<f64, Dim2>::ones([6, 8]).set_requires_grad(true);
        let sliced = a.slice(&[(1, 4), (2, 6)]);

        assert!(sliced.requires_grad());
        sliced.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[6, 8]);
    }

    #[test]
    fn test_concat_backward_shape() {
        // Test that ConcatBackward produces correct gradient shapes
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = Tensor::<f32, Dim2>::ones([4, 3]).set_requires_grad(true);
        let c = Tensor::concat(&[&a, &b], 0);

        assert!(c.requires_grad());
        assert_eq!(c.shape(), &[6, 3]);

        // Backward
        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]); // Same as original a
        assert_eq!(grad_b.shape(), &[4, 3]); // Same as original b
    }

    #[test]
    fn test_concat_backward_axis1() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = Tensor::<f32, Dim2>::ones([2, 5]).set_requires_grad(true);
        let c = Tensor::concat(&[&a, &b], 1);

        assert_eq!(c.shape(), &[2, 8]);
        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[2, 5]);
    }

    #[test]
    fn test_concat_backward_multiple() {
        let a = Tensor::<f32, Dim2>::ones([1, 3]).set_requires_grad(true);
        let b = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let c = Tensor::<f32, Dim2>::ones([3, 3]).set_requires_grad(true);
        let d = Tensor::concat(&[&a, &b, &c], 0);

        assert_eq!(d.shape(), &[6, 3]);
        d.backward();

        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");
        let grad_c = c.grad().expect("c should have gradient");
        assert_eq!(grad_a.shape(), &[1, 3]);
        assert_eq!(grad_b.shape(), &[2, 3]);
        assert_eq!(grad_c.shape(), &[3, 3]);
    }

    #[test]
    fn test_concat_backward_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = Tensor::<f64, Dim2>::ones([4, 3]).set_requires_grad(true);
        let c = Tensor::concat(&[&a, &b], 0);

        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[4, 3]);
    }

    #[test]
    fn test_pad_slice_inverse_backward() {
        // pad and slice are inverses for gradients
        // If we pad then slice back, gradient should flow correctly
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let padded = a.pad_zero(&[(1, 1), (2, 2)]);
        let sliced = padded.slice(&[(1, 3), (2, 5)]); // Slice back to original shape

        assert_eq!(sliced.shape(), &[2, 3]);
        sliced.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_no_grad_propagation() {
        // When input doesn't require grad, output shouldn't either
        let a = Tensor::<f32, Dim2>::ones([2, 3]); // No requires_grad
        let padded = a.pad_zero(&[(1, 1), (0, 0)]);
        assert!(!padded.requires_grad());

        let b = Tensor::<f32, Dim2>::ones([4, 5]);
        let sliced = b.slice(&[(1, 3), (2, 5)]);
        assert!(!sliced.requires_grad());

        let c = Tensor::<f32, Dim2>::ones([2, 3]);
        let d = Tensor::<f32, Dim2>::ones([4, 3]);
        let concated = Tensor::concat(&[&c, &d], 0);
        assert!(!concated.requires_grad());
    }

    // ========================================================================
    // Gradient tests for view operations (squeeze, unsqueeze, reshape, expand, etc.)
    // ========================================================================

    #[test]
    fn test_squeeze_backward_shape() {
        use crate::tensor::Dim3;
        let a = Tensor::<f32, Dim3>::ones([2, 1, 3]).set_requires_grad(true);
        let squeezed = a.squeeze(1);

        assert!(squeezed.requires_grad());
        assert_eq!(squeezed.shape(), &[2, 3]);

        squeezed.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 1, 3]); // Same as original input
    }

    #[test]
    fn test_unsqueeze_backward_shape() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let unsqueezed = a.unsqueeze(1);

        assert!(unsqueezed.requires_grad());
        assert_eq!(unsqueezed.shape(), &[2, 1, 3]);

        unsqueezed.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
    }

    #[test]
    fn test_reshape_backward_shape() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let reshaped = a.reshape([3, 2]);

        assert!(reshaped.requires_grad());
        assert_eq!(reshaped.shape(), &[3, 2]);

        reshaped.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
    }

    #[test]
    fn test_reshape_dyn_backward_shape() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let reshaped = a.reshape_dyn(&[6]);

        assert!(reshaped.requires_grad());
        assert_eq!(reshaped.shape(), &[6]);

        reshaped.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
    }

    #[test]
    fn test_expand_backward_shape() {
        let a = Tensor::<f32, Dim2>::ones([1, 3]).set_requires_grad(true);
        let expanded = a.expand(&[4, 3]);

        assert!(expanded.requires_grad());
        assert_eq!(expanded.shape(), &[4, 3]);

        expanded.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[1, 3]); // Same as original input
    }

    #[test]
    fn test_permute_backward_shape() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let permuted = a.permute(&[1, 0]);

        assert!(permuted.requires_grad());
        assert_eq!(permuted.shape(), &[3, 2]);

        permuted.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
    }

    #[test]
    fn test_transpose_backward_shape() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let transposed = a.transpose();

        assert!(transposed.requires_grad());
        assert_eq!(transposed.shape(), &[3, 2]);

        transposed.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
    }

    #[test]
    fn test_chained_view_ops_backward() {
        // Test that gradients flow through chained view operations
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = a.unsqueeze(0); // [1, 2, 3]
        let c = b.expand(&[4, 2, 3]); // [4, 2, 3]
        let d = c.reshape_dyn(&[8, 3]); // [8, 3]

        assert!(d.requires_grad());
        d.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_squeeze_unsqueeze_inverse_backward() {
        // squeeze and unsqueeze are inverses
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let unsqueezed = a.unsqueeze(1); // [2, 1, 3]
        let squeezed = unsqueezed.squeeze(1); // [2, 3]

        assert_eq!(squeezed.shape(), &[2, 3]);
        squeezed.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_view_ops_no_grad_propagation() {
        // When input doesn't require grad, output shouldn't either
        let a = Tensor::<f32, Dim2>::ones([2, 3]); // No requires_grad

        let squeezed = a.unsqueeze(0).squeeze(0);
        assert!(!squeezed.requires_grad());

        let reshaped = a.reshape([6]);
        assert!(!reshaped.requires_grad());

        let expanded = Tensor::<f32, Dim2>::ones([1, 3]).expand(&[4, 3]);
        assert!(!expanded.requires_grad());

        let permuted = a.permute(&[1, 0]);
        assert!(!permuted.requires_grad());

        let transposed = a.transpose();
        assert!(!transposed.requires_grad());
    }
}
