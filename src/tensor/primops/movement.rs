//! Movement primitive operations
//!
//! - Squeeze: remove dimensions of size 1
//! - Unsqueeze: add dimension of size 1
//! - Repeat: repeat along dimension
//! - Reshape: change shape (same total elements)
//! - Contiguous: ensure contiguous memory layout
//!
//! These operations are generic over TensorDType since they only manipulate shape.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::ops::PadValue;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{Dim, DimDyn, Dimension, Tensor, TensorDType, TensorInner, TensorOp};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

impl<T: TensorDType, D: Dimension> Tensor<T, D> {
    /// Squeeze: remove a specific dimension of size 1 (primop)
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

    /// Unsqueeze: add a dimension of size 1 at the specified position (primop)
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

    /// Reshape to a new static shape (primop)
    ///
    /// Total number of elements must remain the same.
    pub fn reshape<const M: usize>(&self, new_shape: [usize; M]) -> Tensor<T, Dim<M>>
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

    /// Reshape to dynamic shape (primop)
    pub fn reshape_dyn(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
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

    /// Permute tensor dimensions
    ///
    /// # Arguments
    /// * `axes` - New order of dimensions
    pub fn permute(&self, axes: &[usize]) -> Tensor<T, DimDyn> {
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

    /// Transpose the tensor (swap last two dimensions)
    pub fn transpose(&self) -> Tensor<T, DimDyn> {
        assert!(self.ndim() >= 2, "Transpose requires at least 2 dimensions");

        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        let n = axes.len();
        axes.swap(n - 2, n - 1);
        self.permute(&axes)
    }

    /// Expand tensor to a larger shape (broadcast)
    ///
    /// Dimensions of size 1 can be expanded to larger sizes.
    /// The stride for expanded dimensions is set to 0 to enable broadcasting.
    pub fn expand(&self, new_shape: &[usize]) -> Tensor<T, DimDyn> {
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

    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Tensor<T, Dim<1>> {
        self.reshape([self.numel()])
    }

    /// Pad tensor with a specified value - type-safe version
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

        Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Pad tensor with zeros (convenience method for sum reduction) - type-safe version
    pub fn pad_zero(&self, padding: &[(usize, usize)]) -> Tensor<T, D> {
        self.pad(padding, PadValue::Zero)
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
}
