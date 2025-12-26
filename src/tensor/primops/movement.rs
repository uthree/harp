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

use crate::tensor::shape::{Expr, View};
use crate::tensor::{Dim, DimDyn, Dimension, Tensor, TensorDType, TensorInner, TensorOp};

// ============================================================================
// Helper for type conversion
// ============================================================================

/// Convert any Tensor<T, D> to Tensor<f32, DimDyn> for graph operations.
fn to_graph_ref<T: TensorDType, D: Dimension>(tensor: &Tensor<T, D>) -> Tensor<f32, DimDyn> {
    Tensor {
        inner: tensor.inner.clone(),
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

impl<T: TensorDType, D: Dimension> Tensor<T, D> {
    /// Squeeze: remove dimensions of size 1 (primop)
    pub fn squeeze(&self) -> Tensor<T, DimDyn> {
        let new_shape: Vec<usize> = self.shape().iter().filter(|&&s| s != 1).copied().collect();
        if new_shape.is_empty() {
            // Scalar case - keep at least one dimension
            self.reshape_dyn(&[1])
        } else {
            self.reshape_dyn(&new_shape)
        }
    }

    /// Squeeze a specific dimension (primop)
    pub fn squeeze_dim(&self, dim: usize) -> Tensor<T, DimDyn> {
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
        self.reshape_dyn(&new_shape)
    }

    /// Unsqueeze: add a dimension of size 1 at the specified position (primop)
    pub fn unsqueeze(&self, dim: usize) -> Tensor<T, DimDyn> {
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
        let input = Arc::new(to_graph_ref(self));
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
        let input = Arc::new(to_graph_ref(self));
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
        let input = Arc::new(to_graph_ref(self));
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
        let input = Arc::new(to_graph_ref(self));
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
        let input = Arc::new(to_graph_ref(self));
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
        };

        let view = View::Linear {
            shape: new_shape_exprs,
            strides: new_strides,
            offset: input_offset,
        };

        let input = Arc::new(to_graph_ref(self));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_squeeze() {
        let a = Tensor::<f32, DimDyn>::ones_dyn(&[1, 2, 1, 3]);
        let b = a.squeeze();
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_squeeze_dim() {
        let a = Tensor::<f32, DimDyn>::ones_dyn(&[1, 2, 3]);
        let b = a.squeeze_dim(0);
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
    fn test_squeeze_f64() {
        let a = Tensor::<f64, DimDyn>::ones_dyn(&[1, 2, 1, 3]);
        let b = a.squeeze();
        assert_eq!(b.shape(), &[2, 3]);
    }

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
}
