//! Movement primitive operations
//!
//! - Squeeze: remove dimensions of size 1
//! - Unsqueeze: add dimension of size 1
//! - Repeat: repeat along dimension
//! - Reshape: change shape (same total elements)
//! - Contiguous: ensure contiguous memory layout

use crate::core::DType;
use crate::core::shape::{Expr, View};
use crate::tensor::{Dim, DimDyn, Dimension, Tensor, TensorNode, TensorOp};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

impl<D: Dimension> Tensor<D> {
    /// Squeeze: remove dimensions of size 1 (primop)
    pub fn squeeze(&self) -> Tensor<DimDyn> {
        let new_shape: Vec<usize> = self.shape().iter().filter(|&&s| s != 1).copied().collect();
        if new_shape.is_empty() {
            // Scalar case - keep at least one dimension
            self.reshape_dyn(&[1])
        } else {
            self.reshape_dyn(&new_shape)
        }
    }

    /// Squeeze a specific dimension (primop)
    pub fn squeeze_dim(&self, dim: usize) -> Tensor<DimDyn> {
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

    /// Reshape to a new static shape (primop)
    ///
    /// Total number of elements must remain the same.
    pub fn reshape<const M: usize>(&self, new_shape: [usize; M]) -> Tensor<Dim<M>>
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
        let tensor_node = TensorNode::new(
            TensorOp::View,
            vec![self.clone().into_dyn()],
            view,
            DType::F32,
        );
        Tensor::from_tensor_node(tensor_node, new_shape.to_vec())
    }

    /// Reshape to dynamic shape (primop)
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

        let view = view_from_shape(new_shape);
        let tensor_node = TensorNode::new(
            TensorOp::View,
            vec![self.clone().into_dyn()],
            view,
            DType::F32,
        );
        Tensor::from_tensor_node(tensor_node, new_shape.to_vec())
    }

    /// Repeat tensor along each dimension (primop)
    ///
    /// # Arguments
    /// * `repeats` - Number of times to repeat along each dimension
    pub fn repeat(&self, repeats: &[usize]) -> Tensor<DimDyn> {
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
        let tensor_node = TensorNode::new(
            TensorOp::View,
            vec![self.clone().into_dyn()],
            view,
            DType::F32,
        );
        Tensor::from_tensor_node(tensor_node, new_shape)
    }

    /// Ensure contiguous memory layout (primop)
    ///
    /// Returns a tensor with the same data but guaranteed contiguous memory.
    pub fn contiguous(&self) -> Tensor<D> {
        let view = view_from_shape(self.shape());
        let tensor_node = TensorNode::new(
            TensorOp::Contiguous,
            vec![self.clone().into_dyn()],
            view,
            DType::F32,
        );
        Tensor::from_tensor_node(tensor_node, self.shape().to_vec())
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
        let new_view = self.inner.view.clone().permute(axes.to_vec());
        let tensor_node = TensorNode::new(
            TensorOp::View,
            vec![self.clone().into_dyn()],
            new_view,
            DType::F32,
        );
        Tensor::from_tensor_node(tensor_node, new_shape)
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

        let view = view_from_shape(new_shape);
        let tensor_node = TensorNode::new(
            TensorOp::View,
            vec![self.clone().into_dyn()],
            view,
            DType::F32,
        );
        Tensor::from_tensor_node(tensor_node, new_shape.to_vec())
    }

    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Tensor<Dim<1>> {
        self.reshape([self.numel()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_squeeze() {
        let a = Tensor::<DimDyn>::ones_dyn(&[1, 2, 1, 3]);
        let b = a.squeeze();
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_squeeze_dim() {
        let a = Tensor::<DimDyn>::ones_dyn(&[1, 2, 3]);
        let b = a.squeeze_dim(0);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_unsqueeze() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.unsqueeze(0);
        assert_eq!(b.shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.reshape([6]);
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_reshape_dyn() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.reshape_dyn(&[3, 2]);
        assert_eq!(b.shape(), &[3, 2]);
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
    fn test_contiguous() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.contiguous();
        assert_eq!(b.shape(), &[2, 3]);
    }
}
