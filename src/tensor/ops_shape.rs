use super::{Shape, Tensor, TensorData, TensorOp};

impl Tensor {
    /// Permutes the dimensions of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axes` - A vector specifying the new ordering of dimensions.
    pub fn permute(&self, axes: Vec<usize>) -> Tensor {
        let data = self.0.borrow();
        let new_shape: Shape = axes.iter().map(|&i| data.shape[i]).collect();

        TensorData {
            op: TensorOp::Permute(axes),
            src: vec![self.clone()],
            shape: new_shape,
            dtype: data.dtype.clone(),
            buffer: None,
            grad: None,
            requires_grad: data.requires_grad,
            backend: data.backend.clone(),
        }
        .into()
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        let data = self.0.borrow();
        assert_eq!(
            data.shape.iter().product::<usize>(),
            shape.iter().product::<usize>(),
            "Reshape must not change the total number of elements."
        );

        TensorData {
            op: TensorOp::Reshape(shape.clone()),
            src: vec![self.clone()],
            shape,
            dtype: data.dtype.clone(),
            buffer: None,
            grad: None,
            requires_grad: data.requires_grad,
            backend: data.backend.clone(),
        }
        .into()
    }

    /// Expands the tensor to a new shape by broadcasting its dimensions.
    ///
    /// A dimension can be expanded if it is 1.
    pub fn expand(&self, shape: Vec<usize>) -> Tensor {
        let data = self.0.borrow();
        // Basic validation, more thorough checks are in the graph layer.
        assert_eq!(
            data.shape.len(),
            shape.len(),
            "Expand must have the same number of dimensions."
        );

        TensorData {
            op: TensorOp::Expand(shape.clone()),
            src: vec![self.clone()],
            shape,
            dtype: data.dtype.clone(),
            buffer: None,
            grad: None,
            requires_grad: data.requires_grad,
            backend: data.backend.clone(),
        }
        .into()
    }

    /// Squeezes a dimension of size 1 from the tensor's shape.
    pub fn squeeze(&self, dim: usize) -> Tensor {
        let data = self.0.borrow();
        let mut new_shape = data.shape.clone();
        if new_shape[dim] == 1 {
            new_shape.remove(dim);
        } else {
            // For consistency with frameworks like PyTorch, squeezing a non-1 dimension is a no-op.
        }

        TensorData {
            op: TensorOp::Squeeze(dim),
            src: vec![self.clone()],
            shape: new_shape,
            dtype: data.dtype.clone(),
            buffer: None,
            grad: None,
            requires_grad: data.requires_grad,
            backend: data.backend.clone(),
        }
        .into()
    }

    /// Unsqueezes a new dimension of size 1 into the tensor's shape.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let data = self.0.borrow();
        let mut new_shape = data.shape.clone();
        new_shape.insert(dim, 1);

        TensorData {
            op: TensorOp::Unsqueeze(dim),
            src: vec![self.clone()],
            shape: new_shape,
            dtype: data.dtype.clone(),
            buffer: None,
            grad: None,
            requires_grad: data.requires_grad,
            backend: data.backend.clone(),
        }
        .into()
    }

    /// Slices the tensor along specified ranges for each dimension.
    pub fn slice(&self, args: Vec<(usize, usize)>) -> Tensor {
        let data = self.0.borrow();
        let new_shape: Shape = args.iter().map(|(start, end)| end - start).collect();

        TensorData {
            op: TensorOp::Slice(args),
            src: vec![self.clone()],
            shape: new_shape,
            dtype: data.dtype.clone(),
            buffer: None,
            grad: None,
            requires_grad: data.requires_grad,
            backend: data.backend.clone(),
        }
        .into()
    }
}
