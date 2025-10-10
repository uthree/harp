use crate::graph::shape::Expr as ShapeExpr;
use crate::graph::{GraphNode, GraphOp};

impl GraphNode {
    pub fn unsqueeze(self, axis: usize) -> GraphNode {
        let new_view = self.view.clone().unsqueeze(axis);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }

    pub fn squeeze(self, axis: usize) -> GraphNode {
        let new_view = self.view.clone().squeeze(axis);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }

    pub fn expand(self, new_shape: Vec<ShapeExpr>) -> GraphNode {
        let new_view = self.view.clone().expand(new_shape);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }

    pub fn permute(self, axes: Vec<usize>) -> GraphNode {
        let new_view = self.view.clone().permute(axes);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }

    pub fn flip(self, axis: usize) -> GraphNode {
        let new_view = self.view.clone().flip(axis);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }

    /// Convert to contiguous memory layout
    /// If the view is already contiguous, this creates a copy operation anyway
    /// (optimizer may eliminate it if not needed)
    pub fn contiguous(self) -> GraphNode {
        let shape = self.view.shape().to_vec();
        let new_view = crate::graph::shape::view::View::new_contiguous(shape);
        GraphNode::new(
            GraphOp::Contiguous(self.clone()),
            self.dtype.clone(),
            new_view,
        )
    }

    /// Create a view with custom strides (low-level operation).
    ///
    /// See `View::as_strided` for more details.
    pub fn as_strided(self, new_shape: Vec<ShapeExpr>, new_strides: Vec<ShapeExpr>) -> GraphNode {
        let new_view = self.view.clone().as_strided(new_shape, new_strides);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }

    /// Unfold operation: extract sliding local blocks from a tensor.
    ///
    /// This operation adds a new dimension for sliding windows.
    /// For example, [B, C, L] with window_size=K, stride=S, dilation=D becomes [B, C, L', K]
    /// where L' = (L - D*(K-1) - 1) / S + 1.
    ///
    /// This is the inverse of the `fold` operation.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to create sliding windows
    /// * `window_size` - The size of each window
    /// * `stride` - The stride between windows
    /// * `dilation` - The dilation (spacing between kernel elements)
    ///
    /// # Example
    /// ```ignore
    /// // For Conv1d: input [B, C_in, L], kernel [C_out, C_in, K]
    /// let unfolded = input.unfold(2, kernel_size, stride, dilation); // → [B, C_in, L', K]
    /// let kernel_reshaped = kernel.unsqueeze(2); // → [C_out, C_in, 1, K]
    /// let product = unfolded * kernel_reshaped; // → [B, C_out, C_in, L', K]
    /// let output = product.sum(vec![2, 4]); // → [B, C_out, L']
    /// ```
    pub fn unfold<E: Into<ShapeExpr> + Clone>(
        self,
        dim: usize,
        window_size: E,
        stride: E,
        dilation: E,
    ) -> GraphNode {
        let new_view = self.view.clone().unfold(dim, window_size, stride, dilation);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }

    /// Fold operation: combines overlapping blocks into a single tensor (similar to col2im).
    ///
    /// This is the inverse operation of `unfold`. It takes a tensor with an extra dimension
    /// representing sliding windows and combines them back into the original shape.
    /// When windows overlap, the values are summed.
    ///
    /// For example, input [B, C, L', K] with window_size=K, stride=S becomes [B, C, L]
    /// where L = (L' - 1) * S + K.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which the windows were extracted
    /// * `window_size` - The size of each window
    /// * `stride` - The stride between windows
    /// * `output_size` - The size of the output dimension (L in the example above)
    ///
    /// # Example
    /// ```ignore
    /// // Inverse of unfold operation
    /// let unfolded = input.unfold(2, 3, 1); // [B, C, L] → [B, C, L', 3]
    /// let folded = unfolded.fold(2, 3, 1, original_L); // [B, C, L', 3] → [B, C, L]
    /// ```
    pub fn fold(
        self,
        dim: usize,
        window_size: usize,
        stride: usize,
        dilation: usize,
        output_size: usize,
    ) -> GraphNode {
        // Input shape: [..., L', K] where the last dimension is the window dimension
        // Output shape: [..., L] where L = output_size

        let input_shape = self.view.shape();
        let ndim = input_shape.len();

        assert!(
            dim < ndim - 1,
            "dim must be less than ndim - 1 (last dimension is window dimension)"
        );
        assert_eq!(
            input_shape[ndim - 1],
            ShapeExpr::from(window_size as isize),
            "last dimension must match window_size"
        );

        // Calculate output shape: replace dim with output_size and remove last dimension
        let mut output_shape = input_shape.to_vec();
        output_shape[dim] = ShapeExpr::from(output_size as isize);
        output_shape.pop(); // Remove window dimension

        let result_view = crate::graph::shape::view::View::new_contiguous(output_shape);

        GraphNode::new(
            GraphOp::Fold(dim, window_size, stride, dilation, self.clone()),
            self.dtype.clone(),
            result_view,
        )
    }
}
