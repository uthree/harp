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

    /// Create a sliding window view for convolution operations.
    ///
    /// This operation adds a new dimension for sliding windows.
    /// For example, [B, C, L] with window_size=K, stride=S becomes [B, C, L', K]
    /// where L' = (L - K) / S + 1.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to create sliding windows
    /// * `window_size` - The size of each window
    /// * `stride` - The stride between windows
    ///
    /// # Example
    /// ```ignore
    /// // For Conv1d: input [B, C_in, L], kernel [C_out, C_in, K]
    /// let windowed = input.sliding_window(2, kernel_size, stride); // → [B, C_in, L', K]
    /// let kernel_reshaped = kernel.unsqueeze(2); // → [C_out, C_in, 1, K]
    /// let product = windowed * kernel_reshaped; // → [B, C_out, C_in, L', K]
    /// let output = product.sum(vec![2, 4]); // → [B, C_out, L']
    /// ```
    pub fn sliding_window<E: Into<ShapeExpr> + Clone>(
        self,
        dim: usize,
        window_size: E,
        stride: E,
    ) -> GraphNode {
        let new_view = self.view.clone().sliding_window(dim, window_size, stride);
        GraphNode::new(GraphOp::View(self.clone()), self.dtype.clone(), new_view)
    }
}
