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
}
