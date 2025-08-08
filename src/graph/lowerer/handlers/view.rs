use crate::{
    ast::AstNode,
    graph::{
        lowerer::Lowerer,
        node::{NodeData, NodeId},
        shape::{expr::Expr, tracker::ShapeTracker},
    },
};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_permute(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        axes: Vec<usize>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.permute(axes);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(crate) fn lower_squeeze(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        axis: usize,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.squeeze(axis);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(crate) fn lower_unsqueeze(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        axis: usize,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.unsqueeze(axis);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(crate) fn lower_expand(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        new_shape: Vec<Expr>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.expand(new_shape);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(crate) fn lower_slice(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        args: Vec<(Expr, Expr)>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.slice(&args);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(crate) fn lower_unfold1d(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        dim: usize,
        kernel_size: usize,
        stride: usize,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.unfold1d(dim, kernel_size, stride);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(crate) fn lower_unfold2d(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let h_dim = src_tracker.ndim() - 2;
        let w_dim = src_tracker.ndim() - 1;
        let new_tracker = src_tracker.unfold2d(h_dim, w_dim, kernel_size, stride);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(crate) fn lower_reshape(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        new_shape: Vec<Expr>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.reshape(new_shape);
        (src_ast, new_tracker, src_buffer_id)
    }
}
