use crate::{
    ast::{AstNode, AstOp, Const, DType},
    graph::{
        lowerer::Lowerer,
        node::{NodeData, NodeId},
        shape::tracker::ShapeTracker,
    },
};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_input(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let tracker = ShapeTracker::new(node_data.shape.clone());
        (
            AstNode::new(AstOp::Block, vec![], DType::Void),
            tracker,
            node_id,
        )
    }

    pub(crate) fn lower_full(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
        value: Const,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut loops = vec![];
        let mut loop_vars = vec![];
        for shape_expr in dst_tracker.shape().iter() {
            let loop_var = self.new_loop_counter();
            loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
        }

        let dst_offset = dst_tracker.offset_expr(&loop_vars);
        let const_node = AstNode::new(AstOp::Const(value), vec![], value.dtype());
        let store_node = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            const_node,
        );

        let final_block = AstNode::build_loops(loops, vec![store_node]);

        (final_block, dst_tracker, node_id)
    }

    pub(crate) fn lower_rand(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut loops = vec![];
        let mut loop_vars = vec![];
        for shape_expr in dst_tracker.shape().iter() {
            let loop_var = self.new_loop_counter();
            loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
        }

        let dst_offset = dst_tracker.offset_expr(&loop_vars);
        let rand_node = AstNode::new(
            AstOp::Mul,
            vec![
                AstNode::call("rand", vec![]).cast(DType::F32),
                AstNode::var("RAND_MAX").cast(DType::F32).recip(),
            ],
            DType::F32,
        );
        let store_node = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            rand_node,
        );

        let final_block = AstNode::build_loops(loops, vec![store_node]);

        (final_block, dst_tracker, node_id)
    }
}
