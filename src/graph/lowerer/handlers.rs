//! This module contains the handler functions for lowering different `GraphOp` variants.
use super::{AstNode, AstOp, DType, Lowerer, NodeData, NodeId, ShapeTracker};
use crate::ast::Const;
use crate::graph::shape::expr::Expr;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_input(
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

    pub(super) fn lower_full(
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

    pub(super) fn lower_rand(
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

    pub(super) fn lower_contiguous(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        if src_tracker.is_contiguous() {
            return (src_ast, src_tracker, src_buffer_id);
        }

        let src_buffer = self.get_buffer_var(src_buffer_id);
        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut loops = vec![];
        let mut loop_vars = vec![];
        for shape_expr in dst_tracker.shape().iter() {
            let loop_var = self.new_loop_counter();
            loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
        }

        let src_offset = src_tracker.offset_expr(&loop_vars);
        let dst_offset = dst_tracker.offset_expr(&loop_vars);

        let load_node = AstNode::deref(src_buffer.buffer_index(src_offset.simplify().into()));
        let store_node = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            load_node,
        );

        let final_block = AstNode::build_loops(loops, vec![store_node]);

        (final_block, dst_tracker, node_id)
    }

    pub(super) fn lower_permute(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        axes: Vec<usize>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.permute(axes);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(super) fn lower_squeeze(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        axis: usize,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.squeeze(axis);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(super) fn lower_unsqueeze(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        axis: usize,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.unsqueeze(axis);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(super) fn lower_expand(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        new_shape: Vec<Expr>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.expand(new_shape);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(super) fn lower_slice(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        args: Vec<(Expr, Expr)>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.slice(&args);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(super) fn lower_unfold1d(
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

    pub(super) fn lower_unfold2d(
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

    pub(super) fn lower_reshape(
        &mut self,
        _node_id: NodeId,
        node_data: &NodeData,
        new_shape: Vec<Expr>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);
        let new_tracker = src_tracker.reshape(new_shape);
        (src_ast, new_tracker, src_buffer_id)
    }

    pub(super) fn lower_elementwise(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
        op: AstOp,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let mut src_asts = vec![];
        for &src_id in &node_data.src {
            let (src_ast, _, _) = self.lower_node(src_id);
            src_asts.push(src_ast);
        }

        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut loops = vec![];
        let mut loop_vars = vec![];
        for shape_expr in dst_tracker.shape().iter() {
            let loop_var = self.new_loop_counter();
            loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
        }

        let mut loaded_srcs = vec![];
        for &src_id in node_data.src.iter() {
            let (_, tracker, buffer_id) = self.cache.get(&src_id).unwrap().clone();
            let buffer = self.get_buffer_var(buffer_id);
            let offset = tracker.offset_expr(&loop_vars);
            let load = AstNode::deref(buffer.buffer_index(offset.simplify().into()));
            loaded_srcs.push(load);
        }

        let op_node = AstNode::new(op, loaded_srcs, node_data.dtype.clone());
        let dst_offset = dst_tracker.offset_expr(&loop_vars);
        let store_node = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            op_node,
        );

        let mut final_block_src = vec![];
        for ast in src_asts {
            if let AstOp::Block = ast.op {
                final_block_src.extend(ast.src);
            } else {
                final_block_src.push(ast);
            }
        }
        final_block_src.push(AstNode::build_loops(loops, vec![store_node]));
        let final_block = AstNode::new(AstOp::Block, final_block_src, DType::Void);

        (final_block, dst_tracker, node_id)
    }

    pub(super) fn lower_fused_elementwise(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
        elementwise_ast: AstNode,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let mut src_asts = vec![];
        for &src_id in &node_data.src {
            let (src_ast, _, _) = self.lower_node(src_id);
            src_asts.push(src_ast);
        }

        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut loops = vec![];
        let mut loop_vars = vec![];
        for shape_expr in dst_tracker.shape().iter() {
            let loop_var = self.new_loop_counter();
            loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
        }

        let op_node = self.lower_fused_ast(&elementwise_ast, &loop_vars, &node_data.src);

        let dst_offset = dst_tracker.offset_expr(&loop_vars);
        let store_node = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            op_node,
        );

        let mut final_block_src = vec![];
        for ast in src_asts {
            if let AstOp::Block = ast.op {
                final_block_src.extend(ast.src);
            } else {
                final_block_src.push(ast);
            }
        }
        final_block_src.push(AstNode::build_loops(loops, vec![store_node]));
        let final_block = AstNode::new(AstOp::Block, final_block_src, DType::Void);

        (final_block, dst_tracker, node_id)
    }

    pub(super) fn lower_reduce(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
        op: AstOp,
        axis: usize,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);

        let src_buffer = self.get_buffer_var(src_buffer_id);
        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut loops = vec![];
        let mut outer_loop_vars = vec![];
        for shape_expr in dst_tracker.shape().iter() {
            let loop_var = self.new_loop_counter();
            outer_loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
        }

        let acc_var = self.new_accumulator_name();
        let init_val = match op {
            AstOp::Add => AstNode::from(0.0f32),
            AstOp::Mul => AstNode::from(1.0f32),
            AstOp::Max => AstNode::from(f32::NEG_INFINITY),
            _ => unimplemented!("Unsupported reduce op"),
        }
        .with_type(node_data.dtype.clone());

        let init_acc = AstNode::declare(acc_var.clone(), node_data.dtype.clone(), init_val);

        let inner_loop_var = self.new_loop_counter();
        let mut full_indices = outer_loop_vars.clone();
        full_indices.insert(axis, inner_loop_var.clone());
        let src_offset = src_tracker.offset_expr(&full_indices);
        let load_val = AstNode::deref(src_buffer.buffer_index(src_offset.simplify().into()));

        let update_acc = AstNode::assign(
            AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
            AstNode::new(
                op,
                vec![
                    AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
                    load_val,
                ],
                node_data.dtype.clone(),
            ),
        );

        let inner_loop = AstNode::range(
            inner_loop_var,
            src_tracker.shape()[axis].clone().into(),
            vec![update_acc],
        );

        let dst_offset = dst_tracker.offset_expr(&outer_loop_vars);
        let store_result = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
        );

        let mut final_block_src = vec![];
        if let AstOp::Block = src_ast.op {
            final_block_src.extend(src_ast.src);
        } else {
            final_block_src.push(src_ast);
        }
        final_block_src.push(AstNode::build_loops(
            loops,
            vec![init_acc, inner_loop, store_result],
        ));
        let final_block = AstNode::new(AstOp::Block, final_block_src, DType::Void);

        (final_block, dst_tracker, node_id)
    }

    pub(super) fn lower_cumulative(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
        op: AstOp,
        axis: usize,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);

        let src_buffer = self.get_buffer_var(src_buffer_id);
        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut loops = vec![];
        let mut outer_loop_vars = vec![];
        for (i, shape_expr) in src_tracker.shape().iter().enumerate() {
            if i == axis {
                continue;
            }
            let loop_var = self.new_loop_counter();
            outer_loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
        }

        let acc_var = self.new_accumulator_name();
        let init_val = match op {
            AstOp::Add => AstNode::from(0.0f32),
            AstOp::Mul => AstNode::from(1.0f32),
            AstOp::Max => AstNode::from(f32::NEG_INFINITY),
            _ => unimplemented!("Unsupported cumulative op"),
        }
        .with_type(node_data.dtype.clone());

        let init_acc = AstNode::declare(acc_var.clone(), node_data.dtype.clone(), init_val);

        let inner_loop_var = self.new_loop_counter();
        let mut full_indices = outer_loop_vars.clone();
        full_indices.insert(axis, inner_loop_var.clone());

        let src_offset = src_tracker.offset_expr(&full_indices);
        let load_val = AstNode::deref(src_buffer.buffer_index(src_offset.simplify().into()));

        let update_acc = AstNode::assign(
            AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
            AstNode::new(
                op,
                vec![
                    AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
                    load_val,
                ],
                node_data.dtype.clone(),
            ),
        );

        let dst_offset = dst_tracker.offset_expr(&full_indices);
        let store_result = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
        );

        let inner_loop = AstNode::range(
            inner_loop_var,
            src_tracker.shape()[axis].clone().into(),
            vec![update_acc, store_result],
        );

        let mut final_block_src = vec![];
        if let AstOp::Block = src_ast.op {
            final_block_src.extend(src_ast.src);
        } else {
            final_block_src.push(src_ast);
        }
        final_block_src.push(AstNode::build_loops(loops, vec![init_acc, inner_loop]));
        let final_block = AstNode::new(AstOp::Block, final_block_src, DType::Void);

        (final_block, dst_tracker, node_id)
    }
}
