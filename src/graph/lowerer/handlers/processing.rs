use crate::{
    ast::{AstNode, AstOp, DType},
    graph::{
        lowerer::Lowerer,
        node::{NodeData, NodeId},
        shape::tracker::ShapeTracker,
    },
};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_contiguous(
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
            loops.push(AstNode::range(
                loop_var,
                shape_expr.clone().into(),
                vec![],
                false,
            ));
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

    pub(crate) fn lower_elementwise(
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
            loops.push(AstNode::range(
                loop_var,
                shape_expr.clone().into(),
                vec![],
                false,
            ));
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

    pub(crate) fn lower_fused_elementwise(
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
            loops.push(AstNode::range(
                loop_var,
                shape_expr.clone().into(),
                vec![],
                false,
            ));
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

    pub(crate) fn lower_reduce(
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
            loops.push(AstNode::range(
                loop_var,
                shape_expr.clone().into(),
                vec![],
                false,
            ));
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
            false,
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

    pub(crate) fn lower_cumulative(
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
            loops.push(AstNode::range(
                loop_var,
                shape_expr.clone().into(),
                vec![],
                false,
            ));
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
            false,
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

    pub(crate) fn lower_fused_elementwise_reduce(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
        elementwise_ast: AstNode,
        op: AstOp,
        axis: usize, // For now, assume single axis
    ) -> (AstNode, ShapeTracker, NodeId) {
        // 1. Lower source nodes to populate cache
        let mut src_asts = vec![];
        for &src_id in &node_data.src {
            let (src_ast, _, _) = self.lower_node(src_id);
            src_asts.push(src_ast);
        }

        // 2. Setup trackers and buffers
        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        // 3. Create outer loops for non-reducing dimensions
        let mut loops = vec![];
        let mut outer_loop_vars = vec![];
        for shape_expr in node_data.shape.iter() {
            // The shape of the fused op is the output shape, which is already reduced.
            let loop_var = self.new_loop_counter();
            outer_loop_vars.push(loop_var.clone());
            loops.push(AstNode::range(
                loop_var,
                shape_expr.clone().into(),
                vec![],
                false,
            ));
        }

        // 4. Initialize accumulator
        let acc_var = self.new_accumulator_name();
        let init_val = match op {
            AstOp::Add => AstNode::from(0.0f32),
            AstOp::Mul => AstNode::from(1.0f32),
            AstOp::Max => AstNode::from(f32::NEG_INFINITY),
            _ => unimplemented!("Unsupported reduce op for fusion"),
        }
        .with_type(node_data.dtype.clone());
        let init_acc = AstNode::declare(acc_var.clone(), node_data.dtype.clone(), init_val);

        // 5. Create inner loop for reduction
        let inner_loop_var = self.new_loop_counter();
        let mut full_indices = outer_loop_vars.clone();
        // We need the original source shape to know the reduction size
        let src_shape = self.graph.nodes.borrow()[node_data.src[0].0].shape.clone();
        full_indices.insert(axis, inner_loop_var.clone());

        // 6. Lower the fused AST inside the loop
        let fused_op_node = self.lower_fused_ast(&elementwise_ast, &full_indices, &node_data.src);

        // 7. Update accumulator with the fused result
        let update_acc = AstNode::assign(
            AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
            AstNode::new(
                op.clone(),
                vec![
                    AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
                    fused_op_node,
                ],
                node_data.dtype.clone(),
            ),
        );

        let inner_loop = AstNode::range(
            inner_loop_var,
            src_shape[axis].clone().into(),
            vec![update_acc],
            false,
        );

        // 8. Store the final result
        let dst_offset = dst_tracker.offset_expr(&outer_loop_vars);
        let store_result = AstNode::store(
            dst_buffer.buffer_index(dst_offset.simplify().into()),
            AstNode::var(&acc_var).with_type(node_data.dtype.clone()),
        );

        // 9. Assemble the final AST block
        let mut final_block_src = vec![];
        for ast in src_asts {
            if let AstOp::Block = ast.op {
                final_block_src.extend(ast.src);
            } else {
                final_block_src.push(ast);
            }
        }
        final_block_src.push(AstNode::build_loops(
            loops,
            vec![init_acc, inner_loop, store_result],
        ));
        let final_block = AstNode::new(AstOp::Block, final_block_src, DType::Void);

        (final_block, dst_tracker, node_id)
    }

    pub(crate) fn lower_fused_reduce(
        &mut self,
        node_id: NodeId,
        node_data: &NodeData,
        op: AstOp,
        axes: Vec<usize>,
    ) -> (AstNode, ShapeTracker, NodeId) {
        let (src_ast, src_tracker, src_buffer_id) = self.lower_node(node_data.src[0]);

        let src_buffer = self.get_buffer_var(src_buffer_id);
        let dst_buffer = self.get_buffer_var(node_id);
        let dst_tracker = ShapeTracker::new(node_data.shape.clone());

        let mut outer_loops = vec![];
        let mut outer_loop_vars = vec![];
        let mut full_indices = vec![];

        for (i, shape_expr) in src_tracker.shape().iter().enumerate() {
            if axes.contains(&i) {
                // This is a reduction axis, will be handled by inner loops
                full_indices.push("".to_string()); // Placeholder
            } else {
                // This is a dimension that we keep
                let loop_var = self.new_loop_counter();
                outer_loop_vars.push(loop_var.clone());
                full_indices.push(loop_var.clone());
                outer_loops.push(AstNode::range(
                    loop_var,
                    shape_expr.clone().into(),
                    vec![],
                    false,
                ));
            }
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

        let mut inner_loops = vec![];

        for (i, shape_expr) in src_tracker.shape().iter().enumerate() {
            if axes.contains(&i) {
                let loop_var = self.new_loop_counter();
                full_indices[i] = loop_var.clone();
                inner_loops.push(AstNode::range(
                    loop_var,
                    shape_expr.clone().into(),
                    vec![],
                    false,
                ));
            }
        }

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

        let nested_inner_loop = AstNode::build_loops(inner_loops, vec![update_acc]);

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
            outer_loops,
            vec![init_acc, nested_inner_loop, store_result],
        ));
        let final_block = AstNode::new(AstOp::Block, final_block_src, DType::Void);

        (final_block, dst_tracker, node_id)
    }
}
