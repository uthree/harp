use rustc_hash::FxHashMap;

use crate::{
    ast::{AstNode, AstOp, DType},
    backend::{BufferInfo, KernelDetails},
    graph::{NodeId, shape::tracker::ShapeTracker},
};

use super::Lowerer;

pub trait LoweringOrchestrator {
    fn lower(&mut self) -> (AstNode, KernelDetails);
}

impl<'a> LoweringOrchestrator for Lowerer<'a> {
    /// Lowers the entire graph into a single C-callable function `kernel_main`.
    ///
    /// This method creates a unified entry point with a stable signature:
    /// `void kernel_main(void** buffers, size_t* shape_vars)`
    fn lower(&mut self) -> (AstNode, KernelDetails) {
        log::info!("Starting lowering process for the entire graph...");

        let mut details = KernelDetails::default();
        let mut buffer_info_map = FxHashMap::default();
        let mut node_id_to_buffer_index = FxHashMap::default();
        let mut current_buffer_idx = 0;

        // 1. Assign buffer indices to all input nodes first.
        let inputs = self.graph.inputs.borrow().clone();
        for node_id in &inputs {
            if self.buffer_map.contains_key(node_id) {
                continue;
            }
            self.buffer_map.insert(*node_id, current_buffer_idx);
            node_id_to_buffer_index.insert(*node_id, current_buffer_idx);
            let node = &self.graph.nodes.borrow()[node_id.0];
            buffer_info_map.insert(
                current_buffer_idx,
                BufferInfo {
                    dtype: node.dtype.clone(),
                    shape: node.shape.clone(),
                },
            );
            current_buffer_idx += 1;
        }

        // 2. Assign buffer indices to all output nodes.
        let outputs = self.graph.outputs.borrow().clone();
        for node_id in &outputs {
            if self.buffer_map.contains_key(node_id) {
                continue;
            }
            self.buffer_map.insert(*node_id, current_buffer_idx);
            node_id_to_buffer_index.insert(*node_id, current_buffer_idx);
            let node = &self.graph.nodes.borrow()[node_id.0];
            buffer_info_map.insert(
                current_buffer_idx,
                BufferInfo {
                    dtype: node.dtype.clone(),
                    shape: node.shape.clone(),
                },
            );
            current_buffer_idx += 1;
        }

        // 3. Assign indices to any remaining intermediate nodes that need buffers.
        let nodes = self.graph.nodes.borrow();
        for (i, node) in nodes.iter().enumerate() {
            let node_id = NodeId(i);
            if self.buffer_map.contains_key(&node_id) {
                continue; // Already processed (input or output)
            }

            match node.op {
                crate::graph::GraphOp::Full(_)
                | crate::graph::GraphOp::Rand
                | crate::graph::GraphOp::Contiguous
                | crate::graph::GraphOp::Elementwise(_)
                | crate::graph::GraphOp::Reduce(_, _)
                | crate::graph::GraphOp::Cumulative(_, _)
                | crate::graph::GraphOp::FusedElementwise(_)
                | crate::graph::GraphOp::FusedElementwiseReduce(_, _, _)
                | crate::graph::GraphOp::FusedReduce(_, _) => {
                    self.buffer_map.insert(node_id, current_buffer_idx);
                    node_id_to_buffer_index.insert(node_id, current_buffer_idx);
                    buffer_info_map.insert(
                        current_buffer_idx,
                        BufferInfo {
                            dtype: node.dtype.clone(),
                            shape: node.shape.clone(),
                        },
                    );
                    current_buffer_idx += 1;
                }
                _ => {} // View ops don't need a buffer
            }
        }

        // Populate KernelDetails buffers in the correct order
        for i in 0..current_buffer_idx {
            details.buffers.push(buffer_info_map.remove(&i).unwrap());
        }

        // Collect all unique shape variables from inputs and outputs
        let mut shape_vars = std::collections::HashSet::new();
        for &node_id in inputs.iter().chain(outputs.iter()) {
            for expr in &nodes[node_id.0].shape {
                expr.collect_variables(&mut shape_vars);
            }
        }
        details.shape_variables = shape_vars.into_iter().collect();
        details.buffer_map = self.buffer_map.clone();

        // 2. Lower all output nodes to generate the computation logic.
        let mut computation_body = vec![];
        for &output_id in &outputs {
            let (ast_node, tracker, buffer_id) = self.lower_node(output_id);
            computation_body.push(ast_node);

            // If the output node is a view and its buffer is different from the
            // source buffer, we need to copy the data.
            if buffer_id != output_id {
                let dst_buffer = self.get_buffer_var(output_id);
                let dst_tracker =
                    ShapeTracker::new(self.graph.nodes.borrow()[output_id.0].shape.clone());

                let mut loops = vec![];
                let mut loop_vars = vec![];
                for shape_expr in dst_tracker.shape().iter() {
                    let loop_var = self.new_loop_counter();
                    loop_vars.push(loop_var.clone());
                    loops.push(AstNode::range(loop_var, shape_expr.clone().into(), vec![]));
                }

                let src_buffer = self.get_buffer_var(buffer_id);
                let src_offset = tracker.offset_expr(&loop_vars);
                let dst_offset = dst_tracker.offset_expr(&loop_vars);

                let load_node =
                    AstNode::deref(src_buffer.buffer_index(src_offset.simplify().into()));
                let store_node = AstNode::store(
                    dst_buffer.buffer_index(dst_offset.simplify().into()),
                    load_node,
                );

                computation_body.push(AstNode::build_loops(loops, vec![store_node]));
            }
        }

        // 3. Create the implementation function `kernel_impl`.
        let mut impl_args = vec![];
        let mut impl_call_vars = vec![];
        for i in 0..details.buffers.len() {
            let buf_name = format!("buf{i}");
            let buffer_info = &details.buffers[i];
            let arg_type = DType::Ptr(Box::new(buffer_info.dtype.clone()));
            impl_args.push((buf_name.clone(), arg_type.clone()));
            impl_call_vars.push(AstNode::var(&buf_name).with_type(arg_type));
        }
        let kernel_impl = AstNode::func_def("kernel_impl", impl_args, computation_body);

        // 4. Create the main wrapper function `kernel_main`.
        let main_args = vec![
            (
                "buffers".to_string(),
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))), // void**
            ),
            (
                "shape_vars".to_string(),
                DType::Ptr(Box::new(DType::USize)), // size_t*
            ),
        ];
        let mut main_body = vec![];
        // Create buffer variables: `float* buf0 = (float*)buffers[0];`
        for (node_id, &index) in &node_id_to_buffer_index {
            let var_name = format!("buf{index}");
            let buffer_ptr = self.get_buffer_ptr(*node_id);
            let node_data = &self.graph.nodes.borrow()[node_id.0];
            let var_decl = AstNode::declare(
                var_name,
                DType::Ptr(Box::new(node_data.dtype.clone())),
                buffer_ptr,
            );
            main_body.push(var_decl);
        }
        // Create the call to the implementation function.
        let call_impl = AstNode::call("kernel_impl", impl_call_vars);
        main_body.push(call_impl);

        let kernel_main = AstNode::func_def("kernel_main", main_args, main_body);

        // 5. Combine both functions into a single AST program.
        let final_ast = AstNode::new(AstOp::Program, vec![kernel_impl, kernel_main], DType::Void);

        log::info!("Lowering process completed.");
        (final_ast, details)
    }
}
