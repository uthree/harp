use crate::{
    ast::{AstNode, AstOp, DType},
    backend::{BufferInfo, BufferKind, KernelDetails},
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
    /// `void kernel_main(void** inputs, void** outputs, size_t* shape_vars)`
    fn lower(&mut self) -> (AstNode, KernelDetails) {
        log::info!("Starting lowering process for the entire graph...");

        let mut details = KernelDetails::default();
        let nodes = self.graph.nodes.borrow();
        let graph_inputs = self.graph.inputs.borrow();
        let graph_outputs = self.graph.outputs.borrow();

        // 1. Classify all buffers
        for (i, node) in nodes.iter().enumerate() {
            let node_id = NodeId(i);
            let is_graph_input = graph_inputs.contains(&node_id);
            let is_graph_output = graph_outputs.contains(&node_id);

            // Determine if the node needs a buffer and what kind
            let buffer_kind = if is_graph_input {
                Some(BufferKind::Input)
            } else if is_graph_output {
                Some(BufferKind::Output)
            } else {
                let needs_materialization = matches!(
                    node.op,
                    crate::graph::GraphOp::Full(_)
                        | crate::graph::GraphOp::Rand
                        | crate::graph::GraphOp::Contiguous
                        | crate::graph::GraphOp::Elementwise(_)
                        | crate::graph::GraphOp::Reduce(_, _)
                        | crate::graph::GraphOp::Cumulative(_, _)
                        | crate::graph::GraphOp::FusedElementwise(_)
                        | crate::graph::GraphOp::FusedElementwiseReduce(_, _, _)
                        | crate::graph::GraphOp::FusedReduce(_, _)
                );
                if needs_materialization {
                    Some(BufferKind::Intermediate)
                } else {
                    None // View op
                }
            };

            if let Some(kind) = buffer_kind {
                let info = BufferInfo {
                    kind: kind.clone(),
                    dtype: node.dtype.clone(),
                    shape: node.shape.clone(),
                };
                match kind {
                    BufferKind::Input => {
                        let index = details.inputs.len();
                        details.input_map.insert(node_id, index);
                        details.inputs.push(info);
                    }
                    BufferKind::Output => {
                        let index = details.outputs.len();
                        details.output_map.insert(node_id, index);
                        details.outputs.push(info);
                    }
                    BufferKind::Intermediate => {
                        let index = details.intermediates.len();
                        details.intermediate_map.insert(node_id, index);
                        details.intermediates.push(info);
                    }
                }
            }
        }

        // Update the lowerer's internal state with the new buffer maps
        self.details = details.clone();

        // 2. Collect all unique shape variables from inputs and outputs
        let mut shape_vars = std::collections::HashSet::new();
        for &node_id in graph_inputs.iter().chain(graph_outputs.iter()) {
            for expr in &nodes[node_id.0].shape {
                expr.collect_variables(&mut shape_vars);
            }
        }
        self.details.shape_variables = shape_vars.into_iter().collect();

        // 3. Lower all output nodes to generate the computation logic.
        let mut computation_body = vec![];
        for &output_id in graph_outputs.iter() {
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

        // 4. Create the implementation function `kernel_impl`.
        let mut impl_args = vec![];
        // Add inputs to impl signature
        for (i, info) in self.details.inputs.iter().enumerate() {
            let name = format!("input{i}");
            let arg_type = DType::Ptr(Box::new(info.dtype.clone()));
            impl_args.push((name, arg_type));
        }
        // Add outputs to impl signature
        for (i, info) in self.details.outputs.iter().enumerate() {
            let name = format!("output{i}");
            let arg_type = DType::Ptr(Box::new(info.dtype.clone()));
            impl_args.push((name, arg_type));
        }

        // Create the body for kernel_impl, wrapping computation with malloc/free
        let mut kernel_impl_body = vec![];
        // Allocate and declare intermediate buffers at the start of kernel_impl
        for (i, info) in self.details.intermediates.iter().enumerate() {
            let var_name = format!("tmp{i}");
            let size_expr: AstNode = info
                .shape
                .iter()
                .map(|e| e.clone().into())
                .reduce(|a, b| a * b)
                .unwrap_or_else(|| 1.into());
            let malloc_call = AstNode::malloc(size_expr, info.dtype.clone());
            kernel_impl_body.push(AstNode::declare(
                var_name,
                DType::Ptr(Box::new(info.dtype.clone())),
                malloc_call,
            ));
        }
        // Add the actual computation
        kernel_impl_body.extend(computation_body);
        // Free intermediate buffers at the end of kernel_impl
        for i in 0..self.details.intermediates.len() {
            let var_name = format!("tmp{i}");
            kernel_impl_body.push(AstNode::free(AstNode::var(&var_name)));
        }

        let kernel_impl = AstNode::func_def("kernel_impl", impl_args, kernel_impl_body);

        // 5. Create the main wrapper function `kernel_main`.
        let main_args = vec![
            (
                "inputs".to_string(),
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))), // void**
            ),
            (
                "outputs".to_string(),
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))), // void**
            ),
            (
                "shape_vars".to_string(),
                DType::Ptr(Box::new(DType::USize)), // size_t*
            ),
        ];
        let mut main_body = vec![];
        let mut impl_call_vars = vec![];

        // Declare input buffer pointers and prepare them for the impl call
        for (i, info) in self.details.inputs.iter().enumerate() {
            let var_name = format!("input{i}");
            let buffer_ptr = self.get_buffer_ptr_from_array("inputs", i, &info.dtype);
            main_body.push(AstNode::declare(
                var_name.clone(),
                DType::Ptr(Box::new(info.dtype.clone())),
                buffer_ptr,
            ));
            impl_call_vars
                .push(AstNode::var(&var_name).with_type(DType::Ptr(Box::new(info.dtype.clone()))));
        }
        // Declare output buffer pointers and prepare them for the impl call
        for (i, info) in self.details.outputs.iter().enumerate() {
            let var_name = format!("output{i}");
            let buffer_ptr = self.get_buffer_ptr_from_array("outputs", i, &info.dtype);
            main_body.push(AstNode::declare(
                var_name.clone(),
                DType::Ptr(Box::new(info.dtype.clone())),
                buffer_ptr,
            ));
            impl_call_vars
                .push(AstNode::var(&var_name).with_type(DType::Ptr(Box::new(info.dtype.clone()))));
        }

        // Create the call to the implementation function.
        let call_impl = AstNode::call("kernel_impl", impl_call_vars);
        main_body.push(call_impl);

        let kernel_main = AstNode::func_def("kernel_main", main_args, main_body);

        // 6. Combine both functions into a single AST program.
        let final_ast = AstNode::new(AstOp::Program, vec![kernel_impl, kernel_main], DType::Void);

        log::info!("Lowering process completed.");
        (final_ast, self.details.clone())
    }
}
