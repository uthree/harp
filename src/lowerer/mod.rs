//! Lowers the graph representation to an AST representation.
mod ops;

use crate::ast::{AstNode, AstOp, DType};
use crate::graph::shape::expr::Expr as ShapeExpr;
use crate::graph::shape::view::View;
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::HashMap;

/// Lowers a `Graph` to an `AstNode` representing a computation kernel.
pub fn lower_graph(graph: &Graph) -> AstNode {
    let mut lowerer = Lowerer::new();
    lowerer.lower_internal(graph)
}

#[derive(Default)]
pub struct Lowerer {
    // Cache for lowered nodes. This is not used in the recursive element-wise lowering,
    // but could be used if we were lowering node by node.
    // For now, it's unused.
    _cache: HashMap<GraphNode, AstNode>,
    pub acc_counter: usize,
    pub ridx_counter: usize,
}

impl Lowerer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lowers a `Graph` to an `AstNode` representing a computation kernel.
    ///
    /// The generated AST will be a function that takes input and output tensors as pointers.
    fn lower_internal(&mut self, graph: &Graph) -> AstNode {
        // 1. Generate the implementation function (`kernel_impl`)
        let mut impl_args = vec![];
        let mut call_args = vec![];

        // Add tensor buffers to impl_args and prepare call_args for kernel_main
        // The order is expected to be outputs, then inputs by the backend.
        let mut buffer_count = 0;
        for (i, output_sig) in graph.signature.outputs.iter().enumerate() {
            let arg_name = format!("output{}", i);
            let dtype = DType::Ptr(Box::new(output_sig.dtype.clone()));
            impl_args.push((arg_name.clone(), dtype.clone()));

            let bufs_var = AstNode::var(
                "bufs",
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
            );
            let loaded_ptr = AstNode::load(AstNode::index(
                bufs_var,
                AstNode::from(buffer_count as usize),
            ));
            call_args.push(loaded_ptr.cast(dtype));
            buffer_count += 1;
        }
        for (i, input_sig) in graph.signature.inputs.iter().enumerate() {
            let arg_name = format!("input{}", i);
            let dtype = DType::Ptr(Box::new(input_sig.dtype.clone()));
            impl_args.push((arg_name.clone(), dtype.clone()));

            let bufs_var = AstNode::var(
                "bufs",
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
            );
            let loaded_ptr = AstNode::load(AstNode::index(
                bufs_var,
                AstNode::from(buffer_count as usize),
            ));
            call_args.push(loaded_ptr.cast(dtype));
            buffer_count += 1;
        }

        // Add shape variables to impl_args and prepare call_args for kernel_main
        for (i, shape_var) in graph.signature.shape_variables.iter().enumerate() {
            let arg_name = shape_var.name.clone();
            let dtype = DType::Usize;
            impl_args.push((arg_name, dtype.clone()));

            let shape_vars_var = AstNode::var("shape_vars", DType::Ptr(Box::new(DType::Usize)));
            call_args.push(AstNode::load(AstNode::index(
                shape_vars_var,
                AstNode::from(i),
            )));
        }

        let mut body = vec![];
        for (i, output_node) in graph.outputs.iter().enumerate() {
            let output_ptr = AstNode::var(
                &format!("output{}", i),
                DType::Ptr(Box::new(output_node.dtype.clone())),
            );
            let shape = output_node.shape().to_vec();
            let mut indices = Vec::new();
            let loops = create_loops(
                &shape,
                |indices| {
                    let value_to_store = self.lower_node_rec(output_node, indices, &graph.inputs);
                    let output_view = View::new_contiguous(shape.clone());
                    let physical_index = output_view.to_physical_index_ast(indices);
                    let store_ptr = AstNode::index(output_ptr.clone(), physical_index);
                    AstNode::store(store_ptr, value_to_store)
                },
                &mut indices,
            );
            body.push(loops);
        }

        let kernel_impl = AstNode::func("kernel_impl", impl_args, AstNode::block(body));

        // 2. Generate the main entrypoint function (`kernel_main`)
        let main_args = vec![
            (
                "bufs".to_string(),
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
            ),
            ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize))),
        ];

        let call_impl = AstNode::call("kernel_impl", call_args);

        let kernel_main = AstNode::func(
            "kernel_main",
            main_args,
            AstNode::block(vec![call_impl]),
        );

        // 3. Return a program containing both functions
        AstNode::program(vec![kernel_impl, kernel_main])
    }

    /// Recursively lowers a `GraphNode` to an `AstNode` for a single element computation.
    ///
    /// `indices` represents the logical indices for the element being computed.
    pub fn lower_node_rec(
        &mut self,
        node: &GraphNode,
        indices: &mut [AstNode],
        inputs: &[GraphNode],
    ) -> AstNode {
        match &node.op {
            GraphOp::Input { .. } => ops::input::lower_input(self, node, indices, inputs),
            GraphOp::Full(_) => ops::full::lower_full(self, &node.op),
            GraphOp::Rand => ops::rand::lower_rand(self),
            GraphOp::Elementwise(op) => {
                ops::elementwise::lower_elementwise(self, node, indices, inputs, op)
            }
            GraphOp::FusedElementwise(fused_ast) => {
                ops::fused::lower_fused_elementwise(self, node, indices, inputs, fused_ast)
            }
            GraphOp::FusedElementwiseReduce(fused_ast, op, axes) => {
                ops::fused::lower_fused_elementwise_reduce(
                    self, node, indices, inputs, fused_ast, op, axes,
                )
            }
            GraphOp::FusedReduce(op, axes) => {
                ops::fused::lower_fused_reduce(self, node, indices, inputs, op, axes)
            }
            GraphOp::Reduce(op, axis) => {
                ops::reduce::lower_reduce(self, node, indices, inputs, op, axis)
            }
            GraphOp::Permute(axes) => {
                ops::permute::lower_permute(self, node, indices, inputs, axes)
            }
            GraphOp::Cumulative(op, axis) => {
                ops::cumulative::lower_cumulative(self, node, indices, inputs, op, axis)
            }
            GraphOp::Contiguous => ops::contiguous::lower_contiguous(self, node, indices, inputs),
            _ => unimplemented!(
                "Lowering for this GraphOp is not implemented yet: {:?}",
                node.op
            ),
        }
    }
}

/// Creates nested loops for a given shape.
fn create_loops(
    shape: &[ShapeExpr],
    mut body_builder: impl FnMut(&mut [AstNode]) -> AstNode,
    indices: &mut Vec<AstNode>,
) -> AstNode {
    indices.clear();
    for i in 0..shape.len() {
        let counter_name = format!("idx{}", i);
        indices.push(AstNode::var(&counter_name, DType::Isize));
    }
    let body = body_builder(indices);

    let mut current_body = body;
    // Build loops from the inside out.
    for (i, dim) in shape.iter().enumerate().rev() {
        let counter_name = format!("idx{}", i);
        let loop_content = if let AstOp::Block = current_body.op {
            current_body
        } else {
            AstNode::block(vec![current_body])
        };
        current_body = AstNode::range(&counter_name, 1, dim.clone().into(), loop_content);
    }
    current_body
}
