//! Lowers the graph representation to an AST representation.
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
    acc_counter: usize,
    ridx_counter: usize,
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
            call_args.push(AstNode::_new(
                AstOp::Cast(dtype),
                vec![loaded_ptr],
                DType::Any,
            ));
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
            call_args.push(AstNode::_new(
                AstOp::Cast(dtype),
                vec![loaded_ptr],
                DType::Any,
            ));
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
                AstNode::from(i as usize),
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

        let kernel_impl = AstNode::_new(
            AstOp::Func {
                name: "kernel_impl".to_string(),
                args: impl_args,
            },
            vec![AstNode::_new(AstOp::Block, body, DType::Void)],
            DType::Void,
        );

        // 2. Generate the main entrypoint function (`kernel_main`)
        let main_args = vec![
            (
                "bufs".to_string(),
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
            ),
            ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize))),
        ];

        let call_impl = AstNode::_new(
            AstOp::Call("kernel_impl".to_string()),
            call_args,
            DType::Void,
        );

        let kernel_main = AstNode::_new(
            AstOp::Func {
                name: "kernel_main".to_string(),
                args: main_args,
            },
            vec![AstNode::_new(AstOp::Block, vec![call_impl], DType::Void)],
            DType::Void,
        );

        // 3. Return a program containing both functions
        AstNode::_new(AstOp::Program, vec![kernel_impl, kernel_main], DType::Void)
    }

    /// Recursively lowers a `GraphNode` to an `AstNode` for a single element computation.
    ///
    /// `indices` represents the logical indices for the element being computed.
    fn lower_node_rec(
        &mut self,
        node: &GraphNode,
        indices: &mut [AstNode],
        inputs: &[GraphNode],
    ) -> AstNode {
        match &node.op {
            GraphOp::Input { .. } => {
                let input_idx = inputs
                    .iter()
                    .position(|n| n == node)
                    .expect("Input node not found in graph inputs");
                let input_var = AstNode::var(
                    &format!("input{}", input_idx),
                    DType::Ptr(Box::new(node.dtype.clone())),
                );

                // Use the node's view to calculate the physical memory index from logical indices.
                let physical_index = node.view.to_physical_index_ast(indices);
                let ptr = AstNode::index(input_var, physical_index);
                AstNode::load(ptr)
            }
            GraphOp::Full(c) => AstNode::from(c.clone()),
            GraphOp::Elementwise(op) => {
                let srcs = node
                    .src
                    .iter()
                    .map(|n| self.lower_node_rec(n, indices, inputs))
                    .collect();
                AstNode::_new(op.clone(), srcs, node.dtype.clone())
            }
            GraphOp::Reduce(op, axis) => {
                let acc_name = format!("acc{}", self.acc_counter);
                self.acc_counter += 1;
                let acc_var = AstNode::var(&acc_name, node.dtype.clone());

                // Initialize accumulator.
                let init_val = match op {
                    AstOp::Add => node.dtype.zero(),
                    AstOp::Mul => node.dtype.one(),
                    // TODO: Handle other reduction types like Max.
                    _ => unimplemented!("Unsupported reduction operation: {:?}", op),
                };
                let declare_acc = AstNode::declare(&acc_name, node.dtype.clone(), Some(init_val));

                // Create reduction loop.
                let reduce_dim = node.src[0].shape()[*axis].clone();
                let ridx_name = format!("ridx{}", self.ridx_counter);
                self.ridx_counter += 1;
                let ridx_var = AstNode::var(&ridx_name, DType::Isize);

                let mut inner_indices = indices.to_vec();
                // Insert the reduction loop variable at the reduction axis.
                inner_indices.insert(*axis, ridx_var);

                let value_to_reduce = self.lower_node_rec(&node.src[0], &mut inner_indices, inputs);

                let update_op = AstNode::_new(
                    op.clone(),
                    vec![acc_var.clone(), value_to_reduce],
                    node.dtype.clone(),
                );
                let assign_op = AstNode::assign(acc_var.clone(), update_op);

                let loop_node = AstNode::_new(
                    AstOp::Range {
                        counter: ridx_name,
                        step: 1,
                    },
                    vec![reduce_dim.into(), assign_op],
                    DType::Void,
                );

                // Return a block that declares, computes, and returns the accumulator.
                AstNode::_new(
                    AstOp::Block,
                    vec![declare_acc, loop_node, acc_var],
                    node.dtype.clone(),
                )
            }
            GraphOp::Permute(axes) => {
                // Reorder indices according to the permutation axes before recursing.
                let mut unpermuted_indices = vec![AstNode::from(0isize); indices.len()];
                for (i, &axis) in axes.iter().enumerate() {
                    unpermuted_indices[axis] = indices[i].clone();
                }
                self.lower_node_rec(&node.src[0], &mut unpermuted_indices, inputs)
            }
            GraphOp::Cumulative(op, axis) => {
                // Create an accumulator variable, initialized to the identity of the op.
                let acc_name = format!("acc{}", self.acc_counter);
                self.acc_counter += 1;
                let acc_var = AstNode::var(&acc_name, node.dtype.clone());

                let init_val = match op {
                    AstOp::Add => node.dtype.zero(),
                    AstOp::Mul => node.dtype.one(),
                    _ => unimplemented!("Unsupported cumulative operation: {:?}", op),
                };
                let declare_acc = AstNode::declare(&acc_name, node.dtype.clone(), Some(init_val));

                // Create an inner loop to accumulate values from the source.
                // The loop runs from 0 up to the current index along the cumulative axis.
                let cum_limit = indices[*axis].clone() + AstNode::from(1isize);
                let cidx_name = format!("cidx{}", self.ridx_counter);
                self.ridx_counter += 1;
                let cidx_var = AstNode::var(&cidx_name, DType::Isize);

                let mut inner_indices = indices.to_vec();
                inner_indices[*axis] = cidx_var;

                let value_to_accumulate =
                    self.lower_node_rec(&node.src[0], &mut inner_indices, inputs);

                let update_op = AstNode::_new(
                    op.clone(),
                    vec![acc_var.clone(), value_to_accumulate],
                    node.dtype.clone(),
                );
                let assign_op = AstNode::assign(acc_var.clone(), update_op);

                let loop_node = AstNode::_new(
                    AstOp::Range {
                        counter: cidx_name,
                        step: 1,
                    },
                    vec![cum_limit, assign_op],
                    DType::Void,
                );

                // The final result is a block that declares the accumulator, runs the loop,
                // and then returns the final accumulator value.
                AstNode::_new(
                    AstOp::Block,
                    vec![declare_acc, loop_node, acc_var],
                    node.dtype.clone(),
                )
            }
            GraphOp::Contiguous => {
                // A contiguous op is a hint for memory layout. In a fused kernel,
                // we "look through" it and directly compute from its source,
                // relying on the view of the source node to handle memory access correctly.
                self.lower_node_rec(&node.src[0], indices, inputs)
            }
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
        current_body = AstNode::_new(
            AstOp::Range {
                counter: counter_name,
                step: 1,
            },
            vec![dim.clone().into(), current_body],
            DType::Void,
        );
    }
    current_body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::expr::Expr as ShapeExpr;

    #[test]
    fn test_lower_simple_add() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b;
        graph.outputs.push(c);
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        });

        let ast = lower_graph(&graph);

        // Expect a Program node with two functions: kernel_impl and kernel_main
        if let AstNode {
            op: AstOp::Program,
            src,
            ..
        } = &ast
        {
            assert_eq!(src.len(), 2);

            // Check kernel_impl
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(0)
            {
                assert_eq!(name, "kernel_impl");
                assert_eq!(args.len(), 3); // 1 output, 2 inputs
                assert_eq!(args[0].0, "output0");
                assert_eq!(args[1].0, "input0");
                assert_eq!(args[2].0, "input1");
            } else {
                panic!("Expected kernel_impl function");
            }

            // Check kernel_main
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(1)
            {
                assert_eq!(name, "kernel_main");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0].0, "bufs");
                assert_eq!(args[1].0, "shape_vars");
            } else {
                panic!("Expected kernel_main function");
            }
        } else {
            panic!("Expected a program AST node, got {:?}", ast);
        }
    }

    #[test]
    fn test_lower_reduce_sum() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.reduce(AstOp::Add, 1); // Reduce along axis 1
        graph.outputs.push(b.clone());
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype,
            shape: b.shape().to_vec(),
        });

        let ast = lower_graph(&graph);

        // Expect a Program node with two functions: kernel_impl and kernel_main
        if let AstNode {
            op: AstOp::Program,
            src,
            ..
        } = &ast
        {
            assert_eq!(src.len(), 2);

            // Check kernel_impl
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(0)
            {
                assert_eq!(name, "kernel_impl");
                assert_eq!(args.len(), 2); // 1 output, 1 input
            } else {
                panic!("Expected kernel_impl function");
            }

            // Check kernel_main
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(1)
            {
                assert_eq!(name, "kernel_main");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected kernel_main function");
            }
        } else {
            panic!("Expected a program AST node, got {:?}", ast);
        }
    }

    #[test]
    fn test_lower_cumulative_sum() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.cumulative(AstOp::Add, 0);
        graph.outputs.push(b.clone());
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype,
            shape: b.shape().to_vec(),
        });

        let ast = lower_graph(&graph);

        // Expect a Program node with two functions: kernel_impl and kernel_main
        if let AstNode {
            op: AstOp::Program,
            src,
            ..
        } = &ast
        {
            assert_eq!(src.len(), 2);

            // Check kernel_impl
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(0)
            {
                assert_eq!(name, "kernel_impl");
                assert_eq!(args.len(), 2); // 1 output, 1 input
                assert_eq!(args[0].0, "output0");
                assert_eq!(args[1].0, "input0");
            } else {
                panic!("Expected kernel_impl function");
            }

            // Check kernel_main
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(1)
            {
                assert_eq!(name, "kernel_main");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected kernel_main function");
            }
        } else {
            panic!("Expected a program AST node, got {:?}", ast);
        }
    }
}
