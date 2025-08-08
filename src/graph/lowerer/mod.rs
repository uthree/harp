//! This module provides the `Lowerer`, which is responsible for converting a
//! high-level computation graph (`Graph`) into a low-level Abstract Syntax Tree (`AstNode`).
//!
//! This process, known as "lowering," transforms graph operations into a more
//! explicit, loop-based representation that is closer to executable code.

mod handlers;
pub mod orchestrator;

use crate::ast::{AstNode, AstOp, DType};
use crate::graph::shape::tracker::ShapeTracker;
use crate::graph::{Graph, GraphOp, NodeId};
use log::{debug, trace};
pub use orchestrator::LoweringOrchestrator;
use rustc_hash::FxHashMap;

/// Traverses a `Graph` and converts it into an `AstNode`.
///
/// The `Lowerer` maintains a cache to avoid re-processing nodes and uses
/// separate counters to generate unique names for variables, loops, and buffers.
#[derive(Debug, Clone)]
pub struct Lowerer<'a> {
    /// A reference to the computation graph to be lowered.
    pub(super) graph: &'a Graph,
    /// A cache to store the lowered AST and `ShapeTracker` for each processed `NodeId`.
    pub(super) cache: FxHashMap<NodeId, (AstNode, ShapeTracker, NodeId)>,
    /// A map from NodeId to its corresponding index in the `buffers` array.
    pub(super) buffer_map: FxHashMap<NodeId, usize>,
    /// Counter for generating unique loop variable names (e.g., `ridx0`, `ridx1`).
    pub(super) loop_counter: usize,
    /// Counter for generating unique accumulator names (e.g., `acc0`).
    pub(super) accumulator_counter: usize,
}

impl<'a> Lowerer<'a> {
    /// Creates a new `Lowerer` for a given `Graph`.
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            cache: FxHashMap::default(),
            buffer_map: FxHashMap::default(),
            loop_counter: 0,
            accumulator_counter: 0,
        }
    }

    // --- Private helper methods for variable name generation ---

    pub(super) fn new_loop_counter(&mut self) -> String {
        let name = format!("ridx{}", self.loop_counter);
        self.loop_counter += 1;
        name
    }

    pub(super) fn new_accumulator_name(&mut self) -> String {
        let name = format!("acc{}", self.accumulator_counter);
        self.accumulator_counter += 1;
        name
    }

    /// Retrieves a properly typed variable representing a buffer.
    /// e.g., `buf0`
    pub(super) fn get_buffer_var(&self, node_id: NodeId) -> AstNode {
        let buffer_index = self.buffer_map[&node_id];
        let node_data = &self.graph.nodes.borrow()[node_id.0];
        AstNode::var(&format!("buf{buffer_index}"))
            .with_type(DType::Ptr(Box::new(node_data.dtype.clone())))
    }

    /// Retrieves a properly typed pointer to a buffer from the `buffers` array.
    /// e.g., `((float*)buffers[i])`
    pub(super) fn get_buffer_ptr(&self, node_id: NodeId) -> AstNode {
        let buffer_index = self.buffer_map[&node_id];
        let node_data = &self.graph.nodes.borrow()[node_id.0];
        let buffer_var = AstNode::var("buffers");

        // Creates `(dtype*)buffers[buffer_index]`
        let index_node = AstNode::from(buffer_index as u64).cast(DType::USize);
        let buffer_access = AstNode::new(
            AstOp::BufferIndex,
            vec![buffer_var, index_node],
            // This represents the type of `buffers[i]`, which is `void*`
            DType::Ptr(Box::new(DType::Void)),
        );

        AstNode::new(
            AstOp::Cast(DType::Ptr(Box::new(node_data.dtype.clone()))),
            vec![buffer_access],
            DType::Ptr(Box::new(node_data.dtype.clone())),
        )
    }

    /// Recursively lowers a single node from the graph into an AST.
    pub(super) fn lower_node(&mut self, node_id: NodeId) -> (AstNode, ShapeTracker, NodeId) {
        trace!("Attempting to lower node {node_id:?}");
        if let Some(cached) = self.cache.get(&node_id) {
            debug!("Cache hit for node {node_id:?}");
            return cached.clone();
        }
        trace!("Cache miss for node {node_id:?}. Proceeding with lowering.");

        let node_data = self.graph.nodes.borrow()[node_id.0].clone();
        debug!("Lowering node {:?} with op {:?}", node_id, node_data.op);

        let result = match node_data.op.clone() {
            GraphOp::Input => self.lower_input(node_id, &node_data),
            GraphOp::Full(value) => self.lower_full(node_id, &node_data, value),
            GraphOp::Rand => self.lower_rand(node_id, &node_data),
            GraphOp::Contiguous => self.lower_contiguous(node_id, &node_data),
            GraphOp::Permute(axes) => self.lower_permute(node_id, &node_data, axes),
            GraphOp::Squeeze(axis) => self.lower_squeeze(node_id, &node_data, axis),
            GraphOp::Unsqueeze(axis) => self.lower_unsqueeze(node_id, &node_data, axis),
            GraphOp::Expand(new_shape) => self.lower_expand(node_id, &node_data, new_shape),
            GraphOp::Slice(args) => self.lower_slice(node_id, &node_data, args),
            GraphOp::Unfold1d {
                dim,
                kernel_size,
                stride,
            } => self.lower_unfold1d(node_id, &node_data, dim, kernel_size, stride),
            GraphOp::Unfold2d {
                kernel_size,
                stride,
            } => self.lower_unfold2d(node_id, &node_data, kernel_size, stride),
            GraphOp::Reshape(new_shape) => self.lower_reshape(node_id, &node_data, new_shape),
            GraphOp::Elementwise(op) => self.lower_elementwise(node_id, &node_data, op),
            GraphOp::FusedElementwise(elementwise_ast) => {
                self.lower_fused_elementwise(node_id, &node_data, elementwise_ast)
            }
            GraphOp::Reduce(op, axis) => self.lower_reduce(node_id, &node_data, op, axis),
            GraphOp::Cumulative(op, axis) => self.lower_cumulative(node_id, &node_data, op, axis),
            _ => unimplemented!("This TensorOp is not yet supported for lowering"),
        };

        trace!("Finished lowering node {node_id:?}. Caching result.");
        self.cache.insert(node_id, result.clone());
        result
    }

    pub(super) fn lower_fused_ast(
        &self,
        ast: &AstNode,
        loop_vars: &[String],
        src_nodes: &[NodeId],
    ) -> AstNode {
        match &ast.op {
            AstOp::Capture(n, _) => {
                let src_id = src_nodes[*n];
                // We need to lower the source node to ensure its tracker is in the cache.
                // This might seem redundant, but it's necessary if the source node hasn't been
                // visited yet in the main `lower_node` loop.
                let (_, tracker, buffer_id) = self.cache.get(&src_id).unwrap().clone();
                let buffer = self.get_buffer_var(buffer_id);
                let offset = tracker.offset_expr(loop_vars);
                AstNode::deref(buffer.buffer_index(offset.simplify().into()))
            }
            _ => {
                let new_srcs = ast
                    .src
                    .iter()
                    .map(|src_node| self.lower_fused_ast(src_node, loop_vars, src_nodes))
                    .collect();
                AstNode::new(ast.op.clone(), new_srcs, ast.dtype.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ops::{ElementwiseOps, ReduceOps};

    fn setup_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_lower_simple_add_creates_kernel_main() {
        setup_logger();
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into()]);
        let b = graph.input(DType::F32, vec![10.into()]);
        (a + b).as_output();

        let mut lowerer = Lowerer::new(&graph);
        let (ast, details) = lowerer.lower();

        // The top-level AST node should be a Program.
        if let AstOp::Program = ast.op {
            assert_eq!(ast.src.len(), 2, "Program should contain two functions");

            // Find kernel_main in the program
            let kernel_main = ast
                .src
                .iter()
                .find(|node| {
                    if let AstOp::Func { name, .. } = &node.op {
                        name == "kernel_main"
                    } else {
                        false
                    }
                })
                .expect("kernel_main function not found in program");

            if let AstOp::Func { name, args, .. } = &kernel_main.op {
                assert_eq!(*name, "kernel_main");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0].0, "buffers");
                assert_eq!(args[1].0, "shape_vars");
                // Check the body of kernel_main
                assert!(kernel_main.src.len() > 0);
                assert!(matches!(kernel_main.src[0].op, AstOp::Declare { .. }));
            } else {
                panic!("Expected a FuncDef node for kernel_main");
            }
        } else {
            panic!("Expected a Program node, found {:?}", ast.op);
        }

        assert_eq!(details.buffers.len(), 3);
        assert_eq!(details.shape_variables.len(), 0);
    }

    #[test]
    fn test_lower_elementwise_add_uses_buffers() {
        setup_logger();
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into()]); // buffer 0
        let b = graph.input(DType::F32, vec![10.into()]); // buffer 1
        (a + b).as_output(); // buffer 2

        let mut lowerer = Lowerer::new(&graph);
        let (ast, _details) = lowerer.lower();

        // Find the kernel_impl function
        let kernel_impl = ast
            .src
            .iter()
            .find(|node| {
                if let AstOp::Func { name, .. } = &node.op {
                    name == "kernel_impl"
                } else {
                    false
                }
            })
            .expect("kernel_impl function not found");

        // Check that the body of the function contains a store to the correct buffer
        if let AstOp::Func { .. } = &kernel_impl.op {
            // body -> [elementwise_ast_block]
            let elementwise_ast_block = &kernel_impl.src[0];
            if let AstOp::Block { .. } = &elementwise_ast_block.op {
                let elementwise_ast = &elementwise_ast_block.src[0];
                if let AstOp::Range { .. } = &elementwise_ast.op {
                    // The body of the range is in its src
                    let block = &elementwise_ast.src[1];
                    if let AstOp::Store = &block.op {
                        let dst = &block.src[0];
                        let src = &block.src[1];
                        // dst should be `buf2[ridx0]`
                        if let AstOp::BufferIndex = &dst.op {
                            let buffer_node = &dst.src[0];
                            if let AstOp::Var(name) = &buffer_node.op {
                                assert_eq!(name, "buf2");
                            } else {
                                panic!("Destination buffer is not the correct variable");
                            }
                        } else {
                            panic!("Destination not a buffer index");
                        }

                        // src should be `buf0[...] + buf1[...]`
                        assert!(matches!(src.op, AstOp::Add));
                    } else {
                        panic!("Expected store node, found {:?}", block.op);
                    }
                } else {
                    panic!("Expected range node, found {:?}", elementwise_ast.op);
                }
            } else {
                panic!("Expected block node, found {:?}", elementwise_ast_block.op);
            }
        } else {
            panic!("Expected FuncDef for kernel_impl");
        }
    }

    #[test]
    fn test_lower_reduce_sum_uses_buffers() {
        setup_logger();
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]); // buffer 0
        let _b = a.sum(1).as_output(); // buffer 1

        let mut lowerer = Lowerer::new(&graph);
        let (ast, _details) = lowerer.lower();

        // Find the kernel_impl function
        let kernel_impl = ast
            .src
            .iter()
            .find(|node| {
                if let AstOp::Func { name, .. } = &node.op {
                    name == "kernel_impl"
                } else {
                    false
                }
            })
            .expect("kernel_impl function not found");

        if let AstOp::Func { .. } = &kernel_impl.op {
            let reduce_ast_block = &kernel_impl.src[0];
            if let AstOp::Block { .. } = &reduce_ast_block.op {
                let reduce_ast = &reduce_ast_block.src[0];
                if let AstOp::Range { .. } = &reduce_ast.op {
                    // The body of the range is in its src
                    let block = &reduce_ast.src[1];
                    // block contains [init_acc, inner_loop, store_result]
                    let store_node = &block.src[2];
                    if let AstOp::Store = &store_node.op {
                        let dst = &store_node.src[0];
                        if let AstOp::BufferIndex = &dst.op {
                            let buffer_node = &dst.src[0];
                            // We are checking if the destination is the correct buffer variable
                            if let AstOp::Var(name) = &buffer_node.op {
                                assert_eq!(name, "buf1");
                            } else {
                                panic!("Expected destination to be a buffer variable");
                            }
                        } else {
                            panic!("Expected destination to be a buffer index");
                        }
                    } else {
                        panic!("Expected store node");
                    }
                } else {
                    panic!("Expected outer loop, found {:?}", reduce_ast.op);
                }
            } else {
                panic!("Expected block node, found {:?}", reduce_ast_block.op);
            }
        } else {
            panic!("Expected FuncDef for kernel_impl");
        }
    }

    #[test]
    fn test_lower_full() {
        setup_logger();
        let graph = Graph::new();
        graph.full(42.0f32, vec![10.into()]).as_output();

        let mut lowerer = Lowerer::new(&graph);
        let (ast, details) = lowerer.lower();

        assert_eq!(details.buffers.len(), 1);

        let kernel_impl = ast
            .src
            .iter()
            .find(|node| {
                if let AstOp::Func { name, .. } = &node.op {
                    name == "kernel_impl"
                } else {
                    false
                }
            })
            .expect("kernel_impl function not found");

        if let AstOp::Func { .. } = &kernel_impl.op {
            let full_ast = &kernel_impl.src[0];
            if let AstOp::Range { .. } = &full_ast.op {
                let block = &full_ast.src[1];
                if let AstOp::Store = &block.op {
                    let dst = &block.src[0];
                    let src = &block.src[1];
                    if let AstOp::BufferIndex = &dst.op {
                        let buffer_node = &dst.src[0];
                        if let AstOp::Var(name) = &buffer_node.op {
                            assert_eq!(name, "buf0");
                        } else {
                            panic!("Destination buffer is not the correct variable");
                        }
                    } else {
                        panic!("Destination not a buffer index");
                    }
                    assert!(matches!(src.op, AstOp::Const(_)));
                } else {
                    panic!("Expected store node, found {:?}", block.op);
                }
            } else {
                panic!("Expected range node, found {:?}", full_ast.op);
            }
        } else {
            panic!("Expected FuncDef for kernel_impl");
        }
    }

    #[test]
    fn test_lower_rand() {
        setup_logger();
        let graph = Graph::new();
        graph.rand(DType::F32, vec![10.into()]).as_output();

        let mut lowerer = Lowerer::new(&graph);
        let (ast, details) = lowerer.lower();

        assert_eq!(details.buffers.len(), 1);

        let kernel_impl = ast
            .src
            .iter()
            .find(|node| {
                if let AstOp::Func { name, .. } = &node.op {
                    name == "kernel_impl"
                } else {
                    false
                }
            })
            .expect("kernel_impl function not found");

        if let AstOp::Func { .. } = &kernel_impl.op {
            let rand_ast = &kernel_impl.src[0];
            if let AstOp::Range { .. } = &rand_ast.op {
                let block = &rand_ast.src[1];
                if let AstOp::Store = &block.op {
                    let dst = &block.src[0];
                    let src = &block.src[1];
                    if let AstOp::BufferIndex = &dst.op {
                        let buffer_node = &dst.src[0];
                        if let AstOp::Var(name) = &buffer_node.op {
                            assert_eq!(name, "buf0");
                        } else {
                            panic!("Destination buffer is not the correct variable");
                        }
                    } else {
                        panic!("Destination not a buffer index");
                    }
                    assert!(matches!(src.op, AstOp::Mul));
                } else {
                    panic!("Expected store node, found {:?}", block.op);
                }
            } else {
                panic!("Expected range node, found {:?}", rand_ast.op);
            }
        } else {
            panic!("Expected FuncDef for kernel_impl");
        }
    }
}
