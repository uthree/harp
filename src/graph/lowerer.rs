//! This module provides the `Lowerer`, which is responsible for converting a
//! high-level computation graph (`Graph`) into a low-level Abstract Syntax Tree (`AstNode`).
//!
//! This process, known as "lowering," transforms graph operations into a more
//! explicit, loop-based representation that is closer to executable code.

use crate::ast::{AstNode, AstOp, DType};
use crate::backend::{BufferInfo, KernelDetails};
use crate::graph::shape::expr::Expr;
use crate::graph::shape::tracker::ShapeTracker;
use crate::graph::{Graph, NodeId, TensorOp};
use log::{debug, info, trace};
use rustc_hash::FxHashMap;

/// Traverses a `Graph` and converts it into an `AstNode`.
///
/// The `Lowerer` maintains a cache to avoid re-processing nodes and uses
/// separate counters to generate unique names for variables, loops, and buffers.
#[derive(Debug, Clone)]
pub struct Lowerer<'a> {
    /// A reference to the computation graph to be lowered.
    graph: &'a Graph,
    /// A cache to store the lowered AST and `ShapeTracker` for each processed `NodeId`.
    cache: FxHashMap<NodeId, (AstNode, ShapeTracker)>,
    /// A map from NodeId to its corresponding index in the `buffers` array.
    buffer_map: FxHashMap<NodeId, usize>,
    /// Counter for generating unique loop variable names (e.g., `ridx0`, `ridx1`).
    loop_counter: usize,
    /// Counter for generating unique accumulator names (e.g., `acc0`).
    accumulator_counter: usize,
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

    fn new_loop_counter(&mut self) -> String {
        let name = format!("ridx{}", self.loop_counter);
        self.loop_counter += 1;
        name
    }

    fn new_accumulator_name(&mut self) -> String {
        let name = format!("acc{}", self.accumulator_counter);
        self.accumulator_counter += 1;
        name
    }

    /// Retrieves a properly typed pointer to a buffer from the `buffers` array.
    /// e.g., `((float*)buffers[i])`
    fn get_buffer_ptr(&self, node_id: NodeId) -> AstNode {
        let buffer_index = self.buffer_map[&node_id];
        let node_data = &self.graph.nodes.borrow()[node_id.0];
        let buffer_var = AstNode::var("buffers");

        // Creates `(dtype*)buffers[buffer_index]`
        AstNode::new(
            AstOp::Cast(DType::Ptr(Box::new(node_data.dtype.clone()))),
            vec![Box::new(AstNode::new(
                AstOp::BufferIndex {
                    buffer: Box::new(buffer_var),
                    index: Box::new(AstNode::from(buffer_index as i64)),
                },
                vec![],
                // This represents the type of `buffers[i]`, which is `void*`
                DType::Ptr(Box::new(DType::Void)),
            ))],
            DType::Ptr(Box::new(node_data.dtype.clone())),
        )
    }

    /// Lowers the entire graph into a single C-callable function `kernel_main`.
    ///
    /// This method creates a unified entry point with a stable signature:
    /// `void kernel_main(void** buffers, size_t* shape_vars)`
    pub fn lower(&mut self) -> (AstNode, KernelDetails) {
        info!("Starting lowering process for the entire graph...");

        let mut details = KernelDetails::default();
        let mut buffer_info_map = FxHashMap::default();

        // 1. Identify all input and output buffers and assign them an index.
        // By iterating over `graph.inputs`, we get a deterministic order.
        let mut current_buffer_idx = 0;
        for &node_id in self.graph.inputs.borrow().iter() {
            let node = &self.graph.nodes.borrow()[node_id.0];
            self.buffer_map.insert(node_id, current_buffer_idx);
            buffer_info_map.insert(
                current_buffer_idx,
                BufferInfo {
                    dtype: node.dtype.clone(),
                    shape: node
                        .shape
                        .iter()
                        .map(|e| match e {
                            Expr::Const(v) => *v as usize,
                            _ => panic!("Cannot lower buffer with dynamic shapes"),
                        })
                        .collect(),
                },
            );
            current_buffer_idx += 1;
        }

        let outputs: Vec<_> = self.graph.outputs.borrow().clone();
        for &output_id in &outputs {
            let node_data = &self.graph.nodes.borrow()[output_id.0];
            self.buffer_map.entry(output_id).or_insert_with(|| {
                let idx = current_buffer_idx;
                buffer_info_map.insert(
                    idx,
                    BufferInfo {
                        dtype: node_data.dtype.clone(),
                        shape: node_data
                            .shape
                            .iter()
                            .map(|e| match e {
                                Expr::Const(v) => *v as usize,
                                _ => panic!("Cannot lower buffer with dynamic shapes"),
                            })
                            .collect(),
                    },
                );
                current_buffer_idx += 1;
                idx
            });
        }

        // Populate KernelDetails buffers in the correct order
        for i in 0..current_buffer_idx {
            details.buffers.push(buffer_info_map.remove(&i).unwrap());
        }

        // Collect all unique shape variables from inputs and outputs
        let mut shape_vars = std::collections::HashSet::new();
        let nodes = self.graph.nodes.borrow();
        for &node_id in self.graph.inputs.borrow().iter() {
            for expr in &nodes[node_id.0].shape {
                expr.collect_variables(&mut shape_vars);
            }
        }
        for &node_id in &outputs {
            for expr in &nodes[node_id.0].shape {
                expr.collect_variables(&mut shape_vars);
            }
        }
        details.shape_variables = shape_vars.into_iter().collect();

        // 2. Lower all output nodes to generate the computation logic.
        let mut body = vec![];
        for &output_id in &outputs {
            let (ast_node, _tracker) = self.lower_node(output_id);
            body.push(ast_node);
        }

        // 3. Wrap the logic in the main kernel function definition.
        let func_name = "kernel_main".to_string();
        let args = vec![
            (
                "buffers".to_string(),
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))), // void**
            ),
            (
                "shape_vars".to_string(),
                DType::Ptr(Box::new(DType::U64)), // size_t*
            ),
        ];
        let func_body = AstNode::new(
            AstOp::Block,
            body.into_iter().map(Box::new).collect(),
            DType::Void,
        );

        let final_ast = AstNode::func_def(&func_name, args, func_body);
        info!("Lowering process completed.");
        (final_ast, details)
    }

    /// Recursively lowers a single node from the graph into an AST.
    fn lower_node(&mut self, node_id: NodeId) -> (AstNode, ShapeTracker) {
        trace!("Attempting to lower node {node_id:?}");
        if let Some(cached) = self.cache.get(&node_id) {
            debug!("Cache hit for node {node_id:?}");
            return cached.clone();
        }
        trace!("Cache miss for node {node_id:?}. Proceeding with lowering.");

        let node_data = self.graph.nodes.borrow()[node_id.0].clone();
        debug!("Lowering node {:?} with op {:?}", node_id, node_data.op);

        let result = match node_data.op {
            TensorOp::Input => {
                // Input nodes are just pointers passed in the `buffers` array.
                // We return a ShapeTracker for them. The actual computation is a no-op.
                let tracker = ShapeTracker::new(node_data.shape);
                (AstNode::new(AstOp::Block, vec![], DType::Void), tracker) // No-op AST
            }
            TensorOp::Contiguous => {
                let (src_ast, src_tracker) = self.lower_node(node_data.src[0]);
                if src_tracker.is_contiguous() {
                    debug!("Node {node_id:?} is already contiguous. No-op.");
                    // If it's already contiguous, we don't need to do anything.
                    // The source buffer is already correct.
                    return (src_ast, src_tracker);
                }
                debug!("Node {node_id:?} is not contiguous. Generating copy loop.");

                let src_buffer = self.get_buffer_ptr(node_data.src[0]);
                let dst_buffer = self.get_buffer_ptr(node_id);
                let dst_tracker = ShapeTracker::new(node_data.shape.clone());

                let mut loops = vec![];
                let mut loop_vars = vec![];
                for shape_expr in dst_tracker.shape().iter() {
                    let loop_var = self.new_loop_counter();
                    loop_vars.push(loop_var.clone());
                    loops.push(AstNode::range(
                        loop_var,
                        shape_expr.clone().into(),
                        AstNode::block(vec![]),
                    ));
                }

                let src_offset = src_tracker.offset_expr(&loop_vars);
                let dst_offset = dst_tracker.offset_expr(&loop_vars);

                let load_node =
                    AstNode::deref(src_buffer.buffer_index(src_offset.simplify().into()));
                let store_node = AstNode::store(
                    dst_buffer.buffer_index(dst_offset.simplify().into()),
                    load_node,
                );

                let final_block = AstNode::build_loops(loops, store_node);

                // The result is a block containing the copy loop.
                (final_block, dst_tracker)
            }
            TensorOp::Permute(axes) => {
                let (src_ast, src_tracker) = self.lower_node(node_data.src[0]);
                let new_tracker = src_tracker.permute(axes);
                (src_ast, new_tracker)
            }
            TensorOp::Squeeze(axis) => {
                let (src_ast, src_tracker) = self.lower_node(node_data.src[0]);
                let new_tracker = src_tracker.squeeze(axis);
                (src_ast, new_tracker)
            }
            TensorOp::Unsqueeze(axis) => {
                let (src_ast, src_tracker) = self.lower_node(node_data.src[0]);
                let new_tracker = src_tracker.unsqueeze(axis);
                (src_ast, new_tracker)
            }
            TensorOp::Expand(new_shape) => {
                let (src_ast, src_tracker) = self.lower_node(node_data.src[0]);
                let new_tracker = src_tracker.expand(new_shape);
                (src_ast, new_tracker)
            }
            TensorOp::Elementwise(op) => {
                for &src_id in &node_data.src {
                    self.lower_node(src_id);
                }

                let dst_buffer = self.get_buffer_ptr(node_id);
                let dst_tracker = ShapeTracker::new(node_data.shape.clone());

                let mut loops = vec![];
                let mut loop_vars = vec![];
                for shape_expr in dst_tracker.shape().iter() {
                    let loop_var = self.new_loop_counter();
                    loop_vars.push(loop_var.clone());
                    loops.push(AstNode::range(
                        loop_var,
                        shape_expr.clone().into(),
                        AstNode::block(vec![]),
                    ));
                }

                let mut loaded_srcs = vec![];
                for &src_id in node_data.src.iter() {
                    let (_, tracker) = self.cache.get(&src_id).unwrap();
                    let buffer = self.get_buffer_ptr(src_id);
                    let offset = tracker.offset_expr(&loop_vars);
                    let load = AstNode::deref(buffer.buffer_index(offset.simplify().into()));
                    loaded_srcs.push(Box::new(load));
                }

                let op_node = AstNode::new(op, loaded_srcs, node_data.dtype.clone());
                let dst_offset = dst_tracker.offset_expr(&loop_vars);
                let store_node = AstNode::store(
                    dst_buffer.buffer_index(dst_offset.simplify().into()),
                    op_node,
                );

                let final_block = AstNode::build_loops(loops, store_node);

                (final_block, dst_tracker)
            }
            TensorOp::FusedElementwise(fused_ast) => {
                for &src_id in &node_data.src {
                    self.lower_node(src_id);
                }

                let dst_buffer = self.get_buffer_ptr(node_id);
                let dst_tracker = ShapeTracker::new(node_data.shape.clone());

                let mut loops = vec![];
                let mut loop_vars = vec![];
                for shape_expr in dst_tracker.shape().iter() {
                    let loop_var = self.new_loop_counter();
                    loop_vars.push(loop_var.clone());
                    loops.push(AstNode::range(
                        loop_var,
                        shape_expr.clone().into(),
                        AstNode::block(vec![]),
                    ));
                }

                let mut loaded_srcs = vec![];
                for &src_id in node_data.src.iter() {
                    let (_, tracker) = self.cache.get(&src_id).unwrap();
                    let buffer = self.get_buffer_ptr(src_id);
                    let offset = tracker.offset_expr(&loop_vars);
                    let load = AstNode::deref(buffer.buffer_index(offset.simplify().into()));
                    loaded_srcs.push(load);
                }

                let computation = Self::lower_fused_ast(&fused_ast, &loaded_srcs);
                let dst_offset = dst_tracker.offset_expr(&loop_vars);
                let store_node = AstNode::store(
                    dst_buffer.buffer_index(dst_offset.simplify().into()),
                    computation,
                );

                let final_block = AstNode::build_loops(loops, store_node);
                (final_block, dst_tracker)
            }
            TensorOp::Reduce(op, axis) => {
                self.lower_node(node_data.src[0]);

                let src_buffer = self.get_buffer_ptr(node_data.src[0]);
                let dst_buffer = self.get_buffer_ptr(node_id);
                let (_, src_tracker) = self.cache.get(&node_data.src[0]).unwrap().clone();
                let dst_tracker = ShapeTracker::new(node_data.shape.clone());

                let mut loops = vec![];
                let mut outer_loop_vars = vec![];
                for shape_expr in dst_tracker.shape().iter() {
                    let loop_var = self.new_loop_counter();
                    outer_loop_vars.push(loop_var.clone());
                    loops.push(AstNode::range(
                        loop_var,
                        shape_expr.clone().into(),
                        AstNode::block(vec![]),
                    ));
                }

                let acc_var = self.new_accumulator_name();
                let init_val = match op {
                    AstOp::Add => AstNode::from(0.0f32),
                    AstOp::Mul => AstNode::from(1.0f32),
                    AstOp::Max => AstNode::from(f32::NEG_INFINITY),
                    _ => unimplemented!("Unsupported reduce op"),
                };
                let init_acc = AstNode::assign(AstNode::var(&acc_var), init_val);

                let inner_loop_var = self.new_loop_counter();
                let mut full_indices = outer_loop_vars.clone();
                full_indices.insert(axis, inner_loop_var.clone());
                let src_offset = src_tracker.offset_expr(&full_indices);
                let load_val =
                    AstNode::deref(src_buffer.buffer_index(src_offset.simplify().into()));

                let update_acc = AstNode::assign(
                    AstNode::var(&acc_var),
                    AstNode::new(
                        op,
                        vec![Box::new(AstNode::var(&acc_var)), Box::new(load_val)],
                        node_data.dtype.clone(),
                    ),
                );

                let inner_loop = AstNode::range(
                    inner_loop_var,
                    src_tracker.shape()[axis].clone().into(),
                    update_acc,
                );

                let dst_offset = dst_tracker.offset_expr(&outer_loop_vars);
                let store_result = AstNode::store(
                    dst_buffer.buffer_index(dst_offset.simplify().into()),
                    AstNode::var(&acc_var),
                );

                let reduction_block = AstNode::block(vec![init_acc, inner_loop, store_result]);
                let final_block = AstNode::build_loops(loops, reduction_block);

                (final_block, dst_tracker)
            }
            _ => unimplemented!("This TensorOp is not yet supported for lowering"),
        };

        trace!("Finished lowering node {node_id:?}. Caching result.");
        self.cache.insert(node_id, result.clone());
        result
    }

    /// Recursively expands a fused AST, replacing captures with loaded values.
    fn lower_fused_ast(ast: &AstNode, loaded_srcs: &[AstNode]) -> AstNode {
        match &ast.op {
            AstOp::Capture(id, _) => {
                // Replace the capture node with the corresponding pre-loaded source AST.
                loaded_srcs[*id].clone()
            }
            _ => {
                // For any other node, recursively lower its children.
                let new_srcs = ast
                    .src
                    .iter()
                    .map(|child| Box::new(Self::lower_fused_ast(child, loaded_srcs)))
                    .collect();
                AstNode::new(ast.op.clone(), new_srcs, ast.dtype.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        if let AstOp::Func { name, args, body } = ast.op {
            assert_eq!(name, "kernel_main");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0].0, "buffers");
            assert_eq!(args[1].0, "shape_vars");
            assert!(matches!(body.op, AstOp::Block));
        } else {
            panic!("Expected a FuncDef node, found {:?}", ast.op);
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
        // println!("{}", ast.pretty_print()); // For debugging

        // Check that the body of the function contains a store to buffers[2]
        if let AstOp::Func { body, .. } = ast.op {
            // body -> Block -> [elementwise_ast, reduce_ast, ...]
            let elementwise_ast = &body.src[0];
            if let AstOp::Range { block, .. } = &elementwise_ast.op {
                if let AstOp::Store { dst, src } = &block.op {
                    // dst should be something like `((float*)buffers[2])[ridx0]`
                    if let AstOp::BufferIndex { buffer, .. } = &dst.op {
                        if let AstOp::Cast(..) = &buffer.op {
                            // Correct structure
                        } else {
                            panic!("Destination buffer not a cast");
                        }
                    } else {
                        panic!("Destination not a buffer index");
                    }

                    // src should be `... + ...`
                    assert!(matches!(src.op, AstOp::Add));
                } else {
                    panic!("Expected store node, found {:?}", block.op);
                }
            } else {
                panic!("Expected range node, found {:?}", elementwise_ast.op);
            }
        } else {
            panic!("Expected FuncDef");
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
        // println!("{}", ast.pretty_print()); // For debugging

        if let AstOp::Func { body, .. } = ast.op {
            let reduce_ast = &body.src[0];
            if let AstOp::Range { block, .. } = &reduce_ast.op {
                // block contains [init_acc, inner_loop, store_result]
                let store_node = &block.src[2];
                if let AstOp::Store { dst, .. } = &store_node.op {
                    if let AstOp::BufferIndex { buffer, .. } = &dst.op {
                        // We are checking if the destination is a pointer cast from the buffers array
                        if let AstOp::Cast(..) = &buffer.op {
                            // This is the correct structure
                        } else {
                            panic!("Expected destination to be a cast from buffers array");
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
            panic!("Expected FuncDef");
        }
    }
}
