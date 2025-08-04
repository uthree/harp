use crate::ast::{AstNode, DType, Op as AstOp};
use crate::tensor::graph::{Graph, NodeId, TensorOp};
use crate::tensor::shape::expr::Expr;
use crate::tensor::shape::tracker::ShapeTracker;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Lowerer<'a> {
    graph: &'a Graph,
    cache: HashMap<NodeId, (AstNode, ShapeTracker)>,
    loop_counter: usize,
    temp_var_counter: usize,
    accumulator_counter: usize,
    buffer_counter: usize,
}

impl<'a> Lowerer<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            cache: HashMap::new(),
            loop_counter: 0,
            temp_var_counter: 0,
            accumulator_counter: 0,
            buffer_counter: 0,
        }
    }

    fn new_loop_counter(&mut self) -> String {
        let name = format!("ridx{}", self.loop_counter);
        self.loop_counter += 1;
        name
    }

    fn new_temp_var(&mut self) -> String {
        let name = format!("var{}", self.temp_var_counter);
        self.temp_var_counter += 1;
        name
    }

    fn new_accumulator_name(&mut self) -> String {
        let name = format!("acc{}", self.accumulator_counter);
        self.accumulator_counter += 1;
        name
    }

    fn new_buffer_name(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.buffer_counter);
        self.buffer_counter += 1;
        name
    }

    pub fn lower(&mut self) -> AstNode {
        let mut body = vec![];
        for &output_id in self.graph.outputs.borrow().iter() {
            let (ast_node, _tracker) = self.lower_node(output_id);
            body.push(ast_node);
        }
        AstNode::new(
            AstOp::Block,
            body.into_iter().map(Box::new).collect(),
            DType::None,
        )
    }

    fn lower_node(&mut self, node_id: NodeId) -> (AstNode, ShapeTracker) {
        if let Some(cached) = self.cache.get(&node_id) {
            return cached.clone();
        }

        let node_data = self.graph.nodes.borrow()[node_id.0].clone();
        let result = match node_data.op {
            TensorOp::Input => {
                let name = self.new_buffer_name("input");
                let buffer_node = AstNode::var(&name).with_type(node_data.dtype);
                let tracker = ShapeTracker::new(node_data.shape);
                (buffer_node, tracker)
            }
            TensorOp::Permute(axes) => {
                let (src_buffer, src_tracker) = self.lower_node(node_data.src[0]);
                let new_tracker = src_tracker.permute(axes);
                (src_buffer, new_tracker)
            }
            TensorOp::Elementwise(op) => {
                let (lhs_buffer, lhs_tracker) = self.lower_node(node_data.src[0]);
                let (rhs_buffer, rhs_tracker) = self.lower_node(node_data.src[1]);

                let dst_buffer = AstNode::var(&self.new_buffer_name("output"));
                let dst_tracker = ShapeTracker::new(node_data.shape.clone());

                let mut loops = vec![];
                let mut loop_vars = vec![];
                for shape_expr in dst_tracker.shape().iter() {
                    let loop_var = self.new_loop_counter();
                    loop_vars.push(loop_var.clone());
                    loops.push(AstNode::new(
                        AstOp::Range {
                            loop_var,
                            max: Box::new(shape_expr.clone().into()),
                            block: Box::new(AstNode::new(AstOp::Block, vec![], DType::None)), // Placeholder
                        },
                        vec![],
                        DType::None,
                    ));
                }

                let calculate_offset = |tracker: &ShapeTracker| -> Expr {
                    loop_vars
                        .iter()
                        .zip(tracker.strides().iter())
                        .fold(0.into(), |acc, (var, stride)| {
                            acc + Expr::from(AstNode::var(var)) * stride.clone()
                        })
                };

                let lhs_offset = calculate_offset(&lhs_tracker);
                let rhs_offset = calculate_offset(&rhs_tracker);
                let dst_offset = calculate_offset(&dst_tracker);

                let lhs_load = AstNode::new(
                    AstOp::Load(Box::new(AstNode::new(
                        AstOp::BufferIndex {
                            buffer: Box::new(lhs_buffer),
                            index: Box::new(lhs_offset.simplify().into()),
                        },
                        vec![],
                        DType::None,
                    ))),
                    vec![],
                    node_data.dtype.clone(),
                );

                let rhs_load = AstNode::new(
                    AstOp::Load(Box::new(AstNode::new(
                        AstOp::BufferIndex {
                            buffer: Box::new(rhs_buffer),
                            index: Box::new(rhs_offset.simplify().into()),
                        },
                        vec![],
                        DType::None,
                    ))),
                    vec![],
                    node_data.dtype.clone(),
                );

                let op_node = AstNode::new(
                    op,
                    vec![Box::new(lhs_load), Box::new(rhs_load)],
                    node_data.dtype.clone(),
                );

                let store_node = AstNode::new(
                    AstOp::Store {
                        dst: Box::new(AstNode::new(
                            AstOp::BufferIndex {
                                buffer: Box::new(dst_buffer.clone()),
                                index: Box::new(dst_offset.simplify().into()),
                            },
                            vec![],
                            DType::None,
                        )),
                        src: Box::new(op_node),
                    },
                    vec![],
                    DType::None,
                );

                let mut final_block = store_node;
                for mut loop_node in loops.into_iter().rev() {
                    if let AstOp::Range { ref mut block, .. } = loop_node.op {
                        *block = Box::new(final_block);
                    }
                    final_block = loop_node;
                }

                (final_block, dst_tracker)
            }
            TensorOp::Reduce(op, axis) => {
                let src_node_id = node_data.src[0];
                let (src_buffer, src_tracker) = self.lower_node(src_node_id);

                let dst_buffer = AstNode::var(&self.new_buffer_name("output"));
                let dst_tracker = ShapeTracker::new(node_data.shape.clone());

                let mut loops = vec![];
                let mut outer_loop_vars = vec![];
                let mut src_index_expr: Expr = 0.into();

                for (i, shape_expr) in dst_tracker.shape().iter().enumerate() {
                    let loop_var = self.new_loop_counter();
                    outer_loop_vars.push(loop_var.clone());
                    loops.push(AstNode::new(
                        AstOp::Range {
                            loop_var: loop_var.clone(),
                            max: Box::new(shape_expr.clone().into()),
                            block: Box::new(AstNode::new(AstOp::Block, vec![], DType::None)), // Placeholder
                        },
                        vec![],
                        DType::None,
                    ));
                    // Adjust stride index for reduction
                    let stride_idx = if i < axis { i } else { i + 1 };
                    src_index_expr += Expr::from(AstNode::var(&loop_var))
                        * src_tracker.strides()[stride_idx].clone();
                }

                let inner_loop_var = self.new_loop_counter();
                src_index_expr +=
                    Expr::from(AstNode::var(&inner_loop_var)) * src_tracker.strides()[axis].clone();

                let acc_var = self.new_accumulator_name();
                let init_val = match op {
                    AstOp::Add => AstNode::from(0.0f32), // Assuming F32 for now
                    AstOp::Mul => AstNode::from(1.0f32),
                    AstOp::Max => AstNode::from(f32::NEG_INFINITY),
                    _ => unimplemented!("Unsupported reduce op"),
                };

                let init_acc = AstNode::new(
                    AstOp::Assign {
                        dst: Box::new(AstNode::var(&acc_var)),
                        src: Box::new(init_val),
                    },
                    vec![],
                    node_data.dtype.clone(),
                );

                let load_val = AstNode::new(
                    AstOp::Load(Box::new(AstNode::new(
                        AstOp::BufferIndex {
                            buffer: Box::new(src_buffer.clone()),
                            index: Box::new(src_index_expr.simplify().into()),
                        },
                        vec![],
                        DType::None,
                    ))),
                    vec![],
                    node_data.dtype.clone(),
                );

                let update_acc = AstNode::new(
                    AstOp::Assign {
                        dst: Box::new(AstNode::var(&acc_var)),
                        src: Box::new(AstNode::new(
                            op,
                            vec![Box::new(AstNode::var(&acc_var)), Box::new(load_val)],
                            node_data.dtype.clone(),
                        )),
                    },
                    vec![],
                    node_data.dtype.clone(),
                );

                let inner_loop = AstNode::new(
                    AstOp::Range {
                        loop_var: inner_loop_var,
                        max: Box::new(src_tracker.shape()[axis].clone().into()),
                        block: Box::new(update_acc),
                    },
                    vec![],
                    DType::None,
                );

                let dst_index_expr: Expr = outer_loop_vars
                    .iter()
                    .zip(dst_tracker.strides().iter())
                    .fold(0.into(), |acc, (var, stride)| {
                        acc + Expr::from(AstNode::var(var)) * stride.clone()
                    });

                let store_result = AstNode::new(
                    AstOp::Store {
                        dst: Box::new(AstNode::new(
                            AstOp::BufferIndex {
                                buffer: Box::new(dst_buffer.clone()),
                                index: Box::new(dst_index_expr.simplify().into()),
                            },
                            vec![],
                            DType::None,
                        )),
                        src: Box::new(AstNode::var(&acc_var)),
                    },
                    vec![],
                    DType::None,
                );

                let mut final_block = AstNode::new(
                    AstOp::Block,
                    vec![
                        Box::new(init_acc),
                        Box::new(inner_loop),
                        Box::new(store_result),
                    ],
                    DType::None,
                );

                for mut loop_node in loops.into_iter().rev() {
                    if let AstOp::Range { ref mut block, .. } = loop_node.op {
                        *block = Box::new(final_block);
                    }
                    final_block = loop_node;
                }

                (final_block, dst_tracker)
            }
            _ => unimplemented!("This TensorOp is not yet supported for lowering"),
        };

        self.cache.insert(node_id, result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::shape::expr::Expr;

    #[test]
    fn test_lower_input() {
        let graph = Graph::new();
        let input = graph.input(DType::F32, vec![Expr::from(10)]);
        input.as_output();

        let mut lowerer = Lowerer::new(&graph);
        lowerer.lower(); // Populate cache

        let (input_ast, tracker) = &lowerer.cache[&input.id];
        assert_eq!(input_ast.op, AstOp::Var("input0".to_string()));
        assert_eq!(input_ast.dtype, DType::F32);
        assert_eq!(tracker.shape(), &[Expr::from(10)]);
    }

    #[test]
    fn test_lower_permute() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let b = a.permute(vec![1, 0]);
        b.as_output();

        let mut lowerer = Lowerer::new(&graph);
        lowerer.lower(); // Populate cache

        let (_, tracker) = &lowerer.cache[&b.id];
        assert_eq!(tracker.shape(), &[20.into(), 10.into()]);
        assert_eq!(tracker.strides(), &[1.into(), 20.into()]);
    }

    #[test]
    fn test_lower_elementwise_add() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(10)]);
        let c = (a + b).as_output();

        let mut lowerer = Lowerer::new(&graph);
        lowerer.lower();

        let (ast, _) = &lowerer.cache[&c.id];

        if let AstOp::Range { loop_var, max, .. } = &ast.op {
            assert_eq!(loop_var, "ridx0");
            assert_eq!(max.op, AstOp::Const(crate::ast::Const::I64(10)));
        } else {
            panic!("Expected a range node, found {:?}", ast.op);
        }
    }

    #[test]
    fn test_lower_permuted_add() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let b = graph.input(DType::F32, vec![20.into(), 10.into()]);
        let a_p = a.permute(vec![1, 0]);
        let c = (a_p + b).as_output();

        let mut lowerer = Lowerer::new(&graph);
        lowerer.lower();

        let (ast, _) = &lowerer.cache[&c.id];
        // For debugging:
        // println!("{}", ast.pretty_print());

        // Check for outer loop
        if let AstOp::Range {
            loop_var,
            max,
            block,
        } = &ast.op
        {
            assert_eq!(loop_var, "ridx0");
            assert_eq!(max.op, AstOp::Const(crate::ast::Const::I64(20)));

            // Check for inner loop
            if let AstOp::Range { loop_var, max, .. } = &block.op {
                assert_eq!(loop_var, "ridx1");
                assert_eq!(max.op, AstOp::Const(crate::ast::Const::I64(10)));
            } else {
                panic!("Expected inner range node");
            }
        } else {
            panic!("Expected outer range node");
        }
    }

    #[test]
    fn test_lower_reduce_sum_2d() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10), Expr::from(20)]);
        let b = a.sum(1).as_output(); // Reduce axis 1

        let mut lowerer = Lowerer::new(&graph);
        lowerer.lower();

        let (ast, _) = &lowerer.cache[&b.id];
        // For debugging:
        // println!("{}", ast.pretty_print());

        if let AstOp::Range {
            loop_var,
            max,
            block,
        } = &ast.op
        {
            assert_eq!(loop_var, "ridx0");
            assert_eq!(max.op, AstOp::Const(crate::ast::Const::I64(10)));

            if let AstOp::Block = &block.op {
                let inner_block_nodes = &block.src;
                assert_eq!(inner_block_nodes.len(), 3); // init, loop, store
                let inner_range = &inner_block_nodes[1];
                if let AstOp::Range { loop_var, max, .. } = &inner_range.op {
                    assert_eq!(loop_var, "ridx1");
                    assert_eq!(max.op, AstOp::Const(crate::ast::Const::I64(20)));
                } else {
                    panic!("Expected inner range");
                }
            } else {
                panic!("Expected inner block");
            }
        } else {
            panic!("Expected outer range");
        }
    }
}
