use crate::ast::{AstNode, DType, Op as AstOp};
use crate::tensor::graph::{Graph, NodeId, TensorOp};
use crate::tensor::shape::expr::Expr;
use crate::tensor::shape::tracker::ShapeTracker;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Lowerer<'a> {
    graph: &'a Graph,
    cache: HashMap<NodeId, AstNode>,
    var_counter: usize,
}

impl<'a> Lowerer<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            cache: HashMap::new(),
            var_counter: 0,
        }
    }

    fn new_loop_counter(&mut self) -> String {
        let name = format!("ridx{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    fn new_temp_var(&mut self) -> String {
        let name = format!("var{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    fn new_buffer_name(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.var_counter);
        self.var_counter += 1;
        name
    }

    pub fn lower(&mut self) -> AstNode {
        let mut body = vec![];
        for &output_id in self.graph.outputs.borrow().iter() {
            let ast_node = self.lower_node(output_id);
            body.push(ast_node);
        }
        AstNode::new(AstOp::Block(body), vec![], DType::None)
    }

    fn lower_node(&mut self, node_id: NodeId) -> AstNode {
        if let Some(cached) = self.cache.get(&node_id) {
            return cached.clone();
        }

        let node_data = self.graph.nodes.borrow()[node_id.0].clone();
        let ast_node = match node_data.op {
            TensorOp::Input => {
                let name = self.new_buffer_name("input");
                AstNode::var(&name).with_type(node_data.dtype)
            }
            TensorOp::Elementwise(op) => {
                assert_eq!(node_data.src.len(), 2, "Elementwise op must have 2 sources");
                let lhs_node = self.lower_node(node_data.src[0]);
                let rhs_node = self.lower_node(node_data.src[1]);

                let total_elements = node_data
                    .shape
                    .iter()
                    .fold(Expr::from(1), |acc, val| acc * val.clone())
                    .simplify();
                let max_index: AstNode = total_elements.into();

                let loop_var = self.new_loop_counter();
                let loop_var_node = AstNode::var(&loop_var);

                let lhs_access = AstNode::new(
                    AstOp::BufferIndex {
                        buffer: Box::new(lhs_node),
                        index: Box::new(loop_var_node.clone()),
                    },
                    vec![],
                    DType::None,
                );
                let rhs_access = AstNode::new(
                    AstOp::BufferIndex {
                        buffer: Box::new(rhs_node),
                        index: Box::new(loop_var_node.clone()),
                    },
                    vec![],
                    DType::None,
                );

                let op_node = AstNode::new(
                    op,
                    vec![
                        Box::new(AstNode::new(
                            AstOp::Load(Box::new(lhs_access)),
                            vec![],
                            node_data.dtype.clone(),
                        )),
                        Box::new(AstNode::new(
                            AstOp::Load(Box::new(rhs_access)),
                            vec![],
                            node_data.dtype.clone(),
                        )),
                    ],
                    node_data.dtype.clone(),
                );

                let output_buffer = AstNode::var(&self.new_buffer_name("output"));
                let store_node = AstNode::new(
                    AstOp::Store {
                        dst: Box::new(AstNode::new(
                            AstOp::BufferIndex {
                                buffer: Box::new(output_buffer),
                                index: Box::new(loop_var_node),
                            },
                            vec![],
                            DType::None,
                        )),
                        src: Box::new(op_node),
                    },
                    vec![],
                    DType::None,
                );

                AstNode::new(
                    AstOp::Range {
                        loop_var,
                        max: Box::new(max_index),
                        block: Box::new(store_node),
                    },
                    vec![],
                    DType::None,
                )
            }
            TensorOp::Reduce(op, axis) => {
                let src_node_id = node_data.src[0];
                let src_node_data = self.graph.nodes.borrow()[src_node_id.0].clone();
                let src_tracker = ShapeTracker::new(src_node_data.shape);
                let dst_tracker = ShapeTracker::new(node_data.shape.clone());

                let src_buffer = self.lower_node(src_node_id);

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
                            block: Box::new(AstNode::new(
                                AstOp::Block(vec![]),
                                vec![],
                                DType::None,
                            )), // Placeholder
                        },
                        vec![],
                        DType::None,
                    ));
                    src_index_expr +=
                        Expr::from(AstNode::var(&loop_var)) * src_tracker.strides()[i].clone();
                }

                let inner_loop_var = self.new_loop_counter();
                src_index_expr +=
                    Expr::from(AstNode::var(&inner_loop_var)) * src_tracker.strides()[axis].clone();

                let acc_var = self.new_temp_var();
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

                let output_buffer = AstNode::var(&self.new_buffer_name("output"));
                let store_result = AstNode::new(
                    AstOp::Store {
                        dst: Box::new(AstNode::new(
                            AstOp::BufferIndex {
                                buffer: Box::new(output_buffer),
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
                    AstOp::Block(vec![init_acc, inner_loop, store_result]),
                    vec![],
                    DType::None,
                );

                for mut loop_node in loops.into_iter().rev() {
                    if let AstOp::Range { ref mut block, .. } = loop_node.op {
                        *block = Box::new(final_block);
                    }
                    final_block = loop_node;
                }

                final_block
            }
            _ => unimplemented!("This TensorOp is not yet supported for lowering"),
        };

        self.cache.insert(node_id, ast_node.clone());
        ast_node
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
        let ast = lowerer.lower();

        if let AstOp::Block(body) = ast.op {
            assert_eq!(body.len(), 1);
            let input_ast = &body[0];
            assert_eq!(input_ast.op, AstOp::Var("input0".to_string()));
            assert_eq!(input_ast.dtype, DType::F32);
        } else {
            panic!("Expected a block, found {:?}", ast.op);
        }
    }

    #[test]
    fn test_lower_elementwise_add() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(10)]);
        (a + b).as_output();

        let mut lowerer = Lowerer::new(&graph);
        let ast = lowerer.lower();

        // For debugging:
        // println!("{}", ast.pretty_print());

        if let AstOp::Block(body) = ast.op {
            assert_eq!(body.len(), 1);
            let range_node = &body[0];
            if let AstOp::Range { loop_var, .. } = &range_node.op {
                assert_eq!(loop_var, "ridx2"); // input0, input1, then ridx2
            } else {
                panic!("Expected a range node, found {:?}", range_node.op);
            }
        } else {
            panic!("Expected a block, found {:?}", ast.op);
        }
    }

    #[test]
    fn test_lower_elementwise_add_2d() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10), Expr::from(20)]);
        let b = graph.input(DType::F32, vec![Expr::from(10), Expr::from(20)]);
        (a + b).as_output();

        let mut lowerer = Lowerer::new(&graph);
        let ast = lowerer.lower();

        if let AstOp::Block(body) = ast.op {
            assert_eq!(body.len(), 1);
            let range_node = &body[0];
            if let AstOp::Range { max, .. } = &range_node.op {
                if let AstOp::Const(c) = &max.op {
                    if let crate::ast::Const::I64(v) = c {
                        assert_eq!(*v, 200);
                    } else {
                        panic!("Expected I64 constant");
                    }
                } else {
                    panic!("Expected a constant for loop max, found {:?}", max.op);
                }
            } else {
                panic!("Expected a range node, found {:?}", range_node.op);
            }
        } else {
            panic!("Expected a block, found {:?}", ast.op);
        }
    }

    #[test]
    fn test_lower_reduce_sum_2d() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10), Expr::from(20)]);
        a.sum(1).as_output(); // Reduce axis 1

        let mut lowerer = Lowerer::new(&graph);
        let ast = lowerer.lower();

        // For debugging:
        // println!("{}", ast.pretty_print());

        if let AstOp::Block(body) = ast.op {
            assert_eq!(body.len(), 1);
            let outer_range = &body[0];
            if let AstOp::Range {
                loop_var,
                max,
                block,
            } = &outer_range.op
            {
                assert_eq!(loop_var, "ridx1");
                assert_eq!(max.op, AstOp::Const(crate::ast::Const::I64(10)));

                if let AstOp::Block(inner_block_nodes) = &block.op {
                    assert_eq!(inner_block_nodes.len(), 3); // init, loop, store
                    let inner_range = &inner_block_nodes[1];
                    if let AstOp::Range { loop_var, max, .. } = &inner_range.op {
                        assert_eq!(loop_var, "ridx2");
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
        } else {
            panic!("Expected a block");
        }
    }
}
