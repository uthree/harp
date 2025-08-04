use crate::ast::{AstNode, DType, Op as AstOp};
use crate::tensor::graph::{Graph, NodeId, TensorOp};
use crate::tensor::shape::expr::Expr;
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

    fn new_var(&mut self, name: &str) -> String {
        let var_name = format!("{}_{}", name, self.var_counter);
        self.var_counter += 1;
        var_name
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
                let name = self.new_var("input");
                AstNode::var(&name).with_type(node_data.dtype)
            }
            TensorOp::Elementwise(op) => {
                assert_eq!(node_data.src.len(), 2, "Elementwise op must have 2 sources");
                let lhs_node = self.lower_node(node_data.src[0]);
                let rhs_node = self.lower_node(node_data.src[1]);

                // TODO: Handle multi-dimensional shapes and strides
                let total_elements = node_data
                    .shape
                    .iter()
                    .fold(Expr::from(1), |acc, val| acc * val.clone())
                    .simplify();
                let max_index: AstNode = total_elements.into();

                let loop_var = self.new_var("i");
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

                let output_buffer = AstNode::var(&self.new_var("output"));
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
                unimplemented!("Reduce lowering is not yet implemented")
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
            assert_eq!(input_ast.op, AstOp::Var("input_0".to_string()));
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
                assert_eq!(loop_var, "i_2"); // a, b, then loop_var
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
}
