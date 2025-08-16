use crate::ast::{AstNode, AstOp, DType};
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::HashMap;

pub struct Lowerer {
    memo: HashMap<GraphNode, AstNode>,
    var_counter: usize,
}

impl Lowerer {
    fn new() -> Self {
        Lowerer {
            memo: HashMap::new(),
            var_counter: 0,
        }
    }

    fn new_var(&mut self) -> String {
        let var_name = format!("v{}", self.var_counter);
        self.var_counter += 1;
        var_name
    }

    fn lower_node(&mut self, node: &GraphNode) -> (AstNode, Vec<AstNode>) {
        if let Some(cached) = self.memo.get(node) {
            return (cached.clone(), vec![]);
        }

        let mut prelude = vec![];

        let result_ast = match &node.op {
            GraphOp::Input { .. } => {
                panic!("Input node should be memoized before lowering");
            }
            GraphOp::Elementwise(op) => {
                let mut src_asts = vec![];
                for src_node in &node.src {
                    let (src_ast, src_prelude) = self.lower_node(src_node);
                    prelude.extend(src_prelude);
                    src_asts.push(src_ast);
                }
                AstNode::_new(op.clone(), src_asts, node.dtype.clone())
            }
            GraphOp::Reduce(op, _axis) => {
                let (src_ast, src_prelude) = self.lower_node(&node.src[0]);
                prelude.extend(src_prelude);

                let acc_name = self.new_var();
                let acc_var = AstNode::var(&acc_name);
                let acc_init =
                    AstNode::declare(&acc_name, node.dtype.clone(), Some(node.dtype.zero()));
                prelude.push(acc_init);

                let loop_var = "i".to_string();
                let loop_body = AstNode::assign(
                    acc_var.clone(),
                    AstNode::_new(
                        op.clone(),
                        vec![acc_var, src_ast.clone()], // Simplified
                        node.dtype.clone(),
                    ),
                );

                let loop_node = AstNode::_new(
                    AstOp::Range {
                        counter: loop_var,
                        step: 1,
                    },
                    vec![
                        AstNode::from(10isize), // Simplified
                        loop_body,
                    ],
                    DType::Any,
                );
                prelude.push(loop_node);

                AstNode::var(&acc_name)
            }
            GraphOp::Contiguous => {
                let (src_ast, src_prelude) = self.lower_node(&node.src[0]);
                prelude.extend(src_prelude);
                src_ast
            }
        };

        self.memo.insert(node.clone(), result_ast.clone());
        (result_ast, prelude)
    }
}

pub fn lower_graph(graph: &Graph) -> AstNode {
    let mut lowerer = Lowerer::new();
    let mut func_args = vec![];
    let mut func_body = vec![];

    for (i, input_node) in graph.inputs.iter().enumerate() {
        let input_sig = &graph.signature.inputs[i];
        let arg_name = format!("input_{}", i);
        let arg_dtype = DType::Ptr(Box::new(input_sig.dtype.clone()));
        func_args.push((arg_name.clone(), arg_dtype.clone()));
        lowerer
            .memo
            .insert(input_node.clone(), AstNode::var(&arg_name));
    }

    let output_arg_names: Vec<String> = (0..graph.outputs.len())
        .map(|i| format!("output_{}", i))
        .collect();

    for (i, output_sig) in graph.signature.outputs.iter().enumerate() {
        let arg_dtype = DType::Ptr(Box::new(output_sig.dtype.clone()));
        func_args.push((output_arg_names[i].clone(), arg_dtype));
    }

    for (i, output_node) in graph.outputs.iter().enumerate() {
        let (result_ast, prelude) = lowerer.lower_node(output_node);
        func_body.extend(prelude);

        let output_ptr = AstNode::var(&output_arg_names[i]);
        let store_op = AstNode::store(output_ptr, result_ast);
        func_body.push(store_op);
    }

    AstNode::_new(
        AstOp::Func {
            name: "kernel_main".to_string(),
            args: func_args,
        },
        func_body,
        DType::Any,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorSignature;
    use crate::graph::shape::expr::Expr as ShapeExpr;

    #[test]
    fn test_lower_elementwise_add() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b;

        graph.outputs.push(c);
        graph.signature.outputs.push(TensorSignature {
            dtype: dtype.clone(),
            shape,
        });

        let ast = lower_graph(&graph);

        // Expected AST structure
        let expected_args = vec![
            ("input_0".to_string(), DType::Ptr(Box::new(DType::F32))),
            ("input_1".to_string(), DType::Ptr(Box::new(DType::F32))),
            ("output_0".to_string(), DType::Ptr(Box::new(DType::F32))),
        ];
        let expected_body = vec![AstNode::store(
            AstNode::var("output_0"),
            AstNode::_new(
                AstOp::Add,
                vec![AstNode::var("input_0"), AstNode::var("input_1")],
                DType::F32,
            ),
        )];

        if let AstOp::Func { name, args } = ast.op {
            assert_eq!(name, "kernel_main");
            assert_eq!(args, expected_args);
            assert_eq!(ast.src, expected_body);
        } else {
            panic!("Expected a function AST node");
        }
    }
}
