use crate::ast::{AstNode, AstOp, DType};
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::HashMap;

pub struct Lowerer {
    memo: HashMap<GraphNode, AstNode>,
    var_counter: usize,
    ridx_counter: usize,
    acc_counter: usize,
}

impl Lowerer {
    fn new() -> Self {
        Lowerer {
            memo: HashMap::new(),
            var_counter: 0,
            ridx_counter: 0,
            acc_counter: 0,
        }
    }

    fn new_var_name(&mut self) -> String {
        let name = format!("var{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    fn new_ridx_name(&mut self) -> String {
        let name = format!("ridx{}", self.ridx_counter);
        self.ridx_counter += 1;
        name
    }

    fn new_acc_name(&mut self) -> String {
        let name = format!("acc{}", self.acc_counter);
        self.acc_counter += 1;
        name
    }

    fn lower_node(&mut self, node: &GraphNode) -> (AstNode, Vec<AstNode>) {
        if let Some(cached) = self.memo.get(node) {
            return (cached.clone(), vec![]);
        }

        let mut prelude = vec![];
        let result_ast = match &node.op {
            GraphOp::Input { .. } => panic!("Input node should be memoized"),
            GraphOp::Elementwise(op) => {
                let mut src_asts = vec![];
                for src_node in &node.src {
                    let (mut src_ast, src_prelude) = self.lower_node(src_node);
                    prelude.extend(src_prelude);
                    if let DType::Ptr(_) = src_ast.dtype {
                        src_ast = AstNode::index(src_ast, AstNode::from(0isize));
                    }
                    src_asts.push(src_ast);
                }
                AstNode::_new(op.clone(), src_asts, node.dtype.clone())
            }
            GraphOp::Reduce(op, axis) => {
                let (src_ast, src_prelude) = self.lower_node(&node.src[0]);
                prelude.extend(src_prelude);

                let acc_name = self.new_acc_name();
                let acc_var = AstNode::var(&acc_name, node.dtype.clone());
                prelude.push(AstNode::declare(
                    &acc_name,
                    node.dtype.clone(),
                    Some(node.dtype.zero()),
                ));

                let loop_var_name = self.new_ridx_name();
                let loop_var = AstNode::var(&loop_var_name, DType::Usize);
                let loop_body = AstNode::assign(
                    acc_var.clone(),
                    AstNode::_new(
                        op.clone(),
                        vec![acc_var, AstNode::index(src_ast, loop_var)],
                        node.dtype.clone(),
                    ),
                );

                prelude.push(AstNode::_new(
                    AstOp::Range {
                        counter: loop_var_name,
                        step: 1,
                    },
                    vec![AstNode::from(node.src[0].shape()[*axis].clone()), loop_body],
                    DType::Any,
                ));
                AstNode::var(&acc_name, node.dtype.clone())
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
    let mut kernel_impl_args = vec![];
    let mut kernel_impl_body = vec![];

    let num_inputs = graph.inputs.len();
    let num_outputs = graph.outputs.len();
    let buffer_names: Vec<String> = (0..num_inputs + num_outputs)
        .map(|i| format!("buf{}", i))
        .collect();

    // Outputs first
    for (i, _output_node) in graph.outputs.iter().enumerate() {
        let sig = &graph.signature.outputs[i];
        let dtype = DType::Ptr(Box::new(sig.dtype.clone()));
        kernel_impl_args.push((buffer_names[i].clone(), dtype.clone()));
    }

    // Then inputs
    for (i, input_node) in graph.inputs.iter().enumerate() {
        let sig = &graph.signature.inputs[i];
        let dtype = DType::Ptr(Box::new(sig.dtype.clone()));
        kernel_impl_args.push((buffer_names[num_outputs + i].clone(), dtype.clone()));
        lowerer.memo.insert(
            input_node.clone(),
            AstNode::var(&buffer_names[num_outputs + i], dtype),
        );
    }

    for (i, _output_node) in graph.outputs.iter().enumerate() {
        let (result_ast, prelude) = lowerer.lower_node(&graph.outputs[i]);
        kernel_impl_body.extend(prelude);

        let ptr = AstNode::var(&buffer_names[i], kernel_impl_args[i].1.clone());
        let index = AstNode::from(0isize);
        kernel_impl_body.push(AstNode::store(AstNode::index(ptr, index), result_ast));
    }

    let kernel_impl = AstNode::_new(
        AstOp::Func {
            name: "kernel_impl".to_string(),
            args: kernel_impl_args,
        },
        kernel_impl_body,
        DType::Any,
    );

    let bufs_arg = (
        "bufs".to_string(),
        DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Any)))),
    );
    let shape_vars_arg = ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize)));
    let mut main_body = vec![];
    let mut call_args = vec![];

    if let AstOp::Func { args, .. } = &kernel_impl.op {
        for i in 0..num_inputs + num_outputs {
            let ptr_to_buf_i = AstNode::index(
                AstNode::var("bufs", bufs_arg.1.clone()),
                AstNode::from(i as isize),
            );
            let cast_buf_i = AstNode::_new(
                AstOp::Cast(args[i].1.clone()),
                vec![ptr_to_buf_i],
                args[i].1.clone(),
            );
            call_args.push(cast_buf_i);
        }
    }

    main_body.push(AstNode::_new(
        AstOp::Call("kernel_impl".to_string()),
        call_args,
        DType::Any,
    ));

    let kernel_main = AstNode::_new(
        AstOp::Func {
            name: "kernel_main".to_string(),
            args: vec![bufs_arg, shape_vars_arg],
        },
        main_body,
        DType::Any,
    );

    AstNode::_new(AstOp::Program, vec![kernel_impl, kernel_main], DType::Any)
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

        assert!(matches!(ast.op, AstOp::Program));
        assert_eq!(ast.src.len(), 2);

        let kernel_impl = &ast.src[0];
        let expected_impl_args = vec![
            ("buf0".to_string(), DType::Ptr(Box::new(DType::F32))), // output
            ("buf1".to_string(), DType::Ptr(Box::new(DType::F32))), // input a
            ("buf2".to_string(), DType::Ptr(Box::new(DType::F32))), // input b
        ];
        if let AstOp::Func { name, args } = &kernel_impl.op {
            assert_eq!(name, "kernel_impl");
            assert_eq!(*args, expected_impl_args);
        } else {
            panic!("Expected kernel_impl function");
        }
    }
}
