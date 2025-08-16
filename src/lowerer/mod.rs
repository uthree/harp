use crate::ast::{AstNode, AstOp, DType};
use crate::graph::shape::view::View;
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::HashMap;

pub struct Lowerer {
    memo: HashMap<GraphNode, AstNode>,
}

impl Lowerer {
    fn new() -> Self {
        Lowerer {
            memo: HashMap::new(),
        }
    }

    fn lower_expr(&mut self, node: &GraphNode, indices: &[AstNode]) -> AstNode {
        if let Some(cached) = self.memo.get(node) {
            // Input nodes are memoized with their buffer name.
            // We need to calculate the physical index and load from it.
            if let GraphOp::Input { .. } = &node.op {
                let physical_index = node.view.to_physical_index_ast(indices);
                return AstNode::index(cached.clone(), physical_index);
            }
        }

        match &node.op {
            GraphOp::Input { .. } => panic!("Input node should be memoized"),
            GraphOp::Full(c) => AstNode::from(c.clone()),
            GraphOp::Elementwise(op) => {
                let src_asts: Vec<AstNode> = node
                    .src
                    .iter()
                    .map(|src| self.lower_expr(src, indices))
                    .collect();
                AstNode::_new(op.clone(), src_asts, node.dtype.clone())
            }
            GraphOp::Permute(axes) => {
                let mut permuted_indices = vec![AstNode::from(0isize); indices.len()];
                for (i, &axis) in axes.iter().enumerate() {
                    permuted_indices[axis] = indices[i].clone();
                }
                self.lower_expr(&node.src[0], &permuted_indices)
            }
            GraphOp::Reduce(_op, _axis) => {
                // This requires a more complex implementation with a reduction loop.
                // For now, we'll leave it unimplemented to focus on the view-aware lowering.
                todo!("Lowering for Reduce op is not implemented yet in the new Lowerer");
            }
            GraphOp::Contiguous => {
                // In the new model, Contiguous might mean we need to compute and store
                // the result in a temporary buffer. For now, we pass through.
                self.lower_expr(&node.src[0], indices)
            }
        }
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
        let buffer_name = buffer_names[num_outputs + i].clone();
        kernel_impl_args.push((buffer_name.clone(), dtype.clone()));
        lowerer
            .memo
            .insert(input_node.clone(), AstNode::var(&buffer_name, dtype));
    }

    for (i, output_node) in graph.outputs.iter().enumerate() {
        let output_ptr = AstNode::var(&buffer_names[i], kernel_impl_args[i].1.clone());

        let mut loops = vec![];
        let mut indices = vec![];
        for (dim, size) in output_node.shape().iter().enumerate() {
            let loop_var_name = format!("idx{}", dim);
            let loop_var = AstNode::var(&loop_var_name, DType::Usize);
            indices.push(loop_var);
            loops.push(AstNode::_new(
                AstOp::Range {
                    counter: loop_var_name,
                    step: 1,
                },
                vec![size.to_ast()],
                DType::Any,
            ));
        }

        let final_expr = lowerer.lower_expr(output_node, &indices);

        let physical_index =
            View::new_contiguous(output_node.shape().to_vec()).to_physical_index_ast(&indices);
        let store_op = AstNode::store(AstNode::index(output_ptr, physical_index), final_expr);

        // Nest loops
        let mut nested_loops = store_op;
        for mut loop_node in loops.into_iter().rev() {
            loop_node.src.push(nested_loops);
            nested_loops = loop_node;
        }
        kernel_impl_body.push(nested_loops);
    }

    let kernel_impl = AstNode::_new(
        AstOp::Func {
            name: "kernel_impl".to_string(),
            args: kernel_impl_args,
        },
        kernel_impl_body,
        DType::Any,
    );

    // --- Main function wrapper (remains mostly the same) ---
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
    use crate::ast::Const;
    use crate::backend::Renderer;
    use crate::backend::c::CRenderer;
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

    #[test]
    fn test_lower_full() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let c = GraphNode::full(Const::F32(3.14f32.to_bits()), shape.clone());

        graph.outputs.push(c);
        graph.signature.outputs.push(TensorSignature {
            dtype: DType::F32,
            shape,
        });

        let ast = lower_graph(&graph);
        let mut renderer = CRenderer::new();
        let code = renderer.render(ast);

        let expected_code = r#"buf0[((idx0*20)+idx1)]=3.1400001;"#;
        assert!(
            code.replace([' ', '\t', '\n'], "")
                .contains(&expected_code.replace([' ', '\t', '\n'], "")),
            "Generated code:\n{}",
            code
        );
    }

    #[test]
    fn test_lower_full_with_op() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let a = graph.add_input(shape.clone(), &DType::F32);
        let b = GraphNode::full(Const::F32(1.0f32.to_bits()), shape.clone());
        let c = a + b;

        graph.outputs.push(c);
        graph.signature.outputs.push(TensorSignature {
            dtype: DType::F32,
            shape,
        });

        let ast = lower_graph(&graph);
        let mut renderer = CRenderer::new();
        let code = renderer.render(ast);

        let expected_code = r#"buf0[idx0] = (buf1[idx0] + 1.0000000);"#;
        assert!(
            code.replace([' ', '\t', '\n'], "")
                .contains(&expected_code.replace([' ', '\t', '\n'], "")),
            "Generated code:\n{}",
            code
        );
    }

    #[test]
    fn test_lower_permute() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let a = graph.add_input(shape, &DType::F32);
        let b = a.permute(vec![1, 0]);

        graph.outputs.push(b);
        graph.signature.outputs.push(TensorSignature {
            dtype: DType::F32,
            shape: vec![ShapeExpr::from(20), ShapeExpr::from(10)],
        });

        let ast = lower_graph(&graph);
        let mut renderer = CRenderer::new();
        let code = renderer.render(ast);

        let expected_store_and_load = r#"buf0[((idx0*10)+idx1)]=buf1[((idx1*20)+idx0)];"#;

        assert!(
            code.replace([' ', '\t', '\n'], "")
                .contains(&expected_store_and_load.replace([' ', '\t', '\n'], "")),
            "Generated code:\n{}",
            code
        );
    }
}
