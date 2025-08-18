use crate::ast::{AstNode, AstOp, DType};
use crate::graph::shape::view::View;
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::HashMap;

pub struct Lowerer {
    memo: HashMap<GraphNode, AstNode>,
    acc_counter: usize,
    ridx_counter: usize,
}

impl Lowerer {
    fn new() -> Self {
        Lowerer {
            memo: HashMap::new(),
            acc_counter: 0,
            ridx_counter: 0,
        }
    }

    fn lower_expr(&mut self, node: &GraphNode, indices: &[AstNode]) -> AstNode {
        if let Some(cached) = self.memo.get(node) {
            // Input nodes are memoized with their buffer name.
            // We need to calculate the physical index and load from it.
            if let GraphOp::Input { .. } = &node.op {
                let physical_index = node.view.to_physical_index_ast(indices);
                return AstNode::load(AstNode::index(cached.clone(), physical_index));
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
            GraphOp::Permute(_) => {
                let permuted_indices = node.permute_logical_indices(indices);
                self.lower_expr(&node.src[0], &permuted_indices)
            }
            GraphOp::Reduce(_, _) => {
                panic!("Reduce op cannot be an intermediate operation in this lowering model");
            }
            GraphOp::Cumulative(_, _) => {
                panic!("Cumulative op cannot be an intermediate operation in this lowering model");
            }
            GraphOp::Contiguous => {
                // In the new model, Contiguous might mean we need to compute and store
                // the result in a temporary buffer. For now, we pass through.
                self.lower_expr(&node.src[0], indices)
            }
            GraphOp::Capture(_) => panic!("Capture node should not be lowered"),
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
        .map(|i| format!("buf{i}"))
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

        let body = if let GraphOp::Reduce(op, axis) = &output_node.op {
            let src_node = &output_node.src[0];
            let output_shape = output_node.shape();
            let reduction_axis_size = src_node.shape()[*axis].clone();

            // --- Outer loops for the output tensor ---
            let mut outer_loops = vec![];
            let mut outer_indices = vec![];
            for (dim, size) in output_shape.iter().enumerate() {
                let loop_var_name = format!("idx{}", dim);
                let loop_var = AstNode::var(&loop_var_name, DType::Usize);
                outer_indices.push(loop_var);
                outer_loops.push(AstNode::_new(
                    AstOp::Range {
                        counter: loop_var_name,
                        step: 1,
                    },
                    vec![size.to_ast()],
                    DType::Void,
                ));
            }

            // --- Accumulator initialization ---
            let acc_name = format!("acc{}", lowerer.acc_counter);
            lowerer.acc_counter += 1;
            let acc_var = AstNode::var(&acc_name, output_node.dtype.clone());
            let identity = match op {
                AstOp::Add => output_node.dtype.zero(),
                AstOp::Mul => output_node.dtype.one(),
                _ => panic!("Unsupported reduction op: {:?}", op),
            };
            let acc_init = AstNode::declare(&acc_name, output_node.dtype.clone(), Some(identity));

            // --- Inner reduction loop ---
            let reduction_loop_var_name = format!("ridx{}", lowerer.ridx_counter);
            lowerer.ridx_counter += 1;
            let reduction_loop_var = AstNode::var(&reduction_loop_var_name, DType::Usize);
            let mut reduction_loop = AstNode::_new(
                AstOp::Range {
                    counter: reduction_loop_var_name,
                    step: 1,
                },
                vec![reduction_axis_size.to_ast()],
                DType::Void,
            );

            // --- Accumulation logic ---
            let mut src_indices = outer_indices.clone();
            src_indices.insert(*axis, reduction_loop_var);
            let src_expr = lowerer.lower_expr(src_node, &src_indices);
            let acc_update = AstNode::assign(
                acc_var.clone(),
                AstNode::_new(
                    op.clone(),
                    vec![acc_var.clone(), src_expr],
                    output_node.dtype.clone(),
                ),
            );
            reduction_loop.src.push(acc_update);

            // --- Store result ---
            let physical_output_index =
                View::new_contiguous(output_shape.to_vec()).to_physical_index_ast(&outer_indices);
            let store_op = AstNode::store(
                AstNode::index(output_ptr.clone(), physical_output_index),
                acc_var,
            );

            // --- Nesting ---
            let main_block = AstNode::_new(
                AstOp::Block,
                vec![acc_init, reduction_loop, store_op],
                DType::Void,
            );

            let mut nested_loops = main_block;
            for mut loop_node in outer_loops.into_iter().rev() {
                if !matches!(nested_loops.op, AstOp::Block) {
                    nested_loops = AstNode::_new(AstOp::Block, vec![nested_loops], DType::Void);
                }
                loop_node.src.push(nested_loops);
                nested_loops = loop_node;
            }
            nested_loops
        } else if let GraphOp::Cumulative(op, axis) = &output_node.op {
            let src_node = &output_node.src[0];
            let shape = src_node.shape();

            // --- Create all loops ---
            let mut loops = vec![];
            let mut indices = vec![];
            for (dim, size) in shape.iter().enumerate() {
                let loop_var_name = format!("idx{}", dim);
                let loop_var = AstNode::var(&loop_var_name, DType::Usize);
                indices.push(loop_var);
                loops.push(AstNode::_new(
                    AstOp::Range {
                        counter: loop_var_name,
                        step: 1,
                    },
                    vec![size.to_ast()],
                    DType::Void,
                ));
            }

            // --- Create the innermost body ---
            let acc_name = format!("acc{}", lowerer.acc_counter);
            let acc_var = AstNode::var(&acc_name, output_node.dtype.clone());

            let src_expr = lowerer.lower_expr(src_node, &indices);
            let acc_update = AstNode::assign(
                acc_var.clone(),
                AstNode::_new(
                    op.clone(),
                    vec![acc_var.clone(), src_expr],
                    output_node.dtype.clone(),
                ),
            );

            let physical_index =
                View::new_contiguous(output_node.shape().to_vec()).to_physical_index_ast(&indices);
            let store_op = AstNode::store(
                AstNode::index(output_ptr.clone(), physical_index),
                acc_var.clone(),
            );

            let innermost_body =
                AstNode::_new(AstOp::Block, vec![acc_update, store_op], DType::Void);

            // --- Nest loops and place accumulator ---
            let mut nested_loops = innermost_body;
            for (dim, mut loop_node) in loops.into_iter().enumerate().rev() {
                if !matches!(nested_loops.op, AstOp::Block) {
                    nested_loops = AstNode::_new(AstOp::Block, vec![nested_loops], DType::Void);
                }
                loop_node.src.push(nested_loops);
                nested_loops = loop_node;

                if dim == *axis {
                    let identity = match op {
                        AstOp::Add => output_node.dtype.zero(),
                        AstOp::Mul => output_node.dtype.one(),
                        _ => panic!("Unsupported cumulative op: {:?}", op),
                    };
                    let acc_init =
                        AstNode::declare(&acc_name, output_node.dtype.clone(), Some(identity));

                    nested_loops =
                        AstNode::_new(AstOp::Block, vec![acc_init, nested_loops], DType::Void);
                }
            }
            lowerer.acc_counter += 1;
            nested_loops
        } else {
            // Original logic for other ops
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
                    DType::Void,
                ));
            }

            let final_expr = lowerer.lower_expr(output_node, &indices);

            let physical_index =
                View::new_contiguous(output_node.shape().to_vec()).to_physical_index_ast(&indices);
            let store_op = AstNode::store(AstNode::index(output_ptr, physical_index), final_expr);

            // Nest loops
            let mut nested_loops = store_op;
            for mut loop_node in loops.into_iter().rev() {
                if !matches!(nested_loops.op, AstOp::Block) {
                    nested_loops = AstNode::_new(AstOp::Block, vec![nested_loops], DType::Void);
                }
                loop_node.src.push(nested_loops);
                nested_loops = loop_node;
            }
            nested_loops
        };
        kernel_impl_body.push(body);
    }

    let kernel_impl = AstNode::_new(
        AstOp::Func {
            name: "kernel_impl".to_string(),
            args: kernel_impl_args,
        },
        kernel_impl_body,
        DType::Void,
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
        for (i, arg) in args.iter().enumerate() {
            let ptr_to_buf_i = AstNode::index(
                AstNode::var("bufs", bufs_arg.1.clone()),
                AstNode::from(i as isize),
            );
            let cast_buf_i = AstNode::_new(
                AstOp::Cast(arg.1.clone()),
                vec![ptr_to_buf_i],
                arg.1.clone(),
            );
            call_args.push(cast_buf_i);
        }
    }

    main_body.push(AstNode::_new(
        AstOp::Call("kernel_impl".to_string()),
        call_args,
        DType::Void,
    ));

    let kernel_main = AstNode::_new(
        AstOp::Func {
            name: "kernel_main".to_string(),
            args: vec![bufs_arg, shape_vars_arg],
        },
        main_body,
        DType::Void,
    );

    AstNode::_new(AstOp::Program, vec![kernel_impl, kernel_main], DType::Void)
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

        let expected_code = r###"buf0[((idx0*20)+idx1)]=3.1400001;"###;
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

        let expected_code = r###"buf0[idx0]=(buf1[idx0]+1.0000000);"###;
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

        let expected_store_and_load = r###"buf0[((idx0*10)+idx1)]=buf1[((idx1*20)+idx0)];"###;

        assert!(
            code.replace([' ', '\t', '\n'], "")
                .contains(&expected_store_and_load.replace([' ', '\t', '\n'], "")),
            "Generated code:\n{}",
            code
        );
    }

    #[test]
    fn test_lower_cumulative_sum() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.cumulative(AstOp::Add, 0);

        graph.outputs.push(b);
        graph.signature.outputs.push(TensorSignature {
            dtype: dtype.clone(),
            shape,
        });

        let ast = lower_graph(&graph);
        let mut renderer = CRenderer::new();
        let code = renderer.render(ast);

        // Check for accumulator initialization
        assert!(code.contains("float acc0 = 0.0000000;"));
        // Check for the loop and accumulation
        let expected_loop_body = r###"acc0=(acc0+buf1[idx0]);buf0[idx0]=acc0;"###;
        assert!(
            code.replace([' ', '\t', '\n'], "")
                .contains(&expected_loop_body.replace([' ', '\t', '\n'], "")),
            "Generated code:\n{}",
            code
        );
    }

    #[test]
    fn test_lower_cumulative_sum_2d_axis1() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.cumulative(AstOp::Add, 1);

        graph.outputs.push(b);
        graph.signature.outputs.push(TensorSignature {
            dtype: dtype.clone(),
            shape,
        });

        let ast = lower_graph(&graph);
        let mut renderer = CRenderer::new();
        let code = renderer.render(ast);

        // Check that the accumulator is initialized inside the outer loop
        let expected_init = "for(size_t idx0=0; idx0<10; idx0++){\nfloatacc0=0.0000000;";
        assert!(
            code.replace([' ', '\t', '\n'], "")
                .contains(&expected_init.replace([' ', '\t', '\n'], "")),
            "Generated code:\n{}",
            code
        );

        // Check the inner loop body
        let expected_body = "acc0=(acc0+buf1[((idx0*20)+idx1)]);buf0[((idx0*20)+idx1)]=acc0;";
        assert!(
            code.replace([' ', '\t', '\n'], "")
                .contains(&expected_body.replace([' ', '\t', '\n'], "")),
            "Generated code:\n{}",
            code
        );
    }

    #[test]
    fn test_lower_reduce_sum() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let dtype = DType::F32;

        let a = graph.add_input(shape, &dtype);
        let b = a.reduce(AstOp::Add, 1); // Reduce along axis 1

        graph.outputs.push(b);
        graph.signature.outputs.push(TensorSignature {
            dtype: dtype.clone(),
            shape: vec![ShapeExpr::from(10)],
        });

        let ast = lower_graph(&graph);
        let mut renderer = CRenderer::new();
        let code = renderer.render(ast);

        // Check for accumulator initialization
        assert!(code.contains("float acc0 = 0.0000000;"));
        // Check for the reduction loop
        assert!(code.contains("for (size_t ridx0 = 0; ridx0 < 20; ridx0++)"));
        // Check for the accumulation
        assert!(code.contains("acc0 = (acc0 + buf1[((idx0 * 20) + ridx0)]);"));
        // Check for the final store
        assert!(code.contains("buf0[idx0] = acc0;"));
    }
}
