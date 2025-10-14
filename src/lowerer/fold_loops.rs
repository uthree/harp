use super::Lowerer;
use crate::ast::helper::{block_with_statements, store};
use crate::ast::{AstNode, ConstLiteral};
use crate::graph::shape::view::View;
use crate::lowerer::utils::LowererUtils;

impl Lowerer {
    /// Foldのためのループを作成 (col2im operation)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_fold_loops(
        input_view: &View,
        result_view: &View,
        dim: usize,
        stride: usize,
        dilation: usize,
        input_var: &str,
        result_var: &str,
    ) -> AstNode {
        // Phase 1: Initialize output to zero
        let init_loop = Self::create_fold_init_loop(result_view, result_var, 0);

        // Phase 2: Accumulate values from input windows
        let accum_loop = Self::create_fold_accumulate_loop(
            input_view,
            result_view,
            dim,
            stride,
            dilation,
            input_var,
            result_var,
            0,
        );

        // Combine init and accumulate in a block
        block_with_statements(vec![init_loop, accum_loop])
    }

    /// Initialize output buffer to zero for fold operation
    fn create_fold_init_loop(result_view: &View, result_var: &str, dim: usize) -> AstNode {
        let View::Linear {
            shape: result_shape,
            strides: result_strides,
            offset: result_offset,
        } = result_view;

        if dim >= result_shape.len() {
            // Initialize to zero
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);
            store(
                AstNode::Var(result_var.to_string()),
                result_index,
                AstNode::Const(ConstLiteral::F32(0.0)),
            )
        } else {
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_fold_init_loop(result_view, result_var, dim + 1);

            LowererUtils::create_dimension_loop(loop_var, &result_shape[dim], inner_body)
        }
    }

    /// Accumulate values from input windows into output for fold operation
    #[allow(clippy::too_many_arguments)]
    fn create_fold_accumulate_loop(
        input_view: &View,
        result_view: &View,
        fold_dim: usize,
        stride: usize,
        dilation: usize,
        input_var: &str,
        result_var: &str,
        current_dim: usize,
    ) -> AstNode {
        let View::Linear {
            shape: input_shape,
            strides: input_strides,
            offset: input_offset,
        } = input_view;
        let View::Linear {
            strides: _result_strides,
            offset: _result_offset,
            ..
        } = result_view;

        let window_dim = input_shape.len() - 1; // Last dimension is window dimension

        if current_dim >= input_shape.len() {
            // All dimensions processed: perform accumulation
            // input[..., i_fold_dim, i_window_dim] accumulates to
            // output[..., i_fold_dim * stride + i_window_dim * dilation]

            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());

            // Compute result index with stride and dilation adjustment
            // For fold_dim: use i_fold_dim * stride + i_window_dim * dilation instead of i_fold_dim
            let result_index = Self::compute_fold_result_index(
                result_view,
                fold_dim,
                window_dim,
                stride,
                dilation,
                input_shape.len(),
            );

            // result[idx] += input[idx]
            store(
                AstNode::Var(result_var.to_string()),
                result_index.clone(),
                AstNode::Add(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index,
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
            )
        } else {
            // Generate loop for current dimension
            let loop_var = format!("ridx{}", current_dim);
            let inner_body = Self::create_fold_accumulate_loop(
                input_view,
                result_view,
                fold_dim,
                stride,
                dilation,
                input_var,
                result_var,
                current_dim + 1,
            );

            LowererUtils::create_dimension_loop(loop_var, &input_shape[current_dim], inner_body)
        }
    }

    /// Compute result index for fold operation
    /// Maps input[..., i_fold_dim, ..., i_window_dim] to output[..., i_fold_dim * stride + i_window_dim * dilation, ...]
    fn compute_fold_result_index(
        result_view: &View,
        fold_dim: usize,
        window_dim: usize,
        stride: usize,
        dilation: usize,
        num_input_dims: usize,
    ) -> AstNode {
        let View::Linear {
            strides: result_strides,
            offset: result_offset,
            ..
        } = result_view;
        let mut index = LowererUtils::shape_expr_to_ast_node(result_offset);

        for dim in 0..num_input_dims {
            if dim == window_dim {
                // Skip window dimension (it's been folded into fold_dim)
                continue;
            }

            let loop_var = format!("ridx{}", dim);
            let result_dim = if dim > fold_dim { dim - 1 } else { dim };

            if dim == fold_dim {
                // For fold_dim: result_index = i_fold_dim * stride + i_window_dim * dilation
                let fold_index = AstNode::Add(
                    Box::new(AstNode::Mul(
                        Box::new(AstNode::Var(loop_var)),
                        Box::new(AstNode::Const(ConstLiteral::Isize(stride as isize))),
                    )),
                    Box::new(AstNode::Mul(
                        Box::new(AstNode::Var(format!("ridx{}", window_dim))),
                        Box::new(AstNode::Const(ConstLiteral::Isize(dilation as isize))),
                    )),
                );
                index += LowererUtils::shape_expr_to_ast_node(&result_strides[result_dim].clone())
                    * fold_index;
            } else {
                index += LowererUtils::shape_expr_to_ast_node(&result_strides[result_dim].clone())
                    * AstNode::Var(loop_var);
            }
        }

        index
    }
}
