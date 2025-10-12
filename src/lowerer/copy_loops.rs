use super::Lowerer;
use crate::ast::{AstNode, ConstLiteral, DType};
use crate::graph::shape::view::View;
use crate::lowerer::utils::LowererUtils;

impl Lowerer {
    /// Contiguous変換のためのコピーループを作成
    pub(super) fn create_contiguous_copy_loop(
        input_view: &View,
        result_view: &View,
        input_var: &str,
        result_var: &str,
        dim: usize,
    ) -> AstNode {
        let (
            View::Linear {
                shape,
                strides: input_strides,
                offset: input_offset,
            },
            View::Linear {
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);

        if dim >= shape.len() {
            // 最内レベル: コピーを実行
            let input_index = LowererUtils::compute_memory_index(input_strides, input_offset, dim);
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Deref(Box::new(
                    AstNode::Var(input_var.to_string()) + input_index,
                ))),
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_contiguous_copy_loop(
                input_view,
                result_view,
                input_var,
                result_var,
                dim + 1,
            );

            let shape_size = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
            }
        }
    }

    /// Castのためのループを作成
    pub(super) fn create_cast_loop(
        input_view: &View,
        result_view: &View,
        input_var: &str,
        result_var: &str,
        target_dtype: &DType,
        dim: usize,
    ) -> AstNode {
        let (
            View::Linear {
                shape,
                strides: input_strides,
                offset: input_offset,
            },
            View::Linear {
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);

        if dim >= shape.len() {
            // 最内レベル: キャストを実行
            let input_index = LowererUtils::compute_memory_index(input_strides, input_offset, dim);
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);

            // Cast AstNodeを使用して型変換
            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Cast {
                    dtype: target_dtype.clone(),
                    expr: Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                }),
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_cast_loop(
                input_view,
                result_view,
                input_var,
                result_var,
                target_dtype,
                dim + 1,
            );

            let shape_size = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
            }
        }
    }
}
