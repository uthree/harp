use super::Lowerer;
use crate::ast::{AstNode, ConstLiteral, DType};
use crate::lowerer::utils::LowererUtils;

impl Lowerer {
    /// Contiguous変換のためのコピーループを作成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_contiguous_copy_loop(
        shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        input_var: &str,
        result_var: &str,
        dim: usize,
    ) -> AstNode {
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
                shape,
                input_strides,
                input_offset,
                result_strides,
                result_offset,
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_cast_loop(
        shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        input_var: &str,
        result_var: &str,
        _target_dtype: &DType,
        dim: usize,
    ) -> AstNode {
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
                    dtype: _target_dtype.clone(),
                    expr: Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                }),
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_cast_loop(
                shape,
                input_strides,
                input_offset,
                result_strides,
                result_offset,
                input_var,
                result_var,
                _target_dtype,
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
