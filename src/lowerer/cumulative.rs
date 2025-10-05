use super::utils::LowererUtils;
use crate::ast::{AstNode, ConstLiteral, DType, VariableDecl};
use crate::graph::{ops::CumulativeOp, GraphNode};

/// Cumulative演算のloweringを担当
pub(super) struct CumulativeLowerer;

impl CumulativeLowerer {
    /// Cumulative演算をlowerする
    pub fn lower(
        node: &GraphNode,
        op: &CumulativeOp,
        axis: usize,
        input: &GraphNode,
        mut get_var: impl FnMut(&GraphNode) -> String,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        let result_var = get_var(node);
        let input_var = get_var(input);

        // 出力ノードの場合は配列を宣言しない（引数として渡される）
        if !result_var.starts_with("output_") {
            // テンソルの場合は配列として宣言する必要がある
            let total_size = LowererUtils::compute_total_size(&node.view);
            let (result_dtype, size_expr) = if let Some(size) = total_size {
                // サイズが静的に決定できる場合は固定サイズ配列型
                (DType::Vec(Box::new(node.dtype.clone()), size), None)
            } else {
                // 動的サイズの場合はポインタ型（mallocで確保）
                let size_expr = LowererUtils::compute_total_size_expr(&node.view);
                (
                    DType::Ptr(Box::new(node.dtype.clone())),
                    Some(Box::new(size_expr)),
                )
            };

            declarations.push(VariableDecl {
                name: result_var.clone(),
                dtype: result_dtype,
                constant: false,
                size_expr,
            });
        }

        // view情報を取得
        let input_view = &input.view;
        let result_view = &node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: input_shape,
                strides: input_strides,
                offset: input_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: result_shape,
                strides: result_strides,
                offset: result_offset,
            },
        ) = (input_view, result_view);

        // 累積演算のための初期値を定義
        let initial_value = match op {
            CumulativeOp::Add => AstNode::Const(ConstLiteral::F32(0.0)),
            CumulativeOp::Mul => AstNode::Const(ConstLiteral::F32(1.0)),
            CumulativeOp::Max => AstNode::Const(ConstLiteral::F32(f32::NEG_INFINITY)),
        };

        // 多重ループでcumulative操作を実行
        Some(Self::create_cumulative_loops(
            input_shape,
            input_strides,
            input_offset,
            result_shape,
            result_strides,
            result_offset,
            axis,
            &input_var,
            &result_var,
            op,
            initial_value,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_cumulative_loops(
        input_shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        cumulative_axis: usize,
        input_var: &str,
        result_var: &str,
        cumulative_op: &CumulativeOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        if dim >= input_shape.len() {
            // 全ての次元を処理した：ここには到達しない（累積軸のループ内で処理される）
            unreachable!()
        } else if dim == cumulative_axis {
            // 累積軸: 特別な処理
            let loop_var = format!("i{}", dim);
            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

            // 累積軸の最初の要素を初期化
            // result[..., 0, ...] = input[..., 0, ...]
            let first_input_index = {
                // i{axis} = 0 の状態でインデックスを計算
                let zero_expr = crate::graph::shape::Expr::Const(0);

                LowererUtils::compute_memory_index_with_override(
                    input_strides,
                    input_offset,
                    input_shape.len(),
                    cumulative_axis,
                    &zero_expr,
                )
            };

            let first_result_index = {
                let zero_expr = crate::graph::shape::Expr::Const(0);
                LowererUtils::compute_memory_index_with_override(
                    result_strides,
                    result_offset,
                    result_shape.len(),
                    cumulative_axis,
                    &zero_expr,
                )
            };

            let first_init = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(first_result_index),
                value: Box::new(AstNode::Deref(Box::new(
                    AstNode::Var(input_var.to_string()) + first_input_index,
                ))),
            };

            // 累積ループ: i = 1 から開始
            // result[..., i, ...] = result[..., i-1, ...] op input[..., i, ...]
            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());
            let result_index = LowererUtils::compute_memory_index(
                result_strides,
                result_offset,
                result_shape.len(),
            );

            // 前の要素のインデックス（i-1）
            let prev_index = {
                let prev_offset = result_offset.clone()
                    + result_strides
                        .iter()
                        .enumerate()
                        .map(|(d, stride)| {
                            if d == cumulative_axis {
                                stride.clone() * crate::graph::shape::Expr::Var(format!("i{}", d))
                                    - stride.clone()
                            } else {
                                stride.clone() * crate::graph::shape::Expr::Var(format!("i{}", d))
                            }
                        })
                        .fold(crate::graph::shape::Expr::Const(0), |acc, x| acc + x);
                LowererUtils::shape_expr_to_ast_node(&prev_offset.simplify())
            };

            let prev_value =
                AstNode::Deref(Box::new(AstNode::Var(result_var.to_string()) + prev_index));
            let input_value =
                AstNode::Deref(Box::new(AstNode::Var(input_var.to_string()) + input_index));

            let cumulative_value = match cumulative_op {
                CumulativeOp::Add => prev_value + input_value,
                CumulativeOp::Mul => prev_value * input_value,
                CumulativeOp::Max => AstNode::Max(Box::new(prev_value), Box::new(input_value)),
            };

            let cumulative_stmt = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(cumulative_value),
            };

            // i=1から開始するループを作成
            // for (size_t i = 1; i < shape_size; i++)
            let cumulative_loop = AstNode::RangeFrom {
                counter_name: loop_var,
                start: Box::new(AstNode::from(1usize)),
                max: Box::new(shape_size),
                body: Box::new(cumulative_stmt),
            };

            // 初期化 + 累積ループをブロックにまとめる
            AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![],
                },
                statements: vec![first_init, cumulative_loop],
            }
        } else {
            // 累積軸以外の次元: 通常のループ
            let loop_var = format!("i{}", dim);
            let inner_body = Self::create_cumulative_loops(
                input_shape,
                input_strides,
                input_offset,
                result_shape,
                result_strides,
                result_offset,
                cumulative_axis,
                input_var,
                result_var,
                cumulative_op,
                initial_value,
                dim + 1,
            );

            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(shape_size),
                body: Box::new(inner_body),
            }
        }
    }
}
