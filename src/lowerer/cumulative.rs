use super::utils::LowererUtils;
use crate::ast::helper::{block, block_with_statements, store};
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::{ops::CumulativeOp, shape::view::View, GraphNode};

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
        LowererUtils::declare_result_variable(&result_var, &node.view, &node.dtype, declarations);

        // view情報を取得
        let input_view = &input.view;
        let result_view = &node.view;

        // 累積演算のための初期値を定義
        let initial_value = LowererUtils::get_cumulative_initial_value(op);

        // 多重ループでcumulative操作を実行
        Some(Self::create_cumulative_loops(
            input_view,
            result_view,
            axis,
            &input_var,
            &result_var,
            op,
            initial_value,
            &node.dtype,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_cumulative_loops(
        input_view: &View,
        result_view: &View,
        cumulative_axis: usize,
        input_var: &str,
        result_var: &str,
        cumulative_op: &CumulativeOp,
        _initial_value: AstNode,
        result_dtype: &DType,
        dim: usize,
    ) -> AstNode {
        let View::Linear {
            shape: input_shape,
            strides: input_strides,
            offset: input_offset,
            ..
        } = input_view;
        let View::Linear {
            shape: result_shape,
            strides: result_strides,
            offset: result_offset,
            ..
        } = result_view;

        if dim >= input_shape.len() {
            // 全ての次元を処理した：ここには到達しない（累積軸のループ内で処理される）
            unreachable!()
        } else if dim == cumulative_axis {
            // 累積軸: アキュムレータ変数を使用
            let acc_var = format!("acc{}", dim);
            let loop_var = format!("ridx{}", dim);

            // アキュムレータの初期化
            let init_stmt = AstNode::Assign(acc_var.clone(), Box::new(_initial_value.clone()));

            // 累積ループの本体
            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());
            let result_index = LowererUtils::compute_memory_index(
                result_strides,
                result_offset,
                result_shape.len(),
            );

            let input_value = AstNode::Load {
                target: Box::new(AstNode::Var(input_var.to_string())),
                index: Box::new(input_index),
                vector_width: 1,
            };

            // acc = acc op input[...]
            let accumulate_stmt = {
                let cumulative_value = match cumulative_op {
                    CumulativeOp::Add => AstNode::Add(
                        Box::new(AstNode::Var(acc_var.clone())),
                        Box::new(input_value),
                    ),
                    CumulativeOp::Mul => AstNode::Mul(
                        Box::new(AstNode::Var(acc_var.clone())),
                        Box::new(input_value),
                    ),
                    CumulativeOp::Max => AstNode::Max(
                        Box::new(AstNode::Var(acc_var.clone())),
                        Box::new(input_value),
                    ),
                };
                AstNode::Assign(acc_var.clone(), Box::new(cumulative_value))
            };

            // result[...] = acc
            let write_stmt = store(
                AstNode::Var(result_var.to_string()),
                result_index,
                AstNode::Var(acc_var.clone()),
            );

            // ループ本体: accumulate + write
            let loop_body = block_with_statements(vec![accumulate_stmt, write_stmt]);

            // 累積ループ
            let cumulative_loop =
                LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], loop_body);

            // アキュムレータ変数の宣言 + 初期化 + 累積ループをブロックにまとめる
            block(
                crate::ast::Scope {
                    declarations: vec![VariableDecl {
                        name: acc_var,
                        dtype: result_dtype.clone(),
                        constant: false,
                        size_expr: None,
                    }],
                },
                vec![init_stmt, cumulative_loop],
            )
        } else {
            // 累積軸以外の次元: 通常のループ
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_cumulative_loops(
                input_view,
                result_view,
                cumulative_axis,
                input_var,
                result_var,
                cumulative_op,
                _initial_value,
                result_dtype,
                dim + 1,
            );

            LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body)
        }
    }
}
