use super::fused_elementwise::FusedElementwiseLowerer;
use super::utils::LowererUtils;
use crate::ast::helper::{block, block_with_statements, store};
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::{ops::CumulativeOp, GraphNode};

/// FusedElementwiseCumulative演算のコード生成を行う構造体
pub(super) struct FusedElementwiseCumulativeLowerer;

impl FusedElementwiseCumulativeLowerer {
    /// FusedElementwiseCumulative演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower(
        node: &GraphNode,
        ast: &AstNode,
        inputs: &[GraphNode],
        op: &CumulativeOp,
        axis: usize,
        declarations: &mut Vec<VariableDecl>,
        mut get_var: impl FnMut(&GraphNode) -> String,
    ) -> Option<AstNode> {
        let result_var = get_var(node);

        // 出力ノードの場合は配列を宣言しない
        LowererUtils::declare_result_variable(&result_var, &node.view, &node.dtype, declarations);

        // 入力の変数名を取得
        let input_vars: Vec<String> = inputs.iter().map(get_var).collect();

        // 初期値を定義
        let initial_value = LowererUtils::get_cumulative_initial_value(op);

        // 入力の最初のノードからshapeを取得（全て同じshapeのはず）
        let input_view = &inputs[0].view;
        let result_view = &node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: input_shape, ..
            },
            crate::graph::shape::view::View::Linear {
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);

        // ループを生成
        Some(Self::create_loops(
            input_shape,
            result_strides,
            result_offset,
            ast,
            &input_vars,
            inputs,
            axis,
            &result_var,
            op,
            initial_value,
            &node.dtype,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_loops(
        input_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        ast: &AstNode,
        input_vars: &[String],
        inputs: &[GraphNode],
        cumulative_axis: usize,
        result_var: &str,
        cumulative_op: &CumulativeOp,
        initial_value: AstNode,
        result_dtype: &DType,
        dim: usize,
    ) -> AstNode {
        if dim >= input_shape.len() {
            // ここには到達しない（累積軸のループ内で処理される）
            unreachable!()
        } else if dim == cumulative_axis {
            // 累積軸: アキュムレータ変数を使用
            let acc_var = format!("acc{}", dim);
            let loop_var = format!("ridx{}", dim);

            // アキュムレータの初期化
            let init_stmt = AstNode::Assign(acc_var.clone(), Box::new(initial_value.clone()));

            // 累積ループの本体
            // 融合されたElementwise式を評価
            let fused_value = FusedElementwiseLowerer::replace_captures_with_input_refs(
                ast,
                input_vars,
                inputs,
                &vec![crate::graph::shape::Expr::from(0); input_shape.len()],
                &crate::graph::shape::Expr::from(0),
                input_shape.len(),
            );

            // acc = acc op fused_value
            let accumulate_stmt = {
                let cumulative_value = match cumulative_op {
                    CumulativeOp::Add => AstNode::Add(
                        Box::new(AstNode::Var(acc_var.clone())),
                        Box::new(fused_value),
                    ),
                    CumulativeOp::Mul => AstNode::Mul(
                        Box::new(AstNode::Var(acc_var.clone())),
                        Box::new(fused_value),
                    ),
                    CumulativeOp::Max => AstNode::Max(
                        Box::new(AstNode::Var(acc_var.clone())),
                        Box::new(fused_value),
                    ),
                };
                AstNode::Assign(acc_var.clone(), Box::new(cumulative_value))
            };

            // 結果インデックスの計算
            let result_index = LowererUtils::compute_memory_index(
                result_strides,
                result_offset,
                input_shape.len(),
            );

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
                LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], loop_body, None);

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
            let inner_body = Self::create_loops(
                input_shape,
                result_strides,
                result_offset,
                ast,
                input_vars,
                inputs,
                cumulative_axis,
                result_var,
                cumulative_op,
                initial_value,
                result_dtype,
                dim + 1,
            );

            LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body, None)
        }
    }
}
