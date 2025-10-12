use super::utils::LowererUtils;
use crate::ast::{AstNode, ConstLiteral, DType, VariableDecl};
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
            input_view,
            result_view,
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
        input_view: &View,
        result_view: &View,
        cumulative_axis: usize,
        input_var: &str,
        result_var: &str,
        cumulative_op: &CumulativeOp,
        _initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        let (
            View::Linear {
                shape: input_shape,
                strides: input_strides,
                offset: input_offset,
            },
            View::Linear {
                shape: result_shape,
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);

        if dim >= input_shape.len() {
            // 全ての次元を処理した：ここには到達しない（累積軸のループ内で処理される）
            unreachable!()
        } else if dim == cumulative_axis {
            // 累積軸: アキュムレータ変数を使用
            let acc_var = format!("acc{}", dim);
            let loop_var = format!("ridx{}", dim);
            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

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

            let input_value =
                AstNode::Deref(Box::new(AstNode::Var(input_var.to_string()) + input_index));

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
            let write_stmt = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Var(acc_var.clone())),
            };

            // ループ本体: accumulate + write
            let loop_body = AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![],
                },
                statements: vec![accumulate_stmt, write_stmt],
            };

            // 累積ループ
            let cumulative_loop = AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                body: Box::new(loop_body),
                unroll: None,
            };

            // アキュムレータ変数の宣言 + 初期化 + 累積ループをブロックにまとめる
            AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![VariableDecl {
                        name: acc_var,
                        dtype: DType::F32, // 型を仮定
                        constant: false,
                        size_expr: None,
                    }],
                },
                statements: vec![init_stmt, cumulative_loop],
            }
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
                dim + 1,
            );

            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
            }
        }
    }
}
