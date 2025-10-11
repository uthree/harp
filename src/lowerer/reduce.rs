use super::utils::LowererUtils;
use crate::ast::{AstNode, ConstLiteral, DType, VariableDecl};
use crate::graph::{ops::ReduceOp, GraphNode};

/// Reduce演算のloweringを担当
pub(super) struct ReduceLowerer;

impl ReduceLowerer {
    /// Reduce演算をlowerする
    pub fn lower(
        node: &GraphNode,
        op: &ReduceOp,
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
                shape: _result_shape,
                strides: result_strides,
                offset: result_offset,
            },
        ) = (input_view, result_view);

        // 縮約操作の初期値を定義
        let initial_value = match op {
            ReduceOp::Add => AstNode::Const(ConstLiteral::F32(0.0)),
            ReduceOp::Mul => AstNode::Const(ConstLiteral::F32(1.0)),
            ReduceOp::Max => AstNode::Const(ConstLiteral::F32(f32::NEG_INFINITY)),
        };

        // 多重ループでreduce操作を実行
        Some(Self::create_reduce_loops(
            input_shape,
            input_strides,
            input_offset,
            _result_shape,
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
    fn create_reduce_loops(
        input_shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        _result_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        reduce_axis: usize,
        input_var: &str,
        result_var: &str,
        reduce_op: &ReduceOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        if dim >= input_shape.len() {
            // 全ての次元を処理した：縮約軸のループ本体を生成
            // この時点でdim == input_shape.len()なので、全てのループ変数が定義されている
            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());
            let result_index = LowererUtils::compute_reduce_result_index(
                result_strides,
                result_offset,
                input_shape.len(),
                reduce_axis,
            );

            // 縮約操作: result[...] = result[...] op input[...]
            let operation_result = match reduce_op {
                ReduceOp::Add => AstNode::Add(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
                ReduceOp::Mul => AstNode::Mul(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
                ReduceOp::Max => AstNode::Max(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
            };

            return AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(operation_result),
            };
        }

        if dim == reduce_axis {
            // 縮約する次元: 初期化 + ループで累積
            // 縮約軸以降の次元のループを再帰的に生成
            let inner_body = Self::create_reduce_loops(
                input_shape,
                input_strides,
                input_offset,
                _result_shape,
                result_strides,
                result_offset,
                reduce_axis,
                input_var,
                result_var,
                reduce_op,
                initial_value.clone(),
                dim + 1,
            );

            let loop_var = format!("ridx{}", dim);
            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

            // 結果の初期化（縮約軸をスキップしたインデックスで計算）
            let result_index = LowererUtils::compute_reduce_result_index(
                result_strides,
                result_offset,
                dim,
                reduce_axis,
            );
            let init_stmt = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(initial_value),
            };

            // 縮約ループ: for (i_reduce) { inner_body }
            let reduce_loop = AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
            };

            // 初期化 + 縮約ループをブロックにまとめる
            AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![],
                },
                statements: vec![init_stmt, reduce_loop],
            }
        } else {
            // 縮約しない次元: 通常のループ
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_reduce_loops(
                input_shape,
                input_strides,
                input_offset,
                _result_shape,
                result_strides,
                result_offset,
                reduce_axis,
                input_var,
                result_var,
                reduce_op,
                initial_value,
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

    /// コピーループを作成（View操作用）
    pub fn create_copy_loop(
        shape: &[crate::graph::shape::Expr],
        strides: &[crate::graph::shape::Expr],
        offset: &crate::graph::shape::Expr,
        source_var: &str,
        dest_var: &str,
        dim: usize,
    ) -> AstNode {
        if dim >= shape.len() {
            // 最内レベル: コピーを実行
            let source_index = LowererUtils::compute_memory_index(strides, offset, dim);
            let dest_index = LowererUtils::compute_memory_index(strides, offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(dest_var.to_string())),
                index: Box::new(dest_index),
                value: Box::new(AstNode::Deref(Box::new(
                    AstNode::Var(source_var.to_string()) + source_index,
                ))),
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body =
                Self::create_copy_loop(shape, strides, offset, source_var, dest_var, dim + 1);

            let shape_size = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

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
