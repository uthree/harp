use super::utils::LowererUtils;
use crate::ast::helper::{block, store};
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::{ops::ReduceOp, shape::view::View, GraphNode};

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
        LowererUtils::declare_result_variable(&result_var, &node.view, &node.dtype, declarations);

        // view情報を取得
        let input_view = &input.view;
        let result_view = &node.view;

        // 縮約操作の初期値を定義
        let initial_value = LowererUtils::get_reduce_initial_value(op);

        // 多重ループでreduce操作を実行
        Some(Self::create_reduce_loops(
            input_view,
            result_view,
            axis,
            &input_var,
            &result_var,
            op,
            initial_value,
            &node.dtype, // 型情報を渡す
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_reduce_loops(
        input_view: &View,
        result_view: &View,
        reduce_axis: usize,
        input_var: &str,
        result_var: &str,
        reduce_op: &ReduceOp,
        initial_value: AstNode,
        result_dtype: &DType, // 型情報を追加
        dim: usize,
    ) -> AstNode {
        let View::Linear {
            shape: input_shape,
            strides: input_strides,
            offset: input_offset,
        } = input_view;
        let View::Linear {
            strides: result_strides,
            offset: result_offset,
            ..
        } = result_view;

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
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(result_var.to_string())),
                        index: Box::new(result_index.clone()),
                        vector_width: 1,
                    }),
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                ),
                ReduceOp::Mul => AstNode::Mul(
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(result_var.to_string())),
                        index: Box::new(result_index.clone()),
                        vector_width: 1,
                    }),
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                ),
                ReduceOp::Max => AstNode::Max(
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(result_var.to_string())),
                        index: Box::new(result_index.clone()),
                        vector_width: 1,
                    }),
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                ),
            };

            return store(
                AstNode::Var(result_var.to_string()),
                result_index,
                operation_result,
            );
        }

        if dim == reduce_axis {
            // 縮約する次元: アキュムレータ変数を使った縮約
            // アキュムレータ変数名を生成
            let acc_var = format!("acc{}", dim);

            // 最内ループ部分を生成（アキュムレータに累積）
            let inner_body = Self::create_reduce_loops_with_accumulator(
                input_view,
                reduce_axis,
                input_var,
                &acc_var, // アキュムレータ変数を渡す
                reduce_op,
                dim + 1,
            );

            let loop_var = format!("ridx{}", dim);

            // アキュムレータの初期化
            let init_stmt = AstNode::Assign(acc_var.clone(), Box::new(initial_value.clone()));

            // 縮約ループ: for (i_reduce) { inner_body }
            let reduce_loop =
                LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body);

            // アキュムレータから結果配列への書き込み
            let result_index = LowererUtils::compute_reduce_result_index(
                result_strides,
                result_offset,
                dim,
                reduce_axis,
            );
            let write_back_stmt = store(
                AstNode::Var(result_var.to_string()),
                result_index,
                AstNode::Var(acc_var.clone()),
            );

            // アキュムレータ変数の宣言 + 初期化 + 縮約ループ + 書き戻しをブロックにまとめる
            block(
                crate::ast::Scope {
                    declarations: vec![VariableDecl {
                        name: acc_var,
                        dtype: result_dtype.clone(),
                        constant: false,
                        size_expr: None,
                    }],
                },
                vec![init_stmt, reduce_loop, write_back_stmt],
            )
        } else {
            // 縮約しない次元: 通常のループ
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_reduce_loops(
                input_view,
                result_view,
                reduce_axis,
                input_var,
                result_var,
                reduce_op,
                initial_value,
                result_dtype,
                dim + 1,
            );

            LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body)
        }
    }

    /// アキュムレータ変数を使った縮約ループの本体を生成
    fn create_reduce_loops_with_accumulator(
        input_view: &View,
        reduce_axis: usize,
        input_var: &str,
        acc_var: &str, // アキュムレータ変数名
        reduce_op: &ReduceOp,
        dim: usize,
    ) -> AstNode {
        let View::Linear {
            shape: input_shape,
            strides: input_strides,
            offset: input_offset,
        } = input_view;

        if dim >= input_shape.len() {
            // 全ての次元を処理した：アキュムレータに累積
            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());

            // 縮約操作: acc = acc op input[...]
            let operation_result = match reduce_op {
                ReduceOp::Add => AstNode::Add(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                ),
                ReduceOp::Mul => AstNode::Mul(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                ),
                ReduceOp::Max => AstNode::Max(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                ),
            };

            return AstNode::Assign(acc_var.to_string(), Box::new(operation_result));
        }

        if dim == reduce_axis {
            // 縮約軸は既に外側で処理されているので、ここでは単に次の次元へ
            return Self::create_reduce_loops_with_accumulator(
                input_view,
                reduce_axis,
                input_var,
                acc_var,
                reduce_op,
                dim + 1,
            );
        }

        // 通常の次元: ループを生成
        let loop_var = format!("ridx{}", dim);
        let inner_body = Self::create_reduce_loops_with_accumulator(
            input_view,
            reduce_axis,
            input_var,
            acc_var,
            reduce_op,
            dim + 1,
        );

        LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body)
    }

    /// コピーループを作成（View操作用）
    pub fn create_copy_loop(view: &View, source_var: &str, dest_var: &str, dim: usize) -> AstNode {
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;

        if dim >= shape.len() {
            // 最内レベル: コピーを実行
            let source_index = LowererUtils::compute_memory_index(strides, offset, dim);
            let dest_index = LowererUtils::compute_memory_index(strides, offset, dim);

            store(
                AstNode::Var(dest_var.to_string()),
                dest_index,
                AstNode::Load {
                    target: Box::new(AstNode::Var(source_var.to_string())),
                    index: Box::new(source_index),
                    vector_width: 1,
                },
            )
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_copy_loop(view, source_var, dest_var, dim + 1);

            LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body)
        }
    }
}
