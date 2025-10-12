use super::fused_elementwise::FusedElementwiseLowerer;
use super::utils::LowererUtils;
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::{ops::ReduceOp, GraphNode};

/// FusedElementwiseReduce演算のコード生成を行う構造体
pub(super) struct FusedElementwiseReduceLowerer;

impl FusedElementwiseReduceLowerer {
    /// FusedElementwiseReduce演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower(
        node: &GraphNode,
        ast: &AstNode,
        inputs: &[GraphNode],
        op: &ReduceOp,
        axes: &[usize],
        declarations: &mut Vec<VariableDecl>,
        mut get_var: impl FnMut(&GraphNode) -> String,
    ) -> Option<AstNode> {
        assert_eq!(
            axes.len(),
            1,
            "FusedElementwiseReduce currently only supports single axis"
        );
        let axis = axes[0];

        let result_var = get_var(node);

        // 出力ノードの場合は配列を宣言しない
        if !result_var.starts_with("output_") {
            let total_size = LowererUtils::compute_total_size(&node.view);
            let (result_dtype, size_expr) = if let Some(size) = total_size {
                (DType::Vec(Box::new(node.dtype.clone()), size), None)
            } else {
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

        // 入力の変数名を取得
        let input_vars: Vec<String> = inputs.iter().map(get_var).collect();

        // 初期値を定義
        let initial_value = match op {
            ReduceOp::Add => AstNode::Const(crate::ast::ConstLiteral::F32(0.0)),
            ReduceOp::Mul => AstNode::Const(crate::ast::ConstLiteral::F32(1.0)),
            ReduceOp::Max => AstNode::Const(crate::ast::ConstLiteral::F32(f32::NEG_INFINITY)),
        };

        // 入力の最初のノードからshapeを取得（全て同じshapeのはず）
        let input_view = &inputs[0].view;
        let result_view = &node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: input_shape,
                strides: _input_strides,
                offset: _input_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _result_shape,
                strides: _result_strides,
                offset: _result_offset,
            },
        ) = (input_view, result_view);

        // ループを生成
        Some(Self::create_loops(
            input_shape,
            result_view,
            ast,
            &input_vars,
            inputs,
            axis,
            &result_var,
            op,
            initial_value,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_loops(
        input_shape: &[crate::graph::shape::Expr],
        result_view: &crate::graph::shape::view::View,
        ast: &AstNode,
        input_vars: &[String],
        inputs: &[GraphNode],
        reduce_axis: usize,
        result_var: &str,
        reduce_op: &ReduceOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        let crate::graph::shape::view::View::Linear {
            shape: _result_shape,
            strides: result_strides,
            offset: result_offset,
        } = result_view;

        if dim >= input_shape.len() {
            // 全ての次元を処理した：reduce操作を実行
            // 融合されたElementwise式を評価
            let fused_value = FusedElementwiseLowerer::replace_captures_with_input_refs(
                ast,
                input_vars,
                inputs,
                &vec![crate::graph::shape::Expr::from(0); input_shape.len()],
                &crate::graph::shape::Expr::from(0),
                input_shape.len(),
            );

            let result_index = LowererUtils::compute_reduce_result_index(
                result_strides,
                result_offset,
                input_shape.len(),
                reduce_axis,
            );

            // 縮約操作
            let operation_result = match reduce_op {
                ReduceOp::Add => AstNode::Add(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(fused_value),
                ),
                ReduceOp::Mul => AstNode::Mul(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(fused_value),
                ),
                ReduceOp::Max => AstNode::Max(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(fused_value),
                ),
            };

            return AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(operation_result),
            };
        }

        if dim == reduce_axis {
            // 縮約する次元の処理
            // 後続の次元がある場合は、それらのループの内側でアキュムレータを使用
            if reduce_axis + 1 < input_shape.len() {
                // 後続の非縮約次元のループを生成
                return Self::create_post_reduce_loops_with_accumulator(
                    input_shape,
                    result_strides,
                    result_offset,
                    result_view,
                    ast,
                    input_vars,
                    inputs,
                    reduce_axis,
                    result_var,
                    reduce_op,
                    initial_value,
                    reduce_axis + 1, // 縮約軸の次の次元から開始
                );
            } else {
                // 最後の軸を縮約する場合：アキュムレータ変数を使った縮約
                let acc_var = format!("acc{}", dim);

                // アキュムレータへの累積を行う内部ループ
                let inner_body = Self::create_with_accumulator(
                    input_shape,
                    ast,
                    input_vars,
                    inputs,
                    reduce_axis,
                    &acc_var,
                    reduce_op,
                    dim + 1,
                );

                let loop_var = format!("ridx{}", dim);

                // アキュムレータの初期化
                let init_stmt = AstNode::Assign(acc_var.clone(), Box::new(initial_value.clone()));

                // 縮約ループ
                let reduce_loop =
                    LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body);

                // アキュムレータから結果配列への書き込み
                let result_index = LowererUtils::compute_reduce_result_index(
                    result_strides,
                    result_offset,
                    dim,
                    reduce_axis,
                );
                let write_back_stmt = AstNode::Store {
                    target: Box::new(AstNode::Var(result_var.to_string())),
                    index: Box::new(result_index),
                    value: Box::new(AstNode::Var(acc_var.clone())),
                };

                // 結果の型を取得
                let result_dtype = DType::F32;

                return AstNode::Block {
                    scope: crate::ast::Scope {
                        declarations: vec![VariableDecl {
                            name: acc_var,
                            dtype: result_dtype,
                            constant: false,
                            size_expr: None,
                        }],
                    },
                    statements: vec![init_stmt, reduce_loop, write_back_stmt],
                };
            }
        }

        // 通常の次元: 単にループを生成
        let inner_body = Self::create_loops(
            input_shape,
            result_view,
            ast,
            input_vars,
            inputs,
            reduce_axis,
            result_var,
            reduce_op,
            initial_value,
            dim + 1,
        );

        let loop_var = format!("ridx{}", dim);

        LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body)
    }

    /// 縮約軸より後の次元のループを生成し、その内側でアキュムレータを使用
    #[allow(clippy::too_many_arguments)]
    fn create_post_reduce_loops_with_accumulator(
        input_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        _result_view: &crate::graph::shape::view::View,
        ast: &AstNode,
        input_vars: &[String],
        inputs: &[GraphNode],
        reduce_axis: usize,
        result_var: &str,
        reduce_op: &ReduceOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        if dim >= input_shape.len() {
            // 全ての後続次元のループを生成した：アキュムレータを使った縮約を実行
            let acc_var = format!("acc{}", reduce_axis);

            // アキュムレータへの累積を行う縮約ループ
            let inner_body = Self::create_with_accumulator(
                input_shape,
                ast,
                input_vars,
                inputs,
                reduce_axis,
                &acc_var,
                reduce_op,
                reduce_axis + 1,
            );

            let loop_var = format!("ridx{}", reduce_axis);

            // アキュムレータの初期化
            let init_stmt = AstNode::Assign(acc_var.clone(), Box::new(initial_value.clone()));

            // 縮約ループ
            let reduce_loop = LowererUtils::create_dimension_loop(
                loop_var,
                &input_shape[reduce_axis],
                inner_body,
            );

            // アキュムレータから結果配列への書き込み
            let result_index = LowererUtils::compute_reduce_result_index(
                result_strides,
                result_offset,
                input_shape.len(),
                reduce_axis,
            );
            let write_back_stmt = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Var(acc_var.clone())),
            };

            // 結果の型を取得
            let result_dtype = DType::F32;

            return AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![VariableDecl {
                        name: acc_var,
                        dtype: result_dtype,
                        constant: false,
                        size_expr: None,
                    }],
                },
                statements: vec![init_stmt, reduce_loop, write_back_stmt],
            };
        }

        // 後続次元のループを生成
        let loop_var = format!("ridx{}", dim);
        let inner_body = Self::create_post_reduce_loops_with_accumulator(
            input_shape,
            result_strides,
            result_offset,
            _result_view,
            ast,
            input_vars,
            inputs,
            reduce_axis,
            result_var,
            reduce_op,
            initial_value,
            dim + 1,
        );

        LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body)
    }

    /// アキュムレータ変数を使ったFusedElementwiseReduce縮約ループの本体を生成
    #[allow(clippy::too_many_arguments)]
    fn create_with_accumulator(
        input_shape: &[crate::graph::shape::Expr],
        ast: &AstNode,
        input_vars: &[String],
        inputs: &[GraphNode],
        reduce_axis: usize,
        acc_var: &str,
        reduce_op: &ReduceOp,
        dim: usize,
    ) -> AstNode {
        if dim >= input_shape.len() {
            // 全ての次元を処理した：アキュムレータに累積
            let fused_value = FusedElementwiseLowerer::replace_captures_with_input_refs(
                ast,
                input_vars,
                inputs,
                &vec![crate::graph::shape::Expr::from(0); input_shape.len()],
                &crate::graph::shape::Expr::from(0),
                input_shape.len(),
            );

            // 縮約操作: acc = acc op fused_value
            let operation_result = match reduce_op {
                ReduceOp::Add => AstNode::Add(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(fused_value),
                ),
                ReduceOp::Mul => AstNode::Mul(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(fused_value),
                ),
                ReduceOp::Max => AstNode::Max(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(fused_value),
                ),
            };

            return AstNode::Assign(acc_var.to_string(), Box::new(operation_result));
        }

        if dim == reduce_axis {
            // 縮約軸は既に外側で処理されているので、ここでは単に次の次元へ
            return Self::create_with_accumulator(
                input_shape,
                ast,
                input_vars,
                inputs,
                reduce_axis,
                acc_var,
                reduce_op,
                dim + 1,
            );
        }

        if dim > reduce_axis {
            // 縮約軸より後の次元は外側のループで既に処理されているので、単に次へ進む
            return Self::create_with_accumulator(
                input_shape,
                ast,
                input_vars,
                inputs,
                reduce_axis,
                acc_var,
                reduce_op,
                dim + 1,
            );
        }

        // 縮約軸より前の通常の次元: ループを生成
        let loop_var = format!("ridx{}", dim);
        let inner_body = Self::create_with_accumulator(
            input_shape,
            ast,
            input_vars,
            inputs,
            reduce_axis,
            acc_var,
            reduce_op,
            dim + 1,
        );

        LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body)
    }
}
