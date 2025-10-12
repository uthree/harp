use super::utils::LowererUtils;
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::{ops::ReduceOp, shape::view::View, GraphNode};

/// FusedReduce演算のコード生成を行う構造体
pub(super) struct FusedReduceLowerer;

impl FusedReduceLowerer {
    /// FusedReduce演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower(
        node: &GraphNode,
        op: &ReduceOp,
        axes: &[usize],
        input: &GraphNode,
        declarations: &mut Vec<VariableDecl>,
        mut get_var: impl FnMut(&GraphNode) -> String,
    ) -> Option<AstNode> {
        let result_var = get_var(node);
        let input_var = get_var(input);

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
            ReduceOp::Add => AstNode::Const(crate::ast::ConstLiteral::F32(0.0)),
            ReduceOp::Mul => AstNode::Const(crate::ast::ConstLiteral::F32(1.0)),
            ReduceOp::Max => AstNode::Const(crate::ast::ConstLiteral::F32(f32::NEG_INFINITY)),
        };

        // 多重ループでreduce操作を実行
        Some(Self::create_loops(
            input_view,
            result_view,
            axes,
            &input_var,
            &result_var,
            op,
            initial_value,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_loops(
        input_view: &View,
        result_view: &View,
        reduce_axes: &[usize],
        input_var: &str,
        result_var: &str,
        reduce_op: &ReduceOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        let (
            View::Linear {
                shape: input_shape,
                strides: input_strides,
                offset: input_offset,
            },
            View::Linear {
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);
        let is_reduce_axis = reduce_axes.contains(&dim);

        if dim >= input_shape.len() {
            // 全ての次元を処理した：縮約操作を実行
            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());
            let result_index = LowererUtils::compute_multi_reduce_result_index(
                result_strides,
                result_offset,
                input_shape.len(),
                reduce_axes,
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

        if is_reduce_axis && dim == *reduce_axes.iter().min().unwrap() {
            // 最初の縮約軸の処理
            // アキュムレータ変数は、最初の縮約軸以降に非縮約軸が存在しない場合のみ使用可能
            let first_reduce_axis = *reduce_axes.iter().min().unwrap();
            let has_non_reduce_after =
                (first_reduce_axis + 1..input_shape.len()).any(|d| !reduce_axes.contains(&d));
            let use_accumulator = !has_non_reduce_after;

            if use_accumulator {
                // アキュムレータ変数を使った縮約
                let acc_var = format!("acc{}", dim);

                // アキュムレータへの累積を行う内部ループ
                let inner_body = Self::create_with_accumulator(
                    input_view,
                    reduce_axes,
                    input_var,
                    &acc_var,
                    reduce_op,
                    dim + 1,
                );

                let loop_var = format!("ridx{}", dim);
                let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

                // アキュムレータの初期化
                let init_stmt = AstNode::Assign(acc_var.clone(), Box::new(initial_value.clone()));

                // 縮約ループ（現在の次元から開始）
                let reduce_loop = AstNode::Range {
                    counter_name: loop_var,
                    start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                    max: Box::new(shape_size),
                    step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                    body: Box::new(inner_body),
                    unroll: None,
                };

                // アキュムレータから結果配列への書き込み
                let result_index = LowererUtils::compute_multi_reduce_result_index(
                    result_strides,
                    result_offset,
                    dim,
                    reduce_axes,
                );
                let write_back_stmt = AstNode::Store {
                    target: Box::new(AstNode::Var(result_var.to_string())),
                    index: Box::new(result_index),
                    value: Box::new(AstNode::Var(acc_var.clone())),
                };

                return AstNode::Block {
                    scope: crate::ast::Scope {
                        declarations: vec![VariableDecl {
                            name: acc_var,
                            dtype: DType::F32, // 型を仮定
                            constant: false,
                            size_expr: None,
                        }],
                    },
                    statements: vec![init_stmt, reduce_loop, write_back_stmt],
                };
            } else {
                // 配列ベースの縮約（元の実装）
                let init_loop = Self::create_init_loops(
                    input_view,
                    result_view,
                    reduce_axes,
                    result_var,
                    initial_value.clone(),
                    dim,
                    0,
                );

                let loop_var = format!("ridx{}", dim);
                let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

                let inner_body = Self::create_loops(
                    input_view,
                    result_view,
                    reduce_axes,
                    input_var,
                    result_var,
                    reduce_op,
                    initial_value.clone(),
                    dim + 1,
                );

                let reduce_loop = AstNode::Range {
                    counter_name: loop_var,
                    start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                    max: Box::new(shape_size),
                    step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                    body: Box::new(inner_body),
                    unroll: None,
                };

                return AstNode::Block {
                    scope: crate::ast::Scope {
                        declarations: vec![],
                    },
                    statements: vec![init_loop, reduce_loop],
                };
            }
        }

        // 通常のループ（reduce軸でないか、最初のreduce軸でない）
        let loop_var = format!("ridx{}", dim);
        let inner_body = Self::create_loops(
            input_view,
            result_view,
            reduce_axes,
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

    /// アキュムレータ変数を使ったFusedReduce縮約ループの本体を生成
    fn create_with_accumulator(
        input_view: &View,
        reduce_axes: &[usize],
        input_var: &str,
        acc_var: &str,
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
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
                ReduceOp::Mul => AstNode::Mul(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
                ReduceOp::Max => AstNode::Max(
                    Box::new(AstNode::Var(acc_var.to_string())),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
            };

            return AstNode::Assign(acc_var.to_string(), Box::new(operation_result));
        }

        if reduce_axes.contains(&dim) {
            // 縮約軸の場合
            // 最初の縮約軸は外側で処理されているので、それ以外の縮約軸についてループを生成
            if dim == *reduce_axes.iter().min().unwrap() {
                // 最初の縮約軸は既に外側で処理されているので、次の次元へ
                return Self::create_with_accumulator(
                    input_view,
                    reduce_axes,
                    input_var,
                    acc_var,
                    reduce_op,
                    dim + 1,
                );
            } else {
                // 2番目以降の縮約軸: ループを生成
                let loop_var = format!("ridx{}", dim);
                let inner_body = Self::create_with_accumulator(
                    input_view,
                    reduce_axes,
                    input_var,
                    acc_var,
                    reduce_op,
                    dim + 1,
                );

                let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

                return AstNode::Range {
                    counter_name: loop_var,
                    start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                    max: Box::new(shape_size),
                    step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                    body: Box::new(inner_body),
                    unroll: None,
                };
            }
        }

        // 通常の次元: ループを生成
        let loop_var = format!("ridx{}", dim);
        let inner_body = Self::create_with_accumulator(
            input_view,
            reduce_axes,
            input_var,
            acc_var,
            reduce_op,
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

    /// FusedReduceの初期化ループを作成（reduce軸でない次元のみ）
    /// start_dim: 初期化ループを開始する次元（最初のreduce軸）
    /// dim: 現在処理中の次元
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_init_loops(
        input_view: &View,
        result_view: &View,
        reduce_axes: &[usize],
        result_var: &str,
        initial_value: AstNode,
        start_dim: usize,
        dim: usize,
    ) -> AstNode {
        let (
            View::Linear {
                shape: input_shape, ..
            },
            View::Linear {
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);

        if dim >= input_shape.len() {
            // 全ての次元を処理した：初期化を実行
            let result_index = LowererUtils::compute_multi_reduce_result_index(
                result_strides,
                result_offset,
                input_shape.len(),
                reduce_axes,
            );

            return AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(initial_value),
            };
        }

        if dim < start_dim {
            // start_dimより前の次元はスキップ（すでに外側のループで処理されている）
            return Self::create_init_loops(
                input_view,
                result_view,
                reduce_axes,
                result_var,
                initial_value,
                start_dim,
                dim + 1,
            );
        }

        if reduce_axes.contains(&dim) {
            // reduce軸はスキップ
            return Self::create_init_loops(
                input_view,
                result_view,
                reduce_axes,
                result_var,
                initial_value,
                start_dim,
                dim + 1,
            );
        }

        // 通常のループ（reduce軸でない）
        let loop_var = format!("ridx{}", dim);
        let inner_body = Self::create_init_loops(
            input_view,
            result_view,
            reduce_axes,
            result_var,
            initial_value,
            start_dim,
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
