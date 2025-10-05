use super::utils::LowererUtils;
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::{ops::ReduceOp, GraphNode};

/// Fused演算のコード生成を行う構造体
pub(super) struct FusedLowerer;

impl FusedLowerer {
    /// FusedElementwise演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_fused_elementwise(
        node: &GraphNode,
        ast: &AstNode,
        inputs: &[GraphNode],
        declarations: &mut Vec<VariableDecl>,
        mut get_var: impl FnMut(&GraphNode) -> String,
    ) -> Option<AstNode> {
        let result_var = get_var(node);

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

        // 入力の変数名を取得
        let input_vars: Vec<String> = inputs.iter().map(get_var).collect();

        // ループを生成
        Some(Self::create_fused_elementwise_loop(
            &node.view,
            ast,
            &input_vars,
            inputs,
            &result_var,
            0,
        ))
    }

    fn create_fused_elementwise_loop(
        view: &crate::graph::shape::view::View,
        ast: &AstNode,
        input_vars: &[String],
        inputs: &[GraphNode],
        result_var: &str,
        dim: usize,
    ) -> AstNode {
        let crate::graph::shape::view::View::Linear {
            shape,
            strides,
            offset,
        } = view;

        if dim >= shape.len() {
            // 最内レベル: 実際の計算を実行
            // AstNode内のCaptureを実際の入力参照に置き換え
            let value_ast = Self::replace_captures_with_input_refs(
                ast,
                input_vars,
                inputs,
                strides,
                offset,
                shape.len(),
            );

            // 結果を格納
            let result_index = LowererUtils::compute_memory_index(strides, offset, shape.len());
            return AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(value_ast),
            };
        }

        // ループを生成
        let loop_var = format!("i{}", dim);
        let inner_body =
            Self::create_fused_elementwise_loop(view, ast, input_vars, inputs, result_var, dim + 1);

        let max_iter = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

        AstNode::Range {
            counter_name: loop_var,
            max: Box::new(max_iter),
            body: Box::new(inner_body),
        }
    }

    /// AstNode内のCaptureを実際の入力変数への参照に置き換え
    pub(super) fn replace_captures_with_input_refs(
        ast: &AstNode,
        input_vars: &[String],
        inputs: &[GraphNode],
        strides: &[crate::graph::shape::Expr],
        offset: &crate::graph::shape::Expr,
        ndim: usize,
    ) -> AstNode {
        match ast {
            AstNode::Capture(idx) => {
                // Capture(idx)を inputs[idx] の参照に置き換え
                let input_var = &input_vars[*idx];
                let input_view = &inputs[*idx].view;

                // 入力のstrideとoffsetを取得
                let crate::graph::shape::view::View::Linear {
                    strides: input_strides,
                    offset: input_offset,
                    ..
                } = input_view;

                // 入力のindexを計算
                let input_index =
                    LowererUtils::compute_memory_index(input_strides, input_offset, ndim);

                // 参照を返す
                AstNode::Deref(Box::new(AstNode::Var(input_var.clone()) + input_index))
            }
            _ => {
                // 再帰的に子ノードを置き換え
                let new_ast = ast.clone();
                let children: Vec<AstNode> = new_ast
                    .children()
                    .iter()
                    .map(|child| {
                        Self::replace_captures_with_input_refs(
                            child, input_vars, inputs, strides, offset, ndim,
                        )
                    })
                    .collect();
                new_ast.replace_children(children)
            }
        }
    }

    /// FusedElementwiseReduce演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_fused_elementwise_reduce(
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
        Some(Self::create_fused_elementwise_reduce_loops(
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
    fn create_fused_elementwise_reduce_loops(
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
            let fused_value = Self::replace_captures_with_input_refs(
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
            // 縮約する次元: 初期化 + ループで累積
            let inner_body = Self::create_fused_elementwise_reduce_loops(
                input_shape,
                result_view,
                ast,
                input_vars,
                inputs,
                reduce_axis,
                result_var,
                reduce_op,
                initial_value.clone(),
                dim + 1,
            );

            let loop_var = format!("i{}", dim);
            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

            // 初期化：reduce軸より後の次元についてもループを生成
            // reduce_axis+1から始めることで、reduce軸より後の全ての次元をループ
            let init_stmt = Self::create_init_loops_after_reduce(
                input_shape,
                result_strides,
                result_offset,
                reduce_axis,
                result_var,
                initial_value,
                reduce_axis + 1, // reduce軸の次の次元から開始
            );

            let reduce_loop = AstNode::Range {
                counter_name: loop_var,
                max: Box::new(shape_size),
                body: Box::new(inner_body),
            };

            return AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![],
                },
                statements: vec![init_stmt, reduce_loop],
            };
        }

        // 通常の次元: 単にループを生成
        let inner_body = Self::create_fused_elementwise_reduce_loops(
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

        let loop_var = format!("i{}", dim);
        let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

        AstNode::Range {
            counter_name: loop_var,
            max: Box::new(shape_size),
            body: Box::new(inner_body),
        }
    }

    /// FusedReduce演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_fused_reduce(
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
        Some(Self::create_fused_reduce_loops(
            input_shape,
            input_strides,
            input_offset,
            result_strides,
            result_offset,
            axes,
            &input_var,
            &result_var,
            op,
            initial_value,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_fused_reduce_loops(
        input_shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        reduce_axes: &[usize],
        input_var: &str,
        result_var: &str,
        reduce_op: &ReduceOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
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
            // 最初の縮約軸: 初期化ループ + 縮約ループ
            // 初期化ループ: 最初のreduce軸以降の次元（reduce軸を除く）でループし、結果を初期化
            let init_loop = Self::create_init_loops_for_fused_reduce(
                input_shape,
                result_strides,
                result_offset,
                reduce_axes,
                result_var,
                initial_value.clone(),
                dim, // start_dim: 最初のreduce軸から開始（それより前の次元はすでにループの中）
                0,   // dim: 0から開始して、start_dim未満はスキップ
            );

            // 縮約ループ（現在の次元から開始）
            let loop_var = format!("i{}", dim);
            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

            let inner_body = Self::create_fused_reduce_loops(
                input_shape,
                input_strides,
                input_offset,
                result_strides,
                result_offset,
                reduce_axes,
                input_var,
                result_var,
                reduce_op,
                initial_value.clone(),
                dim + 1,
            );

            let reduce_loop = AstNode::Range {
                counter_name: loop_var,
                max: Box::new(shape_size),
                body: Box::new(inner_body),
            };

            return AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![],
                },
                statements: vec![init_loop, reduce_loop],
            };
        }

        // 通常のループ（reduce軸でないか、最初のreduce軸でない）
        let loop_var = format!("i{}", dim);
        let inner_body = Self::create_fused_reduce_loops(
            input_shape,
            input_strides,
            input_offset,
            result_strides,
            result_offset,
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
            max: Box::new(shape_size),
            body: Box::new(inner_body),
        }
    }

    /// FusedReduceの初期化ループを作成（reduce軸でない次元のみ）
    /// start_dim: 初期化ループを開始する次元（最初のreduce軸）
    /// dim: 現在処理中の次元
    pub(super) fn create_init_loops_for_fused_reduce(
        input_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        reduce_axes: &[usize],
        result_var: &str,
        initial_value: AstNode,
        start_dim: usize,
        dim: usize,
    ) -> AstNode {
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
            return Self::create_init_loops_for_fused_reduce(
                input_shape,
                result_strides,
                result_offset,
                reduce_axes,
                result_var,
                initial_value,
                start_dim,
                dim + 1,
            );
        }

        if reduce_axes.contains(&dim) {
            // reduce軸はスキップ
            return Self::create_init_loops_for_fused_reduce(
                input_shape,
                result_strides,
                result_offset,
                reduce_axes,
                result_var,
                initial_value,
                start_dim,
                dim + 1,
            );
        }

        // 通常のループ（reduce軸でない）
        let loop_var = format!("i{}", dim);
        let inner_body = Self::create_init_loops_for_fused_reduce(
            input_shape,
            result_strides,
            result_offset,
            reduce_axes,
            result_var,
            initial_value,
            start_dim,
            dim + 1,
        );

        let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[dim]);

        AstNode::Range {
            counter_name: loop_var,
            max: Box::new(shape_size),
            body: Box::new(inner_body),
        }
    }

    /// reduce軸より後の次元について初期化ループを生成
    /// reduce_axis+1からinput_shape.len()までの次元についてループを作成し、
    /// 各要素を初期値で初期化する
    pub(super) fn create_init_loops_after_reduce(
        input_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        reduce_axis: usize,
        result_var: &str,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        if dim >= input_shape.len() {
            // 全ての次元を処理した：初期化を実行
            let mut index_expr = result_offset.clone();
            let mut result_dim = 0;

            for input_dim in 0..input_shape.len() {
                if input_dim != reduce_axis {
                    let loop_var = crate::graph::shape::Expr::Var(format!("i{}", input_dim));
                    let term = loop_var * result_strides[result_dim].clone();
                    index_expr += term;
                    result_dim += 1;
                }
            }

            let simplified = index_expr.simplify();
            let result_index = LowererUtils::shape_expr_to_ast_node(&simplified);

            return AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(initial_value),
            };
        }

        if dim == reduce_axis {
            // reduce軸はスキップ
            return Self::create_init_loops_after_reduce(
                input_shape,
                result_strides,
                result_offset,
                reduce_axis,
                result_var,
                initial_value,
                dim + 1,
            );
        }

        // 通常のループ（reduce軸でない）
        let loop_var = format!("i{}", dim);
        let inner_body = Self::create_init_loops_after_reduce(
            input_shape,
            result_strides,
            result_offset,
            reduce_axis,
            result_var,
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

    /// FusedElementwiseCumulative演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_fused_elementwise_cumulative(
        _node: &GraphNode,
        _ast: &AstNode,
        _inputs: &[GraphNode],
        _op: &crate::graph::ops::CumulativeOp,
        _declarations: &mut Vec<VariableDecl>,
        mut _get_var: impl FnMut(&GraphNode) -> String,
    ) -> Option<AstNode> {
        // TODO: 実装
        todo!("FusedElementwiseCumulative not yet implemented in lowerer")
    }
}
