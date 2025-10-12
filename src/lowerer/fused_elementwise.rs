use super::utils::LowererUtils;
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::GraphNode;

/// FusedElementwise演算のコード生成を行う構造体
pub(super) struct FusedElementwiseLowerer;

impl FusedElementwiseLowerer {
    /// FusedElementwise演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower(
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
        Some(Self::create_loop(
            &node.view,
            ast,
            &input_vars,
            inputs,
            &result_var,
            0,
        ))
    }

    fn create_loop(
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
        let loop_var = format!("ridx{}", dim);
        let inner_body = Self::create_loop(view, ast, input_vars, inputs, result_var, dim + 1);

        let max_iter = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

        AstNode::Range {
            counter_name: loop_var,
            start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
            max: Box::new(max_iter),
            step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
            body: Box::new(inner_body),
            unroll: None,
        }
    }

    /// AstNode内のCaptureを実際の入力変数への参照に置き換え
    pub(super) fn replace_captures_with_input_refs(
        ast: &AstNode,
        input_vars: &[String],
        inputs: &[GraphNode],
        _strides: &[crate::graph::shape::Expr],
        _offset: &crate::graph::shape::Expr,
        ndim: usize,
    ) -> AstNode {
        match ast {
            AstNode::Capture(idx) => {
                // Capture(idx)を inputs[idx] の参照に置き換え
                let input_var = &input_vars[*idx];
                let input_view = &inputs[*idx].view;

                // スカラー（空の形状）の場合は直接変数を使用
                if input_view.shape().is_empty() {
                    return AstNode::Var(input_var.clone());
                }

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
                            child, input_vars, inputs, _strides, _offset, ndim,
                        )
                    })
                    .collect();
                new_ast.replace_children(children)
            }
        }
    }
}
