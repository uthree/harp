use super::utils::LowererUtils;
use crate::ast::{AstNode, DType, VariableDecl};
use crate::graph::{ops::ElementwiseOp, GraphNode};

/// Elementwise演算の lowering を担当
pub(super) struct ElementwiseLowerer;

impl ElementwiseLowerer {
    /// Elementwise演算をlowerする
    pub fn lower(
        node: &GraphNode,
        op: &ElementwiseOp,
        mut get_var: impl FnMut(&GraphNode) -> String,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        let result_var = get_var(node);

        // 出力ノードの場合は配列を宣言しない（引数として渡される）
        if !result_var.starts_with("output_") {
            // テンソルの場合は配列として宣言する必要がある
            let total_size = LowererUtils::compute_total_size(&node.view);
            let result_dtype = if let Some(size) = total_size {
                // サイズが静的に決定できる場合は固定サイズ配列型
                DType::Vec(Box::new(node.dtype.clone()), size)
            } else {
                // 動的サイズの場合はポインタ型（将来的にmallocで対応）
                todo!("Dynamic size arrays not yet supported")
            };

            declarations.push(VariableDecl {
                name: result_var.clone(),
                dtype: result_dtype,
                constant: false,
                size_expr: None,
            });
        }

        // ループでテンソルの各要素を処理
        let body = match op {
            ElementwiseOp::Add(lhs, rhs) => {
                let lhs_var = get_var(lhs);
                let rhs_var = get_var(rhs);
                Self::create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Add(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Mul(lhs, rhs) => {
                let lhs_var = get_var(lhs);
                let rhs_var = get_var(rhs);
                Self::create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Mul(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Max(lhs, rhs) => {
                let lhs_var = get_var(lhs);
                let rhs_var = get_var(rhs);
                Self::create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Max(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Mod(lhs, rhs) => {
                let lhs_var = get_var(lhs);
                let rhs_var = get_var(rhs);
                Self::create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Rem(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Neg(operand) => {
                let operand_var = get_var(operand);
                Self::create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    AstNode::Neg(Box::new(x))
                })
            }
            ElementwiseOp::Recip(operand) => {
                let operand_var = get_var(operand);
                Self::create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.recip()
                })
            }
            ElementwiseOp::Sin(operand) => {
                let operand_var = get_var(operand);
                Self::create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.sin()
                })
            }
            ElementwiseOp::Sqrt(operand) => {
                let operand_var = get_var(operand);
                Self::create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.sqrt()
                })
            }
            ElementwiseOp::Log2(operand) => {
                let operand_var = get_var(operand);
                Self::create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.log2()
                })
            }
            ElementwiseOp::Exp2(operand) => {
                let operand_var = get_var(operand);
                Self::create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.exp2()
                })
            }
        };

        Some(body)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_elementwise_loop<F>(
        result_node: &GraphNode,
        lhs_node: &GraphNode,
        rhs_node: &GraphNode,
        result_var: &str,
        lhs_var: &str,
        rhs_var: &str,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode, AstNode) -> AstNode + Clone,
    {
        // viewから形状情報を取得
        let result_view = &result_node.view;
        let lhs_view = &lhs_node.view;
        let rhs_view = &rhs_node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: _result_shape,
                strides: result_strides,
                offset: result_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: lhs_strides,
                offset: lhs_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: rhs_strides,
                offset: rhs_offset,
            },
        ) = (result_view, lhs_view, rhs_view);

        // 多重ループを生成
        Self::create_nested_loops(
            _result_shape,
            result_strides,
            result_offset,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            result_var,
            lhs_var,
            rhs_var,
            0,
            op,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_nested_loops<F>(
        shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        lhs_strides: &[crate::graph::shape::Expr],
        lhs_offset: &crate::graph::shape::Expr,
        rhs_strides: &[crate::graph::shape::Expr],
        rhs_offset: &crate::graph::shape::Expr,
        result_var: &str,
        lhs_var: &str,
        rhs_var: &str,
        dim: usize,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode, AstNode) -> AstNode + Clone,
    {
        if dim >= shape.len() {
            // 最内ループ: 実際の計算を実行
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);
            let lhs_index = LowererUtils::compute_memory_index(lhs_strides, lhs_offset, dim);
            let rhs_index = LowererUtils::compute_memory_index(rhs_strides, rhs_offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(op(
                    AstNode::Deref(Box::new(AstNode::Var(lhs_var.to_string()) + lhs_index)),
                    AstNode::Deref(Box::new(AstNode::Var(rhs_var.to_string()) + rhs_index)),
                )),
            }
        } else {
            // 再帰的にネストしたループを作成
            let loop_var = format!("i{}", dim);
            let inner_body = Self::create_nested_loops(
                shape,
                result_strides,
                result_offset,
                lhs_strides,
                lhs_offset,
                rhs_strides,
                rhs_offset,
                result_var,
                lhs_var,
                rhs_var,
                dim + 1,
                op,
            );

            // shape[dim]をAstNodeに変換
            let max_iter = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(max_iter),
                body: Box::new(inner_body),
            }
        }
    }

    fn create_unary_elementwise_loop<F>(
        result_node: &GraphNode,
        operand_node: &GraphNode,
        result_var: &str,
        operand_var: &str,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode) -> AstNode + Clone,
    {
        // viewから形状情報を取得
        let result_view = &result_node.view;
        let operand_view = &operand_node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: _result_shape,
                strides: result_strides,
                offset: result_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: operand_strides,
                offset: operand_offset,
            },
        ) = (result_view, operand_view);

        // 多重ループを生成
        Self::create_unary_nested_loops(
            _result_shape,
            result_strides,
            result_offset,
            operand_strides,
            operand_offset,
            result_var,
            operand_var,
            0,
            op,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_unary_nested_loops<F>(
        shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        operand_strides: &[crate::graph::shape::Expr],
        operand_offset: &crate::graph::shape::Expr,
        result_var: &str,
        operand_var: &str,
        dim: usize,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode) -> AstNode + Clone,
    {
        if dim >= shape.len() {
            // 最内ループ: 実際の計算を実行
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);
            let operand_index =
                LowererUtils::compute_memory_index(operand_strides, operand_offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(op(AstNode::Deref(Box::new(
                    AstNode::Var(operand_var.to_string()) + operand_index,
                )))),
            }
        } else {
            // 再帰的にネストしたループを作成
            let loop_var = format!("i{}", dim);
            let inner_body = Self::create_unary_nested_loops(
                shape,
                result_strides,
                result_offset,
                operand_strides,
                operand_offset,
                result_var,
                operand_var,
                dim + 1,
                op,
            );

            // shape[dim]をAstNodeに変換
            let max_iter = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(max_iter),
                body: Box::new(inner_body),
            }
        }
    }
}
