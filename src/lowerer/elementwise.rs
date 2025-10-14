use super::utils::LowererUtils;
use crate::ast::helper::{eq, less_than, select};
use crate::ast::{AstNode, VariableDecl};
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
        LowererUtils::declare_result_variable(&result_var, &node.view, &node.dtype, declarations);

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
            ElementwiseOp::LessThan(lhs, rhs) => {
                let lhs_var = get_var(lhs);
                let rhs_var = get_var(rhs);
                Self::create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    less_than,
                )
            }
            ElementwiseOp::Eq(lhs, rhs) => {
                let lhs_var = get_var(lhs);
                let rhs_var = get_var(rhs);
                Self::create_elementwise_loop(node, lhs, rhs, &result_var, &lhs_var, &rhs_var, eq)
            }
            ElementwiseOp::Select(cond, true_val, false_val) => {
                let cond_var = get_var(cond);
                let true_var = get_var(true_val);
                let false_var = get_var(false_val);
                Self::create_ternary_elementwise_loop(
                    node,
                    cond,
                    true_val,
                    false_val,
                    &result_var,
                    &cond_var,
                    &true_var,
                    &false_var,
                )
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
            lhs_view,
            lhs_strides,
            lhs_offset,
            rhs_view,
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
        lhs_view: &crate::graph::shape::view::View,
        lhs_strides: &[crate::graph::shape::Expr],
        lhs_offset: &crate::graph::shape::Expr,
        rhs_view: &crate::graph::shape::view::View,
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

            // lhs/rhsがスカラー（shape.is_empty()）の場合は直接変数を使用、そうでなければデリファレンス
            let lhs_value = if lhs_view.shape().is_empty() {
                AstNode::Var(lhs_var.to_string())
            } else {
                let lhs_index = LowererUtils::compute_memory_index(lhs_strides, lhs_offset, dim);
                AstNode::Deref(Box::new(AstNode::Var(lhs_var.to_string()) + lhs_index))
            };

            let rhs_value = if rhs_view.shape().is_empty() {
                AstNode::Var(rhs_var.to_string())
            } else {
                let rhs_index = LowererUtils::compute_memory_index(rhs_strides, rhs_offset, dim);
                AstNode::Deref(Box::new(AstNode::Var(rhs_var.to_string()) + rhs_index))
            };

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(op(lhs_value, rhs_value)),
            }
        } else {
            // 再帰的にネストしたループを作成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_nested_loops(
                shape,
                result_strides,
                result_offset,
                lhs_view,
                lhs_strides,
                lhs_offset,
                rhs_view,
                rhs_strides,
                rhs_offset,
                result_var,
                lhs_var,
                rhs_var,
                dim + 1,
                op,
            );

            LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body)
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
            operand_view,
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
        operand_view: &crate::graph::shape::view::View,
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

            // operandがスカラーの場合は直接変数を使用、そうでなければデリファレンス
            let operand_value = if operand_view.shape().is_empty() {
                AstNode::Var(operand_var.to_string())
            } else {
                let operand_index =
                    LowererUtils::compute_memory_index(operand_strides, operand_offset, dim);
                AstNode::Deref(Box::new(
                    AstNode::Var(operand_var.to_string()) + operand_index,
                ))
            };

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(op(operand_value)),
            }
        } else {
            // 再帰的にネストしたループを作成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_unary_nested_loops(
                shape,
                result_strides,
                result_offset,
                operand_view,
                operand_strides,
                operand_offset,
                result_var,
                operand_var,
                dim + 1,
                op,
            );

            LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body)
        }
    }

    /// Selectのための3引数版のelementwiseループ生成
    #[allow(clippy::too_many_arguments)]
    fn create_ternary_elementwise_loop(
        result_node: &GraphNode,
        cond_node: &GraphNode,
        true_node: &GraphNode,
        false_node: &GraphNode,
        result_var: &str,
        cond_var: &str,
        true_var: &str,
        false_var: &str,
    ) -> AstNode {
        // viewから形状情報を取得
        let result_view = &result_node.view;
        let cond_view = &cond_node.view;
        let true_view = &true_node.view;
        let false_view = &false_node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: result_shape,
                strides: result_strides,
                offset: result_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: cond_strides,
                offset: cond_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: true_strides,
                offset: true_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: false_strides,
                offset: false_offset,
            },
        ) = (result_view, cond_view, true_view, false_view);

        // 多重ループを生成
        Self::create_ternary_nested_loops(
            result_shape,
            result_strides,
            result_offset,
            cond_view,
            cond_strides,
            cond_offset,
            true_view,
            true_strides,
            true_offset,
            false_view,
            false_strides,
            false_offset,
            result_var,
            cond_var,
            true_var,
            false_var,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_ternary_nested_loops(
        shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        cond_view: &crate::graph::shape::view::View,
        cond_strides: &[crate::graph::shape::Expr],
        cond_offset: &crate::graph::shape::Expr,
        true_view: &crate::graph::shape::view::View,
        true_strides: &[crate::graph::shape::Expr],
        true_offset: &crate::graph::shape::Expr,
        false_view: &crate::graph::shape::view::View,
        false_strides: &[crate::graph::shape::Expr],
        false_offset: &crate::graph::shape::Expr,
        result_var: &str,
        cond_var: &str,
        true_var: &str,
        false_var: &str,
        dim: usize,
    ) -> AstNode {
        if dim >= shape.len() {
            // 最内ループ: 実際の計算を実行
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);

            // 各オペランドの値を取得
            let cond_value = if cond_view.shape().is_empty() {
                AstNode::Var(cond_var.to_string())
            } else {
                let cond_index = LowererUtils::compute_memory_index(cond_strides, cond_offset, dim);
                AstNode::Deref(Box::new(AstNode::Var(cond_var.to_string()) + cond_index))
            };

            let true_value = if true_view.shape().is_empty() {
                AstNode::Var(true_var.to_string())
            } else {
                let true_index = LowererUtils::compute_memory_index(true_strides, true_offset, dim);
                AstNode::Deref(Box::new(AstNode::Var(true_var.to_string()) + true_index))
            };

            let false_value = if false_view.shape().is_empty() {
                AstNode::Var(false_var.to_string())
            } else {
                let false_index =
                    LowererUtils::compute_memory_index(false_strides, false_offset, dim);
                AstNode::Deref(Box::new(AstNode::Var(false_var.to_string()) + false_index))
            };

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(select(cond_value, true_value, false_value)),
            }
        } else {
            // 再帰的にネストしたループを作成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_ternary_nested_loops(
                shape,
                result_strides,
                result_offset,
                cond_view,
                cond_strides,
                cond_offset,
                true_view,
                true_strides,
                true_offset,
                false_view,
                false_strides,
                false_offset,
                result_var,
                cond_var,
                true_var,
                false_var,
                dim + 1,
            );

            LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body)
        }
    }
}
