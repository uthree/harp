//! Eager Fusion - 演算呼び出し時の融合判定
//!
//! 演算呼び出し時に即座に融合を試み、可能であれば融合演算を返す。
//!
//! ## 融合パターン
//!
//! 新しいMapReduce統一設計では、演算は全てMapReduce variant で表現される:
//! - Elementwise: reduce_op = None, axes = []
//! - Reduce: expr = Wildcard("0"), reduce_op = Some(...), axes = [...]
//! - Fused: 任意のexpr + reduce_op
//!
//! ## 融合条件
//!
//! 1. **View互換**: 同一または互換性のあるView
//! 2. **dtype一致**: 同一データ型
//! 3. **単一消費**: 親テンソルが子演算のみに使用される（所有権ベース設計で保証）

use crate::ast::{AstNode, Literal, helper::*};
use crate::tensor::ops::{ElementwiseOp, TensorOp};

/// ElementwiseOpをAstNodeに変換する
///
/// # Arguments
/// * `op` - 変換するElementwiseOp
/// * `input_indices` - 入力のプレースホルダーインデックス
///
/// # Returns
/// 対応するAstNode式
pub fn elementwise_to_ast(op: &ElementwiseOp, input_indices: &[usize]) -> AstNode {
    match op {
        // 二項演算
        ElementwiseOp::Add => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            lhs + rhs
        }
        ElementwiseOp::Mul => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            lhs * rhs
        }
        ElementwiseOp::Max => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            max(lhs, rhs)
        }
        ElementwiseOp::Rem => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            rem(lhs, rhs)
        }
        ElementwiseOp::Idiv => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            idiv(lhs, rhs)
        }

        // 単項演算
        ElementwiseOp::Neg => {
            // Neg(x) = -1.0 * x
            let input = wildcard(input_indices[0].to_string());
            AstNode::Const(Literal::F32(-1.0)) * input
        }
        ElementwiseOp::Recip => {
            let input = wildcard(input_indices[0].to_string());
            recip(input)
        }
        ElementwiseOp::Log2 => {
            let input = wildcard(input_indices[0].to_string());
            log2(input)
        }
        ElementwiseOp::Exp2 => {
            let input = wildcard(input_indices[0].to_string());
            exp2(input)
        }
        ElementwiseOp::Sin => {
            let input = wildcard(input_indices[0].to_string());
            sin(input)
        }
        ElementwiseOp::Sqrt => {
            let input = wildcard(input_indices[0].to_string());
            sqrt(input)
        }
        ElementwiseOp::Floor => {
            let input = wildcard(input_indices[0].to_string());
            floor(input)
        }

        // ビット演算（二項）
        ElementwiseOp::BitAnd => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            bitand(lhs, rhs)
        }
        ElementwiseOp::BitOr => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            bitor(lhs, rhs)
        }
        ElementwiseOp::BitXor => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            bitxor(lhs, rhs)
        }
        ElementwiseOp::Shl => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            shl(lhs, rhs)
        }
        ElementwiseOp::Shr => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            shr(lhs, rhs)
        }

        // ビット演算（単項）
        ElementwiseOp::BitNot => {
            let input = wildcard(input_indices[0].to_string());
            bitnot(input)
        }
    }
}

/// Wildcard名を別の式で置換
pub fn substitute_wildcard(expr: &AstNode, name: &str, replacement: &AstNode) -> AstNode {
    match expr {
        AstNode::Wildcard(n) if n == name => replacement.clone(),
        AstNode::Add(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Add(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Mul(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Mul(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Recip(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Recip(Box::new(new_inner))
        }
        AstNode::Log2(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Log2(Box::new(new_inner))
        }
        AstNode::Exp2(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Exp2(Box::new(new_inner))
        }
        AstNode::Sin(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Sin(Box::new(new_inner))
        }
        AstNode::Sqrt(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Sqrt(Box::new(new_inner))
        }
        AstNode::Floor(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Floor(Box::new(new_inner))
        }
        AstNode::Max(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Max(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Rem(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Rem(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Idiv(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Idiv(Box::new(new_lhs), Box::new(new_rhs))
        }
        // その他のノードはそのまま返す
        _ => expr.clone(),
    }
}

/// AstNode内の最大Wildcardインデックスを見つける
pub fn find_max_wildcard_index(expr: &AstNode) -> usize {
    match expr {
        AstNode::Wildcard(name) => name.parse::<usize>().unwrap_or(0),
        AstNode::Add(lhs, rhs)
        | AstNode::Mul(lhs, rhs)
        | AstNode::Max(lhs, rhs)
        | AstNode::Rem(lhs, rhs)
        | AstNode::Idiv(lhs, rhs) => find_max_wildcard_index(lhs).max(find_max_wildcard_index(rhs)),
        AstNode::Recip(inner)
        | AstNode::Log2(inner)
        | AstNode::Exp2(inner)
        | AstNode::Sin(inner)
        | AstNode::Sqrt(inner)
        | AstNode::Floor(inner) => find_max_wildcard_index(inner),
        _ => 0,
    }
}

/// 融合が可能かどうかをチェック
///
/// 新しいCompute統一設計では、Compute同士の融合を判定する
pub fn can_fuse(parent: &TensorOp, child: &TensorOp) -> bool {
    match (parent, child) {
        // Elementwise Compute + Elementwise Compute は融合可能
        (
            TensorOp::MapReduce {
                reduce_op: None,
                axes: p_axes,
                ..
            },
            TensorOp::MapReduce {
                reduce_op: None,
                axes: c_axes,
                ..
            },
        ) if p_axes.is_empty() && c_axes.is_empty() => true,

        // Elementwise Compute + Reduce Compute は融合可能
        (
            TensorOp::MapReduce {
                reduce_op: None,
                axes: p_axes,
                ..
            },
            TensorOp::MapReduce {
                reduce_op: Some(_), ..
            },
        ) if p_axes.is_empty() => true,

        // その他は融合不可
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_to_ast_unary() {
        let expr = elementwise_to_ast(&ElementwiseOp::Recip, &[0]);
        // Recip(Wildcard("0"))
        match expr {
            AstNode::Recip(inner) => match *inner {
                AstNode::Wildcard(name) => assert_eq!(name, "0"),
                _ => panic!("Expected Wildcard"),
            },
            _ => panic!("Expected Recip"),
        }
    }

    #[test]
    fn test_elementwise_to_ast_neg() {
        let expr = elementwise_to_ast(&ElementwiseOp::Neg, &[0]);
        // Mul(Const(-1.0), Wildcard("0"))
        match expr {
            AstNode::Mul(lhs, rhs) => {
                match *lhs {
                    AstNode::Const(Literal::F32(v)) => assert_eq!(v, -1.0),
                    _ => panic!("Expected Const(-1.0) for lhs"),
                }
                match *rhs {
                    AstNode::Wildcard(name) => assert_eq!(name, "0"),
                    _ => panic!("Expected Wildcard for rhs"),
                }
            }
            _ => panic!("Expected Mul"),
        }
    }

    #[test]
    fn test_elementwise_to_ast_binary() {
        let expr = elementwise_to_ast(&ElementwiseOp::Add, &[0, 1]);
        // Add(Wildcard("0"), Wildcard("1"))
        match expr {
            AstNode::Add(lhs, rhs) => {
                match *lhs {
                    AstNode::Wildcard(name) => assert_eq!(name, "0"),
                    _ => panic!("Expected Wildcard for lhs"),
                }
                match *rhs {
                    AstNode::Wildcard(name) => assert_eq!(name, "1"),
                    _ => panic!("Expected Wildcard for rhs"),
                }
            }
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_find_max_wildcard_index() {
        let expr = AstNode::Add(
            Box::new(AstNode::Wildcard("0".to_string())),
            Box::new(AstNode::Wildcard("2".to_string())),
        );
        assert_eq!(find_max_wildcard_index(&expr), 2);

        let expr2 = AstNode::Recip(Box::new(AstNode::Wildcard("1".to_string())));
        assert_eq!(find_max_wildcard_index(&expr2), 1);
    }
}
