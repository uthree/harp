//! 式構築ユーティリティ
//!
//! ElementwiseOpからAstNode式への変換を提供します。

use crate::ast::{AstNode, helper::*};
use crate::tensor::ElementwiseOp;

/// ElementwiseOpから演算式を構築
///
/// Wildcardプレースホルダーを使用して式を構築します。
/// - 二項演算: wildcard("0"), wildcard("1")
/// - 単項演算: wildcard("0")
pub fn build_elementwise_expr(op: &ElementwiseOp) -> AstNode {
    match op {
        // 二項演算
        ElementwiseOp::Add => wildcard("0") + wildcard("1"),
        ElementwiseOp::Mul => wildcard("0") * wildcard("1"),
        ElementwiseOp::Max => max(wildcard("0"), wildcard("1")),
        ElementwiseOp::Rem => wildcard("0") % wildcard("1"),
        ElementwiseOp::Idiv => idiv(wildcard("0"), wildcard("1")),

        // 単項演算
        ElementwiseOp::Neg => const_f32(-1.0) * wildcard("0"),
        ElementwiseOp::Recip => recip(wildcard("0")),
        ElementwiseOp::Log2 => log2(wildcard("0")),
        ElementwiseOp::Exp2 => exp2(wildcard("0")),
        ElementwiseOp::Sin => sin(wildcard("0")),
        ElementwiseOp::Sqrt => sqrt(wildcard("0")),
        ElementwiseOp::Floor => AstNode::Floor(Box::new(wildcard("0"))),

        // ビット演算（二項）
        ElementwiseOp::BitAnd => bitand(wildcard("0"), wildcard("1")),
        ElementwiseOp::BitOr => bitor(wildcard("0"), wildcard("1")),
        ElementwiseOp::BitXor => bitxor(wildcard("0"), wildcard("1")),
        ElementwiseOp::Shl => shl(wildcard("0"), wildcard("1")),
        ElementwiseOp::Shr => shr(wildcard("0"), wildcard("1")),

        // ビット演算（単項）
        ElementwiseOp::BitNot => bitnot(wildcard("0")),
    }
}

/// Reduce演算用のパススルー式を構築
///
/// 入力をそのままアキュムレータに渡すための式
pub fn build_reduce_identity_expr() -> AstNode {
    wildcard("0")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_expr() {
        let expr = build_elementwise_expr(&ElementwiseOp::Add);
        match expr {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_mul_expr() {
        let expr = build_elementwise_expr(&ElementwiseOp::Mul);
        match expr {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_unary_neg_expr() {
        let expr = build_elementwise_expr(&ElementwiseOp::Neg);
        match expr {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node for Neg"),
        }
    }

    #[test]
    fn test_unary_recip_expr() {
        let expr = build_elementwise_expr(&ElementwiseOp::Recip);
        match expr {
            AstNode::Recip(_) => {}
            _ => panic!("Expected Recip node"),
        }
    }

    #[test]
    fn test_reduce_identity() {
        let expr = build_reduce_identity_expr();
        match expr {
            AstNode::Wildcard(name) => assert_eq!(name, "0"),
            _ => panic!("Expected Wildcard node"),
        }
    }
}
