//! ビット演算最適化ルール
//!
//! 2の累乗の乗算をシフト演算に変換するなどの最適化ルール。

use crate::ast::pat::AstRewriteRule;
use crate::ast::{AstNode, Literal};
use std::rc::Rc;

/// 値が2の累乗かどうかをチェックするヘルパー関数
fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// 2の累乗の値からlog2を計算するヘルパー関数
fn log2_of_power_of_two(n: i64) -> i64 {
    n.trailing_zeros() as i64
}

/// 乗算を左シフトに変換: x * 2^n = x << n
pub fn mul_power_of_two_to_shift_right() -> Rc<AstRewriteRule> {
    AstRewriteRule::new(
        AstNode::Mul(
            Box::new(AstNode::Wildcard("a".to_string())),
            Box::new(AstNode::Wildcard("b".to_string())),
        ),
        |bindings| {
            let a = bindings.get("a").unwrap();
            let b = bindings.get("b").unwrap();

            if let AstNode::Const(Literal::I64(n)) = b
                && is_power_of_two(*n)
            {
                return AstNode::LeftShift(
                    Box::new(a.clone()),
                    Box::new(AstNode::Const(Literal::I64(log2_of_power_of_two(*n)))),
                );
            }

            // 条件に合わなければ元のノードを返す
            AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))
        },
        |bindings| {
            // 条件：bが2の累乗の定数
            if let Some(AstNode::Const(Literal::I64(n))) = bindings.get("b") {
                is_power_of_two(*n)
            } else {
                false
            }
        },
    )
}

/// 乗算を左シフトに変換: 2^n * x = x << n
pub fn mul_power_of_two_to_shift_left() -> Rc<AstRewriteRule> {
    AstRewriteRule::new(
        AstNode::Mul(
            Box::new(AstNode::Wildcard("a".to_string())),
            Box::new(AstNode::Wildcard("b".to_string())),
        ),
        |bindings| {
            let a = bindings.get("a").unwrap();
            let b = bindings.get("b").unwrap();

            if let AstNode::Const(Literal::I64(n)) = a
                && is_power_of_two(*n)
            {
                return AstNode::LeftShift(
                    Box::new(b.clone()),
                    Box::new(AstNode::Const(Literal::I64(log2_of_power_of_two(*n)))),
                );
            }

            // 条件に合わなければ元のノードを返す
            AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))
        },
        |bindings| {
            // 条件：aが2の累乗の定数
            if let Some(AstNode::Const(Literal::I64(n))) = bindings.get("a") {
                is_power_of_two(*n)
            } else {
                false
            }
        },
    )
}
