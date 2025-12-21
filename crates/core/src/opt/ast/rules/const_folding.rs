//! 定数畳み込みルール
//!
//! コンパイル時に定数式を評価するルール群。

use crate::ast::pat::AstRewriteRule;
use crate::ast::{AstNode, Literal};
use std::rc::Rc;

// ============================================================================
// 定数畳み込みマクロ
// ============================================================================

/// IsizeとF32両方をサポートする二項演算の定数畳み込み
macro_rules! binary_const_fold_both {
    ($name:ident, $node_variant:ident, $op:tt, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            AstRewriteRule::new(
                AstNode::$node_variant(
                    Box::new(AstNode::Wildcard("a".to_string())),
                    Box::new(AstNode::Wildcard("b".to_string())),
                ),
                |bindings| {
                    let a = bindings.get("a").unwrap();
                    let b = bindings.get("b").unwrap();

                    match (a, b) {
                        (AstNode::Const(Literal::I64(av)), AstNode::Const(Literal::I64(bv))) => {
                            AstNode::Const(Literal::I64(av $op bv))
                        }
                        (AstNode::Const(Literal::F32(av)), AstNode::Const(Literal::F32(bv))) => {
                            AstNode::Const(Literal::F32(av $op bv))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone()), Box::new(b.clone())),
                    }
                },
                |bindings| {
                    matches!(
                        (bindings.get("a"), bindings.get("b")),
                        (Some(AstNode::Const(_)), Some(AstNode::Const(_)))
                    )
                },
            )
        }
    };
}

/// maxなどメソッド呼び出しが必要な二項演算
macro_rules! binary_const_fold_method {
    ($name:ident, $node_variant:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            AstRewriteRule::new(
                AstNode::$node_variant(
                    Box::new(AstNode::Wildcard("a".to_string())),
                    Box::new(AstNode::Wildcard("b".to_string())),
                ),
                |bindings| {
                    let a = bindings.get("a").unwrap();
                    let b = bindings.get("b").unwrap();

                    match (a, b) {
                        (AstNode::Const(Literal::I64(av)), AstNode::Const(Literal::I64(bv))) => {
                            AstNode::Const(Literal::I64(*av.$method(bv)))
                        }
                        (AstNode::Const(Literal::F32(av)), AstNode::Const(Literal::F32(bv))) => {
                            AstNode::Const(Literal::F32(av.$method(*bv)))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone()), Box::new(b.clone())),
                    }
                },
                |bindings| {
                    matches!(
                        (bindings.get("a"), bindings.get("b")),
                        (Some(AstNode::Const(_)), Some(AstNode::Const(_)))
                    )
                },
            )
        }
    };
}

/// Isizeのみ、条件付き二項演算（ゼロ除算チェック）
macro_rules! binary_const_fold_isize_checked {
    ($name:ident, $node_variant:ident, $op:tt, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            AstRewriteRule::new(
                AstNode::$node_variant(
                    Box::new(AstNode::Wildcard("a".to_string())),
                    Box::new(AstNode::Wildcard("b".to_string())),
                ),
                |bindings| {
                    let a = bindings.get("a").unwrap();
                    let b = bindings.get("b").unwrap();

                    match (a, b) {
                        (AstNode::Const(Literal::I64(av)), AstNode::Const(Literal::I64(bv)))
                            if *bv != 0 =>
                        {
                            AstNode::Const(Literal::I64(av $op bv))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone()), Box::new(b.clone())),
                    }
                },
                |bindings| {
                    matches!(
                        (bindings.get("a"), bindings.get("b")),
                        (Some(AstNode::Const(Literal::I64(_))), Some(AstNode::Const(Literal::I64(b)))) if *b != 0
                    )
                },
            )
        }
    };
}

/// F32のみ、条件付き単項演算
macro_rules! unary_const_fold_f32_checked {
    ($name:ident, $node_variant:ident, $method:ident, $condition:expr, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            AstRewriteRule::new(
                AstNode::$node_variant(Box::new(AstNode::Wildcard("a".to_string()))),
                |bindings| {
                    let a = bindings.get("a").unwrap();

                    match a {
                        AstNode::Const(Literal::F32(av)) if $condition(*av) => {
                            AstNode::Const(Literal::F32(av.$method()))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone())),
                    }
                },
                |bindings| {
                    matches!(
                        bindings.get("a"),
                        Some(AstNode::Const(Literal::F32(v))) if $condition(*v)
                    )
                },
            )
        }
    };
}

/// F32のみ、条件付き単項演算（カスタム計算式）
macro_rules! unary_const_fold_f32_checked_custom {
    ($name:ident, $node_variant:ident, $expr:expr, $condition:expr, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            AstRewriteRule::new(
                AstNode::$node_variant(Box::new(AstNode::Wildcard("a".to_string()))),
                |bindings| {
                    let a = bindings.get("a").unwrap();

                    match a {
                        AstNode::Const(Literal::F32(av)) if $condition(*av) => {
                            AstNode::Const(Literal::F32($expr(*av)))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone())),
                    }
                },
                |bindings| {
                    matches!(
                        bindings.get("a"),
                        Some(AstNode::Const(Literal::F32(v))) if $condition(*v)
                    )
                },
            )
        }
    };
}

/// F32のみ、無条件単項演算
macro_rules! unary_const_fold_f32 {
    ($name:ident, $node_variant:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            AstRewriteRule::new(
                AstNode::$node_variant(Box::new(AstNode::Wildcard("a".to_string()))),
                |bindings| {
                    let a = bindings.get("a").unwrap();

                    match a {
                        AstNode::Const(Literal::F32(av)) => {
                            AstNode::Const(Literal::F32(av.$method()))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone())),
                    }
                },
                |bindings| matches!(bindings.get("a"), Some(AstNode::Const(Literal::F32(_)))),
            )
        }
    };
}

// ============================================================================
// 定数畳み込みルール定義
// ============================================================================

binary_const_fold_both!(
    const_fold_add,
    Add,
    +,
    "定数加算の畳み込み: Const(a) + Const(b) = Const(a + b)"
);
binary_const_fold_both!(
    const_fold_mul,
    Mul,
    *,
    "定数乗算の畳み込み: Const(a) * Const(b) = Const(a * b)"
);
binary_const_fold_method!(
    const_fold_max,
    Max,
    max,
    "定数maxの畳み込み: max(Const(a), Const(b)) = Const(max(a, b))"
);
binary_const_fold_isize_checked!(
    const_fold_rem,
    Rem,
    %,
    "定数剰余の畳み込み: Const(a) % Const(b) = Const(a % b)"
);
binary_const_fold_isize_checked!(
    const_fold_idiv,
    Idiv,
    /,
    "定数除算の畳み込み: Const(a) / Const(b) = Const(a / b)"
);
unary_const_fold_f32_checked_custom!(
    const_fold_recip,
    Recip,
    |v| 1.0 / v,
    |v| v != 0.0,
    "定数逆数の畳み込み: recip(Const(a)) = Const(1 / a)"
);
unary_const_fold_f32_checked!(
    const_fold_sqrt,
    Sqrt,
    sqrt,
    |v| v >= 0.0,
    "定数平方根の畳み込み: sqrt(Const(a)) = Const(sqrt(a))"
);
unary_const_fold_f32_checked!(
    const_fold_log2,
    Log2,
    log2,
    |v| v > 0.0,
    "定数log2の畳み込み: log2(Const(a)) = Const(log2(a))"
);
unary_const_fold_f32!(
    const_fold_exp2,
    Exp2,
    exp2,
    "定数exp2の畳み込み: exp2(Const(a)) = Const(2^a)"
);
unary_const_fold_f32!(
    const_fold_sin,
    Sin,
    sin,
    "定数sinの畳み込み: sin(Const(a)) = Const(sin(a))"
);

/// 定数畳み込みルール集
pub fn constant_folding_rules() -> Vec<Rc<AstRewriteRule>> {
    vec![
        const_fold_add(),
        const_fold_mul(),
        const_fold_max(),
        const_fold_rem(),
        const_fold_idiv(),
        const_fold_recip(),
        const_fold_sqrt(),
        const_fold_log2(),
        const_fold_exp2(),
        const_fold_sin(),
    ]
}
