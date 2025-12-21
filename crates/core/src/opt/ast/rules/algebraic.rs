//! 代数的変形ルール
//!
//! 単位元、零元、冪等則、逆演算、交換則、結合則、分配則などの代数的変形ルール。

use crate::ast::pat::AstRewriteRule;
use crate::ast::{AstNode, Literal};
use crate::astpat;
use std::rc::Rc;

use super::macros::{associative_rules, commutative_rule, identity_rules, zero_rules};

// ============================================================================
// 単位元ルール (Identity Rules)
// ============================================================================

identity_rules!(
    add_zero_right,
    add_zero_left,
    Add,
    Literal::I64(0),
    "加算の右単位元: a + 0 = a",
    "加算の左単位元: 0 + a = a"
);

identity_rules!(
    mul_one_right,
    mul_one_left,
    Mul,
    Literal::I64(1),
    "乗算の右単位元: a * 1 = a",
    "乗算の左単位元: 1 * a = a"
);

// ============================================================================
// 零元ルール (Zero Rules)
// ============================================================================

zero_rules!(
    mul_zero_right,
    mul_zero_left,
    Mul,
    Literal::I64(0),
    "乗算の右零元: a * 0 = 0",
    "乗算の左零元: 0 * a = 0"
);

// ============================================================================
// 冪等則 (Idempotent Rules)
// ============================================================================

/// max(a, a) = a
pub fn max_idempotent() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Max(Box::new(a.clone()), Box::new(a))
    } => {
        a
    })
}

// ============================================================================
// 逆演算ルール (Inverse Operation Rules)
// ============================================================================

/// 逆数の逆数: recip(recip(a)) = a
pub fn recip_recip() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Recip(Box::new(AstNode::Recip(Box::new(a))))
    } => {
        a
    })
}

/// 平方根の二乗: sqrt(a) * sqrt(a) = a (ただし条件付き)
/// 注: このルールは a >= 0 の場合のみ有効なので、条件付きルールとして実装
pub fn sqrt_squared() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Sqrt(Box::new(a.clone())) * AstNode::Sqrt(Box::new(a))
    } => {
        a
    })
}

/// log2とexp2の逆関数関係: log2(exp2(a)) = a
pub fn log2_exp2() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Log2(Box::new(AstNode::Exp2(Box::new(a))))
    } => {
        a
    })
}

/// exp2とlog2の逆関数関係: exp2(log2(a)) = a (ただし a > 0)
/// 注: このルールは a > 0 の場合のみ有効
pub fn exp2_log2() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Exp2(Box::new(AstNode::Log2(Box::new(a))))
    } => {
        a
    })
}

// ============================================================================
// 交換則 (Commutative Rules)
// ============================================================================

commutative_rule!(add_commutative, Add, "加算の交換則: a + b = b + a");
commutative_rule!(mul_commutative, Mul, "乗算の交換則: a * b = b * a");
commutative_rule!(max_commutative, Max, "maxの交換則: max(a, b) = max(b, a)");

// ============================================================================
// 結合則 (Associative Rules)
// ============================================================================

associative_rules!(
    add_associate_left_to_right,
    add_associate_right_to_left,
    Add,
    "加算の左結合から右結合: (a + b) + c = a + (b + c)",
    "加算の右結合から左結合: a + (b + c) = (a + b) + c"
);

associative_rules!(
    mul_associate_left_to_right,
    mul_associate_right_to_left,
    Mul,
    "乗算の左結合から右結合: (a * b) * c = a * (b * c)",
    "乗算の右結合から左結合: a * (b * c) = (a * b) * c"
);

// ============================================================================
// 同項規則 (Same Term Rules)
// ============================================================================

/// 同じ項の加算: a + a = a * 2
pub fn add_same_to_mul_two() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        a.clone() + a
    } => {
        a * 2isize
    })
}

// ============================================================================
// Block簡約 (Block Simplification)
// ============================================================================

/// 単一要素のBlockを展開: Block { statements: [x], .. } = x
///
/// 単一の文しか含まないBlockノードは、その文に直接置き換えることができます。
/// これにより、ループ交換や他の最適化で生成された不要なBlockが削除されます。
pub fn unwrap_single_statement_block() -> Rc<AstRewriteRule> {
    AstRewriteRule::new(
        AstNode::Wildcard("block".to_string()),
        |bindings| {
            let block = bindings.get("block").unwrap();

            if let AstNode::Block { statements, .. } = block
                && statements.len() == 1
            {
                return statements[0].clone();
            }

            block.clone()
        },
        |bindings| {
            matches!(
                bindings.get("block"),
                Some(AstNode::Block { statements, .. }) if statements.len() == 1
            )
        },
    )
}

// ============================================================================
// 分配則 (Distributive Rules)
// ============================================================================

/// 左分配則: a * (b + c) = a * b + a * c
pub fn distributive_left() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        a.clone() * (b.clone() + c.clone())
    } => {
        a.clone() * b + a * c
    })
}

/// 右分配則: (a + b) * c = a * c + b * c
pub fn distributive_right() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        (a.clone() + b.clone()) * c.clone()
    } => {
        a * c.clone() + b * c
    })
}

/// 因数分解（左）: a * b + a * c = a * (b + c)
pub fn factor_left() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        a.clone() * b.clone() + a.clone() * c.clone()
    } => {
        a * (b + c)
    })
}

/// 因数分解（右）: a * c + b * c = (a + b) * c
pub fn factor_right() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        a.clone() * c.clone() + b.clone() * c.clone()
    } => {
        (a + b) * c
    })
}

// ============================================================================
// FMA (Fused Multiply-Add) Rules
// ============================================================================

/// 乗算と加算の融合（右加算）: a * b + c => fma(a, b, c)
/// FMAは単一の丸め操作で実行され、中間結果の精度が保持される
/// 注意: 浮動小数点型にのみ適用（整数型のFMAはGPUでサポートされないため）
pub fn mul_add_to_fma() -> Rc<AstRewriteRule> {
    use crate::ast::DType;
    use crate::ast::helper::fma;
    use crate::ast::pat::AstRewriteRule;

    // パターン: a * b + c
    let a = AstNode::Wildcard("a".to_string());
    let b = AstNode::Wildcard("b".to_string());
    let c = AstNode::Wildcard("c".to_string());
    let pattern = AstNode::Add(
        Box::new(AstNode::Mul(Box::new(a), Box::new(b))),
        Box::new(c),
    );

    AstRewriteRule::new(
        pattern,
        |bindings| {
            let a = bindings.get("a").unwrap().clone();
            let b = bindings.get("b").unwrap().clone();
            let c = bindings.get("c").unwrap().clone();
            fma(a, b, c)
        },
        |bindings| {
            // 浮動小数点型の場合のみFMA化
            let a = bindings.get("a").unwrap();
            matches!(a.infer_type(), DType::F32)
        },
    )
}

/// 乗算と加算の融合（左加算）: c + a * b => fma(a, b, c)
/// 加算の交換則を適用したパターン
/// 注意: 浮動小数点型にのみ適用（整数型のFMAはGPUでサポートされないため）
pub fn add_mul_to_fma() -> Rc<AstRewriteRule> {
    use crate::ast::DType;
    use crate::ast::helper::fma;
    use crate::ast::pat::AstRewriteRule;

    // パターン: c + a * b
    let a = AstNode::Wildcard("a".to_string());
    let b = AstNode::Wildcard("b".to_string());
    let c = AstNode::Wildcard("c".to_string());
    let pattern = AstNode::Add(
        Box::new(c),
        Box::new(AstNode::Mul(Box::new(a), Box::new(b))),
    );

    AstRewriteRule::new(
        pattern,
        |bindings| {
            let a = bindings.get("a").unwrap().clone();
            let b = bindings.get("b").unwrap().clone();
            let c = bindings.get("c").unwrap().clone();
            fma(a, b, c)
        },
        |bindings| {
            // 浮動小数点型の場合のみFMA化
            let a = bindings.get("a").unwrap();
            matches!(a.infer_type(), DType::F32)
        },
    )
}
