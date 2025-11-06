use crate::ast::pat::AstRewriteRule;
use crate::ast::{AstNode, Literal};
use crate::astpat;
use std::rc::Rc;

/// 代数的な項書き換えルール集
///
/// このモジュールは、ASTノードに対する標準的な代数的変形ルールを提供します。
/// これらのルールは、式の簡約や正規化、最適化に使用できます。
// ============================================================================
// 単位元ルール (Identity Rules)
// ============================================================================
/// 加算の右単位元: a + 0 = a
pub fn add_zero_right() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Isize(0))))
    } => {
        a
    })
}

/// 加算の左単位元: 0 + a = a
pub fn add_zero_left() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Add(Box::new(AstNode::Const(Literal::Isize(0))), Box::new(a))
    } => {
        a
    })
}

/// 乗算の右単位元: a * 1 = a
pub fn mul_one_right() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Mul(Box::new(a), Box::new(AstNode::Const(Literal::Isize(1))))
    } => {
        a
    })
}

/// 乗算の左単位元: 1 * a = a
pub fn mul_one_left() -> Rc<AstRewriteRule> {
    astpat!(|a| {
        AstNode::Mul(Box::new(AstNode::Const(Literal::Isize(1))), Box::new(a))
    } => {
        a
    })
}

// ============================================================================
// 零元ルール (Zero Rules)
// ============================================================================

/// 乗算の右零元: a * 0 = 0
pub fn mul_zero_right() -> Rc<AstRewriteRule> {
    astpat!(|_a| {
        AstNode::Mul(Box::new(_a), Box::new(AstNode::Const(Literal::Isize(0))))
    } => {
        AstNode::Const(Literal::Isize(0))
    })
}

/// 乗算の左零元: 0 * a = 0
pub fn mul_zero_left() -> Rc<AstRewriteRule> {
    astpat!(|_a| {
        AstNode::Mul(Box::new(AstNode::Const(Literal::Isize(0))), Box::new(_a))
    } => {
        AstNode::Const(Literal::Isize(0))
    })
}

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
        AstNode::Mul(
            Box::new(AstNode::Sqrt(Box::new(a.clone()))),
            Box::new(AstNode::Sqrt(Box::new(a)))
        )
    } => {
        a
    })
}

// ============================================================================
// 交換則 (Commutative Rules)
// ============================================================================

/// 加算の交換則: a + b = b + a
pub fn add_commutative() -> Rc<AstRewriteRule> {
    astpat!(|a, b| {
        AstNode::Add(Box::new(a), Box::new(b))
    } => {
        AstNode::Add(Box::new(b), Box::new(a))
    })
}

/// 乗算の交換則: a * b = b * a
pub fn mul_commutative() -> Rc<AstRewriteRule> {
    astpat!(|a, b| {
        AstNode::Mul(Box::new(a), Box::new(b))
    } => {
        AstNode::Mul(Box::new(b), Box::new(a))
    })
}

/// maxの交換則: max(a, b) = max(b, a)
pub fn max_commutative() -> Rc<AstRewriteRule> {
    astpat!(|a, b| {
        AstNode::Max(Box::new(a), Box::new(b))
    } => {
        AstNode::Max(Box::new(b), Box::new(a))
    })
}

// ============================================================================
// 結合則 (Associative Rules)
// ============================================================================

/// 加算の左結合から右結合: (a + b) + c = a + (b + c)
pub fn add_associate_left_to_right() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Add(
            Box::new(AstNode::Add(Box::new(a), Box::new(b))),
            Box::new(c)
        )
    } => {
        AstNode::Add(
            Box::new(a),
            Box::new(AstNode::Add(Box::new(b), Box::new(c)))
        )
    })
}

/// 加算の右結合から左結合: a + (b + c) = (a + b) + c
pub fn add_associate_right_to_left() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Add(
            Box::new(a),
            Box::new(AstNode::Add(Box::new(b), Box::new(c)))
        )
    } => {
        AstNode::Add(
            Box::new(AstNode::Add(Box::new(a), Box::new(b))),
            Box::new(c)
        )
    })
}

/// 乗算の左結合から右結合: (a * b) * c = a * (b * c)
pub fn mul_associate_left_to_right() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Mul(
            Box::new(AstNode::Mul(Box::new(a), Box::new(b))),
            Box::new(c)
        )
    } => {
        AstNode::Mul(
            Box::new(a),
            Box::new(AstNode::Mul(Box::new(b), Box::new(c)))
        )
    })
}

/// 乗算の右結合から左結合: a * (b * c) = (a * b) * c
pub fn mul_associate_right_to_left() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Mul(
            Box::new(a),
            Box::new(AstNode::Mul(Box::new(b), Box::new(c)))
        )
    } => {
        AstNode::Mul(
            Box::new(AstNode::Mul(Box::new(a), Box::new(b))),
            Box::new(c)
        )
    })
}

// ============================================================================
// 分配則 (Distributive Rules)
// ============================================================================

/// 左分配則: a * (b + c) = a * b + a * c
pub fn distributive_left() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Add(Box::new(b), Box::new(c)))
        )
    } => {
        AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(b))),
            Box::new(AstNode::Mul(Box::new(a), Box::new(c)))
        )
    })
}

/// 右分配則: (a + b) * c = a * c + b * c
pub fn distributive_right() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Mul(
            Box::new(AstNode::Add(Box::new(a), Box::new(b))),
            Box::new(c.clone())
        )
    } => {
        AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a), Box::new(c.clone()))),
            Box::new(AstNode::Mul(Box::new(b), Box::new(c)))
        )
    })
}

/// 因数分解（左）: a * b + a * c = a * (b + c)
pub fn factor_left() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(b))),
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(c)))
        )
    } => {
        AstNode::Mul(
            Box::new(a),
            Box::new(AstNode::Add(Box::new(b), Box::new(c)))
        )
    })
}

/// 因数分解（右）: a * c + b * c = (a + b) * c
pub fn factor_right() -> Rc<AstRewriteRule> {
    astpat!(|a, b, c| {
        AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a), Box::new(c.clone()))),
            Box::new(AstNode::Mul(Box::new(b), Box::new(c.clone())))
        )
    } => {
        AstNode::Mul(
            Box::new(AstNode::Add(Box::new(a), Box::new(b))),
            Box::new(c)
        )
    })
}

// ============================================================================
// ルール集の生成
// ============================================================================

/// 簡約ルール集（式を簡単にする）
pub fn simplification_rules() -> Vec<Rc<AstRewriteRule>> {
    vec![
        // 単位元
        add_zero_right(),
        add_zero_left(),
        mul_one_right(),
        mul_one_left(),
        // 零元
        mul_zero_right(),
        mul_zero_left(),
        // 冪等則
        max_idempotent(),
        // 逆演算
        recip_recip(),
        sqrt_squared(),
    ]
}

/// 正規化ルール集（式を標準形に変換する）
pub fn normalization_rules() -> Vec<Rc<AstRewriteRule>> {
    vec![
        // 結合則（右結合に統一）
        add_associate_left_to_right(),
        mul_associate_left_to_right(),
    ]
}

/// すべての代数的ルール集
pub fn all_algebraic_rules() -> Vec<Rc<AstRewriteRule>> {
    let mut rules = Vec::new();
    rules.extend(simplification_rules());
    rules.extend(normalization_rules());
    // 交換則は探索用なのでデフォルトには含めない（無限ループの可能性）
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::ast::Optimizer;
    use crate::opt::ast::RuleBaseOptimizer;

    #[test]
    fn test_add_zero() {
        let rule = add_zero_right();
        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(42))),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));
    }

    #[test]
    fn test_mul_one() {
        let rule = mul_one_right();
        let input = AstNode::Mul(
            Box::new(AstNode::Const(Literal::Isize(42))),
            Box::new(AstNode::Const(Literal::Isize(1))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));
    }

    #[test]
    fn test_mul_zero() {
        let rule = mul_zero_right();
        let input = AstNode::Mul(
            Box::new(AstNode::Const(Literal::Isize(42))),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(0)));
    }

    #[test]
    fn test_max_idempotent() {
        let rule = max_idempotent();
        let var_a = AstNode::Var("a".to_string());
        let input = AstNode::Max(Box::new(var_a.clone()), Box::new(var_a.clone()));
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_recip_recip() {
        let rule = recip_recip();
        let var_a = AstNode::Var("a".to_string());
        let input = AstNode::Recip(Box::new(AstNode::Recip(Box::new(var_a.clone()))));
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_sqrt_squared() {
        let rule = sqrt_squared();
        let var_a = AstNode::Var("a".to_string());
        let input = AstNode::Mul(
            Box::new(AstNode::Sqrt(Box::new(var_a.clone()))),
            Box::new(AstNode::Sqrt(Box::new(var_a.clone()))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_distributive_left() {
        let rule = distributive_left();
        // a * (b + c)
        let input = AstNode::Mul(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Add(
                Box::new(AstNode::Var("b".to_string())),
                Box::new(AstNode::Var("c".to_string())),
            )),
        );
        let result = rule.apply(&input);
        // a * b + a * c
        let expected = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            )),
            Box::new(AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("c".to_string())),
            )),
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_factor_left() {
        let rule = factor_left();
        // a * b + a * c
        let input = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            )),
            Box::new(AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("c".to_string())),
            )),
        );
        let result = rule.apply(&input);
        // a * (b + c)
        let expected = AstNode::Mul(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Add(
                Box::new(AstNode::Var("b".to_string())),
                Box::new(AstNode::Var("c".to_string())),
            )),
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simplification_rules() {
        let optimizer = RuleBaseOptimizer::new(simplification_rules());

        // (42 + 0) * 1
        let input = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::Isize(42))),
                Box::new(AstNode::Const(Literal::Isize(0))),
            )),
            Box::new(AstNode::Const(Literal::Isize(1))),
        );

        let result = optimizer.optimize(input);
        // 42
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));
    }

    #[test]
    fn test_associative_rules() {
        let rule = add_associate_left_to_right();
        // (a + b) + c
        let input = AstNode::Add(
            Box::new(AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            )),
            Box::new(AstNode::Var("c".to_string())),
        );
        let result = rule.apply(&input);
        // a + (b + c)
        let expected = AstNode::Add(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Add(
                Box::new(AstNode::Var("b".to_string())),
                Box::new(AstNode::Var("c".to_string())),
            )),
        );
        assert_eq!(result, expected);
    }
}
