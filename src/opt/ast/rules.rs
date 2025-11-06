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
// 定数畳み込み (Constant Folding)
// ============================================================================

// マクロ: IsizeとF32両方をサポートする二項演算の定数畳み込み
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
                        (AstNode::Const(Literal::Isize(av)), AstNode::Const(Literal::Isize(bv))) => {
                            AstNode::Const(Literal::Isize(av $op bv))
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

// マクロ: maxなどメソッド呼び出しが必要な二項演算
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
                        (
                            AstNode::Const(Literal::Isize(av)),
                            AstNode::Const(Literal::Isize(bv)),
                        ) => AstNode::Const(Literal::Isize(*av.$method(bv))),
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

// マクロ: Isizeのみ、条件付き二項演算（ゼロ除算チェック）
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
                        (AstNode::Const(Literal::Isize(av)), AstNode::Const(Literal::Isize(bv)))
                            if *bv != 0 =>
                        {
                            AstNode::Const(Literal::Isize(av $op bv))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone()), Box::new(b.clone())),
                    }
                },
                |bindings| {
                    matches!(
                        (bindings.get("a"), bindings.get("b")),
                        (Some(AstNode::Const(Literal::Isize(_))), Some(AstNode::Const(Literal::Isize(b)))) if *b != 0
                    )
                },
            )
        }
    };
}

// マクロ: F32のみ、条件付き単項演算
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

// マクロ: F32のみ、条件付き単項演算（カスタム計算式）
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

// マクロ: F32のみ、無条件単項演算
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

// マクロを使って定数畳み込みルールを定義
binary_const_fold_both!(const_fold_add, Add, +, "定数加算の畳み込み: Const(a) + Const(b) = Const(a + b)");
binary_const_fold_both!(const_fold_mul, Mul, *, "定数乗算の畳み込み: Const(a) * Const(b) = Const(a * b)");
binary_const_fold_method!(
    const_fold_max,
    Max,
    max,
    "定数maxの畳み込み: max(Const(a), Const(b)) = Const(max(a, b))"
);
binary_const_fold_isize_checked!(const_fold_rem, Rem, %, "定数剰余の畳み込み: Const(a) % Const(b) = Const(a % b)");
binary_const_fold_isize_checked!(const_fold_idiv, Idiv, /, "定数除算の畳み込み: Const(a) / Const(b) = Const(a / b)");
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

// ============================================================================
// ルール集の生成
// ============================================================================

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

/// すべての代数的ルール集（定数畳み込み含む）
pub fn all_algebraic_rules() -> Vec<Rc<AstRewriteRule>> {
    let mut rules = Vec::new();
    rules.extend(constant_folding_rules());
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

    #[test]
    fn test_const_fold_add() {
        let rule = const_fold_add();

        // Isize: 2 + 3 = 5
        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(2))),
            Box::new(AstNode::Const(Literal::Isize(3))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(5)));

        // F32: 1.5 + 2.5 = 4.0
        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::F32(1.5))),
            Box::new(AstNode::Const(Literal::F32(2.5))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::F32(4.0)));
    }

    #[test]
    fn test_const_fold_mul() {
        let rule = const_fold_mul();

        // Isize: 6 * 7 = 42
        let input = AstNode::Mul(
            Box::new(AstNode::Const(Literal::Isize(6))),
            Box::new(AstNode::Const(Literal::Isize(7))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));

        // F32: 2.5 * 4.0 = 10.0
        let input = AstNode::Mul(
            Box::new(AstNode::Const(Literal::F32(2.5))),
            Box::new(AstNode::Const(Literal::F32(4.0))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::F32(10.0)));
    }

    #[test]
    fn test_const_fold_sqrt() {
        let rule = const_fold_sqrt();

        // sqrt(4.0) = 2.0
        let input = AstNode::Sqrt(Box::new(AstNode::Const(Literal::F32(4.0))));
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::F32(2.0)));

        // sqrt(9.0) = 3.0
        let input = AstNode::Sqrt(Box::new(AstNode::Const(Literal::F32(9.0))));
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::F32(3.0)));
    }

    #[test]
    fn test_const_fold_recip() {
        let rule = const_fold_recip();

        // recip(2.0) = 0.5
        let input = AstNode::Recip(Box::new(AstNode::Const(Literal::F32(2.0))));
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::F32(0.5)));

        // recip(4.0) = 0.25
        let input = AstNode::Recip(Box::new(AstNode::Const(Literal::F32(4.0))));
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::F32(0.25)));
    }

    #[test]
    fn test_const_fold_max() {
        let rule = const_fold_max();

        // max(3, 5) = 5
        let input = AstNode::Max(
            Box::new(AstNode::Const(Literal::Isize(3))),
            Box::new(AstNode::Const(Literal::Isize(5))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(5)));

        // max(2.5, 1.5) = 2.5
        let input = AstNode::Max(
            Box::new(AstNode::Const(Literal::F32(2.5))),
            Box::new(AstNode::Const(Literal::F32(1.5))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::F32(2.5)));
    }

    #[test]
    fn test_const_fold_rem() {
        let rule = const_fold_rem();

        // 10 % 3 = 1
        let input = AstNode::Rem(
            Box::new(AstNode::Const(Literal::Isize(10))),
            Box::new(AstNode::Const(Literal::Isize(3))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(1)));
    }

    #[test]
    fn test_const_fold_idiv() {
        let rule = const_fold_idiv();

        // 10 / 3 = 3
        let input = AstNode::Idiv(
            Box::new(AstNode::Const(Literal::Isize(10))),
            Box::new(AstNode::Const(Literal::Isize(3))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Isize(3)));
    }

    #[test]
    fn test_constant_folding_with_optimizer() {
        let optimizer = RuleBaseOptimizer::new(constant_folding_rules());

        // (2 + 3) * 4 = 5 * 4 = 20
        let input = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::Isize(2))),
                Box::new(AstNode::Const(Literal::Isize(3))),
            )),
            Box::new(AstNode::Const(Literal::Isize(4))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Isize(20)));
    }

    #[test]
    fn test_combined_optimization() {
        // 定数畳み込みと簡約を組み合わせ
        let optimizer = RuleBaseOptimizer::new(all_algebraic_rules());

        // ((2 + 3) * 1) + 0 = 5 * 1 + 0 = 5 + 0 = 5
        let input = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Isize(2))),
                    Box::new(AstNode::Const(Literal::Isize(3))),
                )),
                Box::new(AstNode::Const(Literal::Isize(1))),
            )),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Isize(5)));
    }
}
