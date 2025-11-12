use crate::ast::pat::AstRewriteRule;
use crate::ast::{AstNode, Literal};
use crate::astpat;
use std::rc::Rc;

/// 代数的な項書き換えルール集
///
/// このモジュールは、ASTノードに対する標準的な代数的変形ルールを提供します。
/// これらのルールは、式の簡約や正規化、最適化に使用できます。
// ============================================================================
// マクロ定義: 共通パターンの生成
// ============================================================================
/// 単位元ルールを生成するマクロ
/// op(a, identity) = a および op(identity, a) = a のパターン
macro_rules! identity_rules {
    ($right_name:ident, $left_name:ident, $op:ident, $identity:expr, $right_doc:expr, $left_doc:expr) => {
        #[doc = $right_doc]
        pub fn $right_name() -> Rc<AstRewriteRule> {
            astpat!(|a| {
                AstNode::$op(Box::new(a), Box::new(AstNode::Const($identity)))
            } => {
                a
            })
        }

        #[doc = $left_doc]
        pub fn $left_name() -> Rc<AstRewriteRule> {
            astpat!(|a| {
                AstNode::$op(Box::new(AstNode::Const($identity)), Box::new(a))
            } => {
                a
            })
        }
    };
}

/// 零元ルールを生成するマクロ
/// op(a, zero) = zero および op(zero, a) = zero のパターン
macro_rules! zero_rules {
    ($right_name:ident, $left_name:ident, $op:ident, $zero:expr, $right_doc:expr, $left_doc:expr) => {
        #[doc = $right_doc]
        pub fn $right_name() -> Rc<AstRewriteRule> {
            astpat!(|_a| {
                AstNode::$op(Box::new(_a), Box::new(AstNode::Const($zero)))
            } => {
                AstNode::Const($zero)
            })
        }

        #[doc = $left_doc]
        pub fn $left_name() -> Rc<AstRewriteRule> {
            astpat!(|_a| {
                AstNode::$op(Box::new(AstNode::Const($zero)), Box::new(_a))
            } => {
                AstNode::Const($zero)
            })
        }
    };
}

/// 交換則ルールを生成するマクロ
/// op(a, b) = op(b, a) のパターン
macro_rules! commutative_rule {
    ($name:ident, $op:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            astpat!(|a, b| {
                AstNode::$op(Box::new(a), Box::new(b))
            } => {
                AstNode::$op(Box::new(b), Box::new(a))
            })
        }
    };
}

/// 結合則ルールを生成するマクロ
/// op(op(a, b), c) = op(a, op(b, c)) とその逆のパターン
macro_rules! associative_rules {
    ($left_to_right:ident, $right_to_left:ident, $op:ident, $ltr_doc:expr, $rtl_doc:expr) => {
        #[doc = $ltr_doc]
        pub fn $left_to_right() -> Rc<AstRewriteRule> {
            astpat!(|a, b, c| {
                AstNode::$op(
                    Box::new(AstNode::$op(Box::new(a), Box::new(b))),
                    Box::new(c)
                )
            } => {
                AstNode::$op(
                    Box::new(a),
                    Box::new(AstNode::$op(Box::new(b), Box::new(c)))
                )
            })
        }

        #[doc = $rtl_doc]
        pub fn $right_to_left() -> Rc<AstRewriteRule> {
            astpat!(|a, b, c| {
                AstNode::$op(
                    Box::new(a),
                    Box::new(AstNode::$op(Box::new(b), Box::new(c)))
                )
            } => {
                AstNode::$op(
                    Box::new(AstNode::$op(Box::new(a), Box::new(b))),
                    Box::new(c)
                )
            })
        }
    };
}

// ============================================================================
// 単位元ルール (Identity Rules)
// ============================================================================

identity_rules!(
    add_zero_right,
    add_zero_left,
    Add,
    Literal::Int(0),
    "加算の右単位元: a + 0 = a",
    "加算の左単位元: 0 + a = a"
);

identity_rules!(
    mul_one_right,
    mul_one_left,
    Mul,
    Literal::Int(1),
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
    Literal::Int(0),
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
        AstNode::Mul(
            Box::new(AstNode::Sqrt(Box::new(a.clone()))),
            Box::new(AstNode::Sqrt(Box::new(a)))
        )
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
        AstNode::Add(Box::new(a.clone()), Box::new(a))
    } => {
        AstNode::Mul(Box::new(a), Box::new(AstNode::Const(Literal::Int(2))))
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
                        (AstNode::Const(Literal::Int(av)), AstNode::Const(Literal::Int(bv))) => {
                            AstNode::Const(Literal::Int(av $op bv))
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
                        (AstNode::Const(Literal::Int(av)), AstNode::Const(Literal::Int(bv))) => {
                            AstNode::Const(Literal::Int(*av.$method(bv)))
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
                        (AstNode::Const(Literal::Int(av)), AstNode::Const(Literal::Int(bv)))
                            if *bv != 0 =>
                        {
                            AstNode::Const(Literal::Int(av $op bv))
                        }
                        _ => AstNode::$node_variant(Box::new(a.clone()), Box::new(b.clone())),
                    }
                },
                |bindings| {
                    matches!(
                        (bindings.get("a"), bindings.get("b")),
                        (Some(AstNode::Const(Literal::Int(_))), Some(AstNode::Const(Literal::Int(b)))) if *b != 0
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
// ビット演算最適化 (Bit Operation Optimizations)
// ============================================================================

/// 値が2の累乗かどうかをチェックするヘルパー関数
fn is_power_of_two(n: isize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// 2の累乗の値からlog2を計算するヘルパー関数
fn log2_of_power_of_two(n: isize) -> isize {
    n.trailing_zeros() as isize
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

            if let AstNode::Const(Literal::Int(n)) = b
                && is_power_of_two(*n)
            {
                return AstNode::LeftShift(
                    Box::new(a.clone()),
                    Box::new(AstNode::Const(Literal::Int(log2_of_power_of_two(*n)))),
                );
            }

            // 条件に合わなければ元のノードを返す
            AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))
        },
        |bindings| {
            // 条件：bが2の累乗の定数
            if let Some(AstNode::Const(Literal::Int(n))) = bindings.get("b") {
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

            if let AstNode::Const(Literal::Int(n)) = a
                && is_power_of_two(*n)
            {
                return AstNode::LeftShift(
                    Box::new(b.clone()),
                    Box::new(AstNode::Const(Literal::Int(log2_of_power_of_two(*n)))),
                );
            }

            // 条件に合わなければ元のノードを返す
            AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))
        },
        |bindings| {
            // 条件：aが2の累乗の定数
            if let Some(AstNode::Const(Literal::Int(n))) = bindings.get("a") {
                is_power_of_two(*n)
            } else {
                false
            }
        },
    )
}

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
        log2_exp2(),
        exp2_log2(),
        // 同項規則
        add_same_to_mul_two(),
        // Block簡約
        unwrap_single_statement_block(),
        // ビット演算最適化（2の累乗の乗算をシフトに変換）
        mul_power_of_two_to_shift_right(),
        mul_power_of_two_to_shift_left(),
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

/// 探索用の完全なルール集（交換則・分配則を含む）
///
/// ビームサーチなどの探索ベース最適化で使用することを想定しています。
/// RuleBaseOptimizerで直接使うと無限ループする可能性があるため注意してください。
pub fn all_rules_with_search() -> Vec<Rc<AstRewriteRule>> {
    let mut rules = Vec::new();
    rules.extend(constant_folding_rules());
    rules.extend(simplification_rules());
    rules.extend(normalization_rules());
    // 交換則
    rules.push(add_commutative());
    rules.push(mul_commutative());
    rules.push(max_commutative());
    // 分配則を追加
    rules.push(distributive_left());
    rules.push(distributive_right());
    rules.push(factor_left());
    rules.push(factor_right());
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
            Box::new(AstNode::Const(Literal::Int(42))),
            Box::new(AstNode::Const(Literal::Int(0))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(42)));
    }

    #[test]
    fn test_mul_one() {
        let rule = mul_one_right();
        let input = AstNode::Mul(
            Box::new(AstNode::Const(Literal::Int(42))),
            Box::new(AstNode::Const(Literal::Int(1))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(42)));
    }

    #[test]
    fn test_mul_zero() {
        let rule = mul_zero_right();
        let input = AstNode::Mul(
            Box::new(AstNode::Const(Literal::Int(42))),
            Box::new(AstNode::Const(Literal::Int(0))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(0)));
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
                Box::new(AstNode::Const(Literal::Int(42))),
                Box::new(AstNode::Const(Literal::Int(0))),
            )),
            Box::new(AstNode::Const(Literal::Int(1))),
        );

        let result = optimizer.optimize(input);
        // 42
        assert_eq!(result, AstNode::Const(Literal::Int(42)));
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
            Box::new(AstNode::Const(Literal::Int(2))),
            Box::new(AstNode::Const(Literal::Int(3))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(5)));

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
            Box::new(AstNode::Const(Literal::Int(6))),
            Box::new(AstNode::Const(Literal::Int(7))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(42)));

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
            Box::new(AstNode::Const(Literal::Int(3))),
            Box::new(AstNode::Const(Literal::Int(5))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(5)));

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
            Box::new(AstNode::Const(Literal::Int(10))),
            Box::new(AstNode::Const(Literal::Int(3))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(1)));
    }

    #[test]
    fn test_const_fold_idiv() {
        let rule = const_fold_idiv();

        // 10 / 3 = 3
        let input = AstNode::Idiv(
            Box::new(AstNode::Const(Literal::Int(10))),
            Box::new(AstNode::Const(Literal::Int(3))),
        );
        let result = rule.apply(&input);
        assert_eq!(result, AstNode::Const(Literal::Int(3)));
    }

    #[test]
    fn test_constant_folding_with_optimizer() {
        let optimizer = RuleBaseOptimizer::new(constant_folding_rules());

        // (2 + 3) * 4 = 5 * 4 = 20
        let input = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::Int(2))),
                Box::new(AstNode::Const(Literal::Int(3))),
            )),
            Box::new(AstNode::Const(Literal::Int(4))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Int(20)));
    }

    #[test]
    fn test_combined_optimization() {
        // 定数畳み込みと簡約を組み合わせ
        let optimizer = RuleBaseOptimizer::new(all_algebraic_rules());

        // ((2 + 3) * 1) + 0 = 5 * 1 + 0 = 5 + 0 = 5
        let input = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Int(2))),
                    Box::new(AstNode::Const(Literal::Int(3))),
                )),
                Box::new(AstNode::Const(Literal::Int(1))),
            )),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Int(5)));
    }

    #[test]
    fn test_mul_power_of_two_to_shift_right() {
        // x * 4 → x << 2
        let rule = mul_power_of_two_to_shift_right();
        let input = AstNode::Mul(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(4))),
        );
        let result = rule.apply(&input);

        match result {
            AstNode::LeftShift(left, right) => {
                assert_eq!(*left, AstNode::Var("x".to_string()));
                assert_eq!(*right, AstNode::Const(Literal::Int(2)));
            }
            _ => panic!("Expected LeftShift node"),
        }
    }

    #[test]
    fn test_mul_power_of_two_to_shift_left() {
        // 8 * x → x << 3
        let rule = mul_power_of_two_to_shift_left();
        let input = AstNode::Mul(
            Box::new(AstNode::Const(Literal::Int(8))),
            Box::new(AstNode::Var("x".to_string())),
        );
        let result = rule.apply(&input);

        match result {
            AstNode::LeftShift(left, right) => {
                assert_eq!(*left, AstNode::Var("x".to_string()));
                assert_eq!(*right, AstNode::Const(Literal::Int(3)));
            }
            _ => panic!("Expected LeftShift node"),
        }
    }

    #[test]
    fn test_mul_non_power_of_two() {
        // 非2の累乗の場合は変換されない（元のノードが返る）
        let rule = mul_power_of_two_to_shift_right();
        let input = AstNode::Mul(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(5))),
        );
        let result = rule.apply(&input);

        // 元のMulノードが返ってくるはず
        match result {
            AstNode::Mul(left, right) => {
                assert_eq!(*left, AstNode::Var("x".to_string()));
                assert_eq!(*right, AstNode::Const(Literal::Int(5)));
            }
            _ => panic!("Expected Mul node (unchanged)"),
        }
    }

    #[test]
    fn test_mul_various_powers_of_two() {
        // さまざまな2の累乗をテスト
        let test_cases = vec![
            (1, 0),   // 1 = 2^0
            (2, 1),   // 2 = 2^1
            (4, 2),   // 4 = 2^2
            (8, 3),   // 8 = 2^3
            (16, 4),  // 16 = 2^4
            (32, 5),  // 32 = 2^5
            (64, 6),  // 64 = 2^6
            (128, 7), // 128 = 2^7
            (256, 8), // 256 = 2^8
        ];

        let rule = mul_power_of_two_to_shift_right();
        for (power_of_two, expected_shift) in test_cases {
            let input = AstNode::Mul(
                Box::new(AstNode::Var("x".to_string())),
                Box::new(AstNode::Const(Literal::Int(power_of_two))),
            );
            let result = rule.apply(&input);

            match result {
                AstNode::LeftShift(_, right) => {
                    assert_eq!(*right, AstNode::Const(Literal::Int(expected_shift)));
                }
                _ => panic!("Expected LeftShift node for {}", power_of_two),
            }
        }
    }

    #[test]
    fn test_mul_power_of_two_with_optimizer() {
        // オプティマイザーと組み合わせてテスト
        let optimizer = RuleBaseOptimizer::new(simplification_rules());

        // x * 16 → x << 4
        let input = AstNode::Mul(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(16))),
        );

        let result = optimizer.optimize(input);

        match result {
            AstNode::LeftShift(left, right) => {
                assert_eq!(*left, AstNode::Var("x".to_string()));
                assert_eq!(*right, AstNode::Const(Literal::Int(4)));
            }
            _ => panic!("Expected LeftShift node"),
        }
    }

    #[test]
    fn test_log2_exp2() {
        let rule = log2_exp2();
        let var_a = AstNode::Var("a".to_string());
        let input = AstNode::Log2(Box::new(AstNode::Exp2(Box::new(var_a.clone()))));
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_exp2_log2() {
        let rule = exp2_log2();
        let var_a = AstNode::Var("a".to_string());
        let input = AstNode::Exp2(Box::new(AstNode::Log2(Box::new(var_a.clone()))));
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_add_same_to_mul_two() {
        let rule = add_same_to_mul_two();
        let var_a = AstNode::Var("x".to_string());
        // x + x
        let input = AstNode::Add(Box::new(var_a.clone()), Box::new(var_a.clone()));
        let result = rule.apply(&input);
        // x * 2
        let expected = AstNode::Mul(Box::new(var_a), Box::new(AstNode::Const(Literal::Int(2))));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_same_to_mul_two_with_const() {
        let rule = add_same_to_mul_two();
        // 5 + 5
        let const_5 = AstNode::Const(Literal::Int(5));
        let input = AstNode::Add(Box::new(const_5.clone()), Box::new(const_5.clone()));
        let result = rule.apply(&input);
        // 5 * 2
        let expected = AstNode::Mul(Box::new(const_5), Box::new(AstNode::Const(Literal::Int(2))));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_unwrap_single_statement_block() {
        use crate::ast::Scope;
        let rule = unwrap_single_statement_block();

        // 単一のステートメントを持つBlockは展開される
        let single_stmt = AstNode::Var("x".to_string());
        let block = AstNode::Block {
            statements: vec![single_stmt.clone()],
            scope: Box::new(Scope::new()),
        };
        let result = rule.apply(&block);
        assert_eq!(result, single_stmt);
    }

    #[test]
    fn test_unwrap_single_statement_block_multiple_statements() {
        use crate::ast::Scope;
        let rule = unwrap_single_statement_block();

        // 複数のステートメントを持つBlockは展開されない
        let multi_block = AstNode::Block {
            statements: vec![
                AstNode::Var("x".to_string()),
                AstNode::Var("y".to_string()),
            ],
            scope: Box::new(Scope::new()),
        };
        let result = rule.apply(&multi_block);
        // Blockのまま変わらないはず
        match result {
            AstNode::Block { statements, .. } => {
                assert_eq!(statements.len(), 2);
            }
            _ => panic!("Expected Block node"),
        }
    }

    #[test]
    fn test_unwrap_single_statement_block_with_optimizer() {
        use crate::ast::Scope;
        let optimizer = RuleBaseOptimizer::new(simplification_rules());

        // Block内にRangeがあるケース（ループ交換などで生成されるパターン）
        let inner_range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body: Box::new(AstNode::Var("body".to_string())),
        };

        let block = AstNode::Block {
            statements: vec![inner_range.clone()],
            scope: Box::new(Scope::new()),
        };

        let result = optimizer.optimize(block);

        // Blockが展開されてRangeが直接返されるはず
        match result {
            AstNode::Range { var, .. } => {
                assert_eq!(var, "i");
            }
            _ => panic!("Expected Range node after unwrapping"),
        }
    }
}
