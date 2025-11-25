use crate::helper;
use crate::uop::UOp;
use harp::DType;
use std::collections::HashMap;
use std::rc::Rc;

/// 書き換えルールの置換関数の型
type ReplacementFn = Box<dyn Fn(&HashMap<usize, Rc<UOp>>) -> Rc<UOp>>;

/// 書き換えルール
/// pattern にマッチする部分を replacement に置き換える
pub struct RewriteRule {
    pub name: String,
    pub pattern: Rc<UOp>,
    pub replacement: ReplacementFn,
}

impl RewriteRule {
    pub fn new<F>(name: impl Into<String>, pattern: Rc<UOp>, replacement: F) -> Self
    where
        F: Fn(&HashMap<usize, Rc<UOp>>) -> Rc<UOp> + 'static,
    {
        Self {
            name: name.into(),
            pattern,
            replacement: Box::new(replacement),
        }
    }
}

/// Rewriter - UOpグラフに書き換えルールを適用
pub struct Rewriter {
    rules: Vec<RewriteRule>,
}

impl Rewriter {
    pub fn new(rules: Vec<RewriteRule>) -> Self {
        Self { rules }
    }

    /// UOpに書き換えルールを適用（1回のみ）
    pub fn apply_once(&self, uop: &Rc<UOp>) -> Option<Rc<UOp>> {
        // まず子ノードに再帰的に適用
        let (new_uop, changed) = self.apply_to_children(uop);

        let current = if changed { new_uop } else { uop.clone() };

        // 各ルールを試す
        for rule in &self.rules {
            if let Some(mapping) = pattern_match(&rule.pattern, &current) {
                return Some((rule.replacement)(&mapping));
            }
        }

        if changed {
            Some(current)
        } else {
            None
        }
    }

    /// 子ノードに再帰的にルールを適用
    fn apply_to_children(&self, uop: &Rc<UOp>) -> (Rc<UOp>, bool) {
        match &**uop {
            UOp::Input { .. }
            | UOp::Const { .. }
            | UOp::ThreadIdx { .. }
            | UOp::GroupIdx { .. }
            | UOp::Var { .. }
            | UOp::Barrier { .. }
            | UOp::Wildcard { .. } => (uop.clone(), false),

            UOp::Add { dtype, lhs, rhs } => {
                let (new_lhs, lhs_changed) = self
                    .apply_once(lhs)
                    .map(|r| (r, true))
                    .unwrap_or((lhs.clone(), false));
                let (new_rhs, rhs_changed) = self
                    .apply_once(rhs)
                    .map(|r| (r, true))
                    .unwrap_or((rhs.clone(), false));
                if lhs_changed || rhs_changed {
                    (
                        Rc::new(UOp::Add {
                            dtype: dtype.clone(),
                            lhs: new_lhs,
                            rhs: new_rhs,
                        }),
                        true,
                    )
                } else {
                    (uop.clone(), false)
                }
            }

            UOp::Mul { dtype, lhs, rhs } => {
                let (new_lhs, lhs_changed) = self
                    .apply_once(lhs)
                    .map(|r| (r, true))
                    .unwrap_or((lhs.clone(), false));
                let (new_rhs, rhs_changed) = self
                    .apply_once(rhs)
                    .map(|r| (r, true))
                    .unwrap_or((rhs.clone(), false));
                if lhs_changed || rhs_changed {
                    (
                        Rc::new(UOp::Mul {
                            dtype: dtype.clone(),
                            lhs: new_lhs,
                            rhs: new_rhs,
                        }),
                        true,
                    )
                } else {
                    (uop.clone(), false)
                }
            }

            UOp::Max { dtype, lhs, rhs } => {
                let (new_lhs, lhs_changed) = self
                    .apply_once(lhs)
                    .map(|r| (r, true))
                    .unwrap_or((lhs.clone(), false));
                let (new_rhs, rhs_changed) = self
                    .apply_once(rhs)
                    .map(|r| (r, true))
                    .unwrap_or((rhs.clone(), false));
                if lhs_changed || rhs_changed {
                    (
                        Rc::new(UOp::Max {
                            dtype: dtype.clone(),
                            lhs: new_lhs,
                            rhs: new_rhs,
                        }),
                        true,
                    )
                } else {
                    (uop.clone(), false)
                }
            }

            UOp::Recip { dtype, arg } => {
                let (new_arg, changed) = self
                    .apply_once(arg)
                    .map(|r| (r, true))
                    .unwrap_or((arg.clone(), false));
                if changed {
                    (
                        Rc::new(UOp::Recip {
                            dtype: dtype.clone(),
                            arg: new_arg,
                        }),
                        true,
                    )
                } else {
                    (uop.clone(), false)
                }
            }

            UOp::Sqrt { dtype, arg } => {
                let (new_arg, changed) = self
                    .apply_once(arg)
                    .map(|r| (r, true))
                    .unwrap_or((arg.clone(), false));
                if changed {
                    (
                        Rc::new(UOp::Sqrt {
                            dtype: dtype.clone(),
                            arg: new_arg,
                        }),
                        true,
                    )
                } else {
                    (uop.clone(), false)
                }
            }

            UOp::Sequence { dtype, ops } => {
                let mut changed = false;
                let new_ops: Vec<Rc<UOp>> = ops
                    .iter()
                    .map(|o| {
                        if let Some(r) = self.apply_once(o) {
                            changed = true;
                            r
                        } else {
                            o.clone()
                        }
                    })
                    .collect();
                if changed {
                    (
                        Rc::new(UOp::Sequence {
                            dtype: dtype.clone(),
                            ops: new_ops,
                        }),
                        true,
                    )
                } else {
                    (uop.clone(), false)
                }
            }

            // その他のバリアントは簡略化のため変更なしとする
            _ => (uop.clone(), false),
        }
    }

    /// UOpに書き換えルールを繰り返し適用（不動点まで）
    pub fn apply(&self, uop: &Rc<UOp>, max_iterations: usize) -> Rc<UOp> {
        let mut current = uop.clone();
        for _ in 0..max_iterations {
            if let Some(rewritten) = self.apply_once(&current) {
                current = rewritten;
            } else {
                break;
            }
        }
        current
    }
}

/// パターンマッチング
/// patternをuopにマッチさせ、Wildcardのマッピングを返す
fn pattern_match(pattern: &Rc<UOp>, uop: &Rc<UOp>) -> Option<HashMap<usize, Rc<UOp>>> {
    let mut mapping = HashMap::new();
    if pattern_match_impl(pattern, uop, &mut mapping) {
        Some(mapping)
    } else {
        None
    }
}

fn pattern_match_impl(
    pattern: &Rc<UOp>,
    uop: &Rc<UOp>,
    mapping: &mut HashMap<usize, Rc<UOp>>,
) -> bool {
    match (&**pattern, &**uop) {
        (UOp::Wildcard { id, .. }, _) => {
            // すでにマッピングがある場合は一致を確認
            if let Some(existing) = mapping.get(id) {
                existing == uop
            } else {
                mapping.insert(*id, uop.clone());
                true
            }
        }

        // 子がないノードは値の比較
        (
            UOp::Const {
                value: v1,
                dtype: d1,
            },
            UOp::Const {
                value: v2,
                dtype: d2,
            },
        ) => (v1 - v2).abs() < f64::EPSILON && d1 == d2,

        (
            UOp::Var {
                name: n1,
                dtype: d1,
            },
            UOp::Var {
                name: n2,
                dtype: d2,
            },
        ) => n1 == n2 && d1 == d2,

        (
            UOp::ThreadIdx {
                dim: dim1,
                dtype: d1,
            },
            UOp::ThreadIdx {
                dim: dim2,
                dtype: d2,
            },
        ) => dim1 == dim2 && d1 == d2,

        (
            UOp::GroupIdx {
                dim: dim1,
                dtype: d1,
            },
            UOp::GroupIdx {
                dim: dim2,
                dtype: d2,
            },
        ) => dim1 == dim2 && d1 == d2,

        // 二項演算
        (
            UOp::Add {
                lhs: l1, rhs: r1, ..
            },
            UOp::Add {
                lhs: l2, rhs: r2, ..
            },
        )
        | (
            UOp::Mul {
                lhs: l1, rhs: r1, ..
            },
            UOp::Mul {
                lhs: l2, rhs: r2, ..
            },
        )
        | (
            UOp::Max {
                lhs: l1, rhs: r1, ..
            },
            UOp::Max {
                lhs: l2, rhs: r2, ..
            },
        )
        | (
            UOp::Rem {
                lhs: l1, rhs: r1, ..
            },
            UOp::Rem {
                lhs: l2, rhs: r2, ..
            },
        )
        | (
            UOp::Idiv {
                lhs: l1, rhs: r1, ..
            },
            UOp::Idiv {
                lhs: l2, rhs: r2, ..
            },
        )
        | (
            UOp::LessThan {
                lhs: l1, rhs: r1, ..
            },
            UOp::LessThan {
                lhs: l2, rhs: r2, ..
            },
        ) => pattern_match_impl(l1, l2, mapping) && pattern_match_impl(r1, r2, mapping),

        // 単項演算
        (UOp::Recip { arg: a1, .. }, UOp::Recip { arg: a2, .. })
        | (UOp::Sqrt { arg: a1, .. }, UOp::Sqrt { arg: a2, .. }) => {
            pattern_match_impl(a1, a2, mapping)
        }

        // Select
        (
            UOp::Select {
                cond: c1,
                then_: t1,
                else_: e1,
                ..
            },
            UOp::Select {
                cond: c2,
                then_: t2,
                else_: e2,
                ..
            },
        ) => {
            pattern_match_impl(c1, c2, mapping)
                && pattern_match_impl(t1, t2, mapping)
                && pattern_match_impl(e1, e2, mapping)
        }

        // Sequence
        (UOp::Sequence { ops: ops1, .. }, UOp::Sequence { ops: ops2, .. }) => {
            if ops1.len() != ops2.len() {
                return false;
            }
            for (o1, o2) in ops1.iter().zip(ops2.iter()) {
                if !pattern_match_impl(o1, o2, mapping) {
                    return false;
                }
            }
            true
        }

        // その他のノードはバリアントが異なればマッチしない
        _ => false,
    }
}

/// 基本的な最適化ルールを生成
pub fn basic_optimization_rules() -> Vec<RewriteRule> {
    vec![
        // x + 0 = x
        RewriteRule::new(
            "add_zero_right",
            helper::add(
                helper::wildcard(0, DType::F32),
                helper::const_val(0.0, DType::F32),
            ),
            |m| m[&0].clone(),
        ),
        // 0 + x = x
        RewriteRule::new(
            "add_zero_left",
            helper::add(
                helper::const_val(0.0, DType::F32),
                helper::wildcard(0, DType::F32),
            ),
            |m| m[&0].clone(),
        ),
        // x * 1 = x
        RewriteRule::new(
            "mul_one_right",
            helper::mul(
                helper::wildcard(0, DType::F32),
                helper::const_val(1.0, DType::F32),
            ),
            |m| m[&0].clone(),
        ),
        // 1 * x = x
        RewriteRule::new(
            "mul_one_left",
            helper::mul(
                helper::const_val(1.0, DType::F32),
                helper::wildcard(0, DType::F32),
            ),
            |m| m[&0].clone(),
        ),
        // x * 0 = 0
        RewriteRule::new(
            "mul_zero_right",
            helper::mul(
                helper::wildcard(0, DType::F32),
                helper::const_val(0.0, DType::F32),
            ),
            |_| helper::const_val(0.0, DType::F32),
        ),
        // 0 * x = 0
        RewriteRule::new(
            "mul_zero_left",
            helper::mul(
                helper::const_val(0.0, DType::F32),
                helper::wildcard(0, DType::F32),
            ),
            |_| helper::const_val(0.0, DType::F32),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper;

    #[test]
    fn test_pattern_match_wildcard() {
        let pattern = helper::wildcard(0, DType::F32);
        let uop = helper::const_val(42.0, DType::F32);

        let mapping = pattern_match(&pattern, &uop).unwrap();
        assert_eq!(mapping.len(), 1);
        assert_eq!(&mapping[&0], &uop);
    }

    #[test]
    fn test_pattern_match_add() {
        let pattern = helper::add(
            helper::wildcard(0, DType::F32),
            helper::wildcard(1, DType::F32),
        );
        let a = helper::const_val(1.0, DType::F32);
        let b = helper::const_val(2.0, DType::F32);
        let uop = helper::add(a.clone(), b.clone());

        let mapping = pattern_match(&pattern, &uop).unwrap();
        assert_eq!(mapping.len(), 2);
        assert_eq!(&mapping[&0], &a);
        assert_eq!(&mapping[&1], &b);
    }

    #[test]
    fn test_rewrite_add_zero() {
        let rules = basic_optimization_rules();
        let rewriter = Rewriter::new(rules);

        let x = helper::var("x", DType::F32);
        let zero = helper::const_val(0.0, DType::F32);
        let expr = helper::add(x.clone(), zero);

        let result = rewriter.apply(&expr, 10);

        // x + 0 → x
        match &*result {
            UOp::Var { name, .. } => assert_eq!(name, "x"),
            _ => panic!("Expected Var, got {:?}", result),
        }
    }

    #[test]
    fn test_rewrite_mul_one() {
        let rules = basic_optimization_rules();
        let rewriter = Rewriter::new(rules);

        let x = helper::var("x", DType::F32);
        let one = helper::const_val(1.0, DType::F32);
        let expr = helper::mul(x.clone(), one);

        let result = rewriter.apply(&expr, 10);

        // x * 1 → x
        match &*result {
            UOp::Var { name, .. } => assert_eq!(name, "x"),
            _ => panic!("Expected Var, got {:?}", result),
        }
    }

    #[test]
    fn test_rewrite_nested() {
        let rules = basic_optimization_rules();
        let rewriter = Rewriter::new(rules);

        let x = helper::var("x", DType::F32);
        let zero = helper::const_val(0.0, DType::F32);
        let one = helper::const_val(1.0, DType::F32);

        // (x * 1) + 0
        let expr = helper::add(helper::mul(x.clone(), one), zero);

        let result = rewriter.apply(&expr, 10);

        // (x * 1) + 0 → x + 0 → x
        match &*result {
            UOp::Var { name, .. } => assert_eq!(name, "x"),
            _ => panic!("Expected Var, got {:?}", result),
        }
    }
}
