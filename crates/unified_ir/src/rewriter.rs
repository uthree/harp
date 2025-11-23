use crate::uop::{UOp, UOpKind};
use harp::DType;
use std::collections::HashMap;

/// 書き換えルール
/// pattern にマッチする部分を replacement に置き換える
pub struct RewriteRule {
    pub name: String,
    pub pattern: UOp,
    pub replacement: Box<dyn Fn(&HashMap<usize, UOp>) -> UOp>,
}

impl RewriteRule {
    pub fn new<F>(name: impl Into<String>, pattern: UOp, replacement: F) -> Self
    where
        F: Fn(&HashMap<usize, UOp>) -> UOp + 'static,
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
    pub fn apply_once(&self, uop: &UOp) -> Option<UOp> {
        // まず子ノードに再帰的に適用
        let mut src_changed = false;
        let new_src: Vec<UOp> = uop
            .0
            .src
            .iter()
            .map(|s| {
                if let Some(rewritten) = self.apply_once(s) {
                    src_changed = true;
                    rewritten
                } else {
                    s.clone()
                }
            })
            .collect();

        let current = if src_changed {
            UOp::new(uop.0.op.clone(), uop.0.dtype.clone(), new_src)
        } else {
            uop.clone()
        };

        // 各ルールを試す
        for rule in &self.rules {
            if let Some(mapping) = pattern_match(&rule.pattern, &current) {
                return Some((rule.replacement)(&mapping));
            }
        }

        if src_changed {
            Some(current)
        } else {
            None
        }
    }

    /// UOpに書き換えルールを繰り返し適用（不動点まで）
    pub fn apply(&self, uop: &UOp, max_iterations: usize) -> UOp {
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
fn pattern_match(pattern: &UOp, uop: &UOp) -> Option<HashMap<usize, UOp>> {
    let mut mapping = HashMap::new();
    if pattern_match_impl(pattern, uop, &mut mapping) {
        Some(mapping)
    } else {
        None
    }
}

fn pattern_match_impl(pattern: &UOp, uop: &UOp, mapping: &mut HashMap<usize, UOp>) -> bool {
    match &pattern.0.op {
        UOpKind::Wildcard { id } => {
            // すでにマッピングがある場合は一致を確認
            if let Some(existing) = mapping.get(id) {
                existing == uop
            } else {
                mapping.insert(*id, uop.clone());
                true
            }
        }
        _ => {
            // 演算の種類が一致するか確認
            if pattern.0.op != uop.0.op {
                return false;
            }

            // 子ノードの数が一致するか確認
            if pattern.0.src.len() != uop.0.src.len() {
                return false;
            }

            // 再帰的に子ノードをマッチ
            for (p, u) in pattern.0.src.iter().zip(&uop.0.src) {
                if !pattern_match_impl(p, u, mapping) {
                    return false;
                }
            }

            true
        }
    }
}

/// 基本的な最適化ルールを生成
pub fn basic_optimization_rules() -> Vec<RewriteRule> {
    let mut rules = Vec::new();

    // ========== 定数畳み込み ==========

    // x + 0 = x
    rules.push(RewriteRule::new(
        "add_zero_right",
        UOp::add(
            UOp::wildcard(0, DType::F32),
            UOp::const_val(0.0, DType::F32),
        ),
        |m| m[&0].clone(),
    ));

    // 0 + x = x
    rules.push(RewriteRule::new(
        "add_zero_left",
        UOp::add(
            UOp::const_val(0.0, DType::F32),
            UOp::wildcard(0, DType::F32),
        ),
        |m| m[&0].clone(),
    ));

    // x * 1 = x
    rules.push(RewriteRule::new(
        "mul_one_right",
        UOp::mul(
            UOp::wildcard(0, DType::F32),
            UOp::const_val(1.0, DType::F32),
        ),
        |m| m[&0].clone(),
    ));

    // 1 * x = x
    rules.push(RewriteRule::new(
        "mul_one_left",
        UOp::mul(
            UOp::const_val(1.0, DType::F32),
            UOp::wildcard(0, DType::F32),
        ),
        |m| m[&0].clone(),
    ));

    // x * 0 = 0
    rules.push(RewriteRule::new(
        "mul_zero_right",
        UOp::mul(
            UOp::wildcard(0, DType::F32),
            UOp::const_val(0.0, DType::F32),
        ),
        |_| UOp::const_val(0.0, DType::F32),
    ));

    // 0 * x = 0
    rules.push(RewriteRule::new(
        "mul_zero_left",
        UOp::mul(
            UOp::const_val(0.0, DType::F32),
            UOp::wildcard(0, DType::F32),
        ),
        |_| UOp::const_val(0.0, DType::F32),
    ));

    // 定数同士の加算
    // TODO: 実行時に定数値を取得する仕組みが必要

    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_match_wildcard() {
        let pattern = UOp::wildcard(0, DType::F32);
        let uop = UOp::const_val(42.0, DType::F32);

        let mapping = pattern_match(&pattern, &uop).unwrap();
        assert_eq!(mapping.len(), 1);
        assert_eq!(&mapping[&0], &uop);
    }

    #[test]
    fn test_pattern_match_add() {
        let pattern = UOp::add(UOp::wildcard(0, DType::F32), UOp::wildcard(1, DType::F32));
        let a = UOp::const_val(1.0, DType::F32);
        let b = UOp::const_val(2.0, DType::F32);
        let uop = UOp::add(a.clone(), b.clone());

        let mapping = pattern_match(&pattern, &uop).unwrap();
        assert_eq!(mapping.len(), 2);
        assert_eq!(&mapping[&0], &a);
        assert_eq!(&mapping[&1], &b);
    }

    #[test]
    fn test_rewrite_add_zero() {
        let rules = basic_optimization_rules();
        let rewriter = Rewriter::new(rules);

        let x = UOp::var("x", DType::F32);
        let zero = UOp::const_val(0.0, DType::F32);
        let expr = UOp::add(x.clone(), zero);

        let result = rewriter.apply(&expr, 10);

        // x + 0 → x
        match &result.0.op {
            UOpKind::Var { name } => assert_eq!(name, "x"),
            _ => panic!("Expected Var, got {:?}", result.0.op),
        }
    }

    #[test]
    fn test_rewrite_mul_one() {
        let rules = basic_optimization_rules();
        let rewriter = Rewriter::new(rules);

        let x = UOp::var("x", DType::F32);
        let one = UOp::const_val(1.0, DType::F32);
        let expr = UOp::mul(x.clone(), one);

        let result = rewriter.apply(&expr, 10);

        // x * 1 → x
        match &result.0.op {
            UOpKind::Var { name } => assert_eq!(name, "x"),
            _ => panic!("Expected Var, got {:?}", result.0.op),
        }
    }

    #[test]
    fn test_rewrite_nested() {
        let rules = basic_optimization_rules();
        let rewriter = Rewriter::new(rules);

        let x = UOp::var("x", DType::F32);
        let zero = UOp::const_val(0.0, DType::F32);
        let one = UOp::const_val(1.0, DType::F32);

        // (x * 1) + 0
        let expr = UOp::add(UOp::mul(x.clone(), one), zero);

        let result = rewriter.apply(&expr, 10);

        // (x * 1) + 0 → x + 0 → x
        match &result.0.op {
            UOpKind::Var { name } => assert_eq!(name, "x"),
            _ => panic!("Expected Var, got {:?}", result.0.op),
        }
    }
}
