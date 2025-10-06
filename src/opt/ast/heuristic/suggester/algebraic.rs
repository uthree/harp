use crate::ast::AstNode;
use crate::ast_pattern;
use crate::opt::ast::heuristic::RewriteSuggester;
use std::rc::Rc;

/// A suggester for distributive and associative laws
pub struct AlgebraicLawSuggester {
    rules: Vec<Rc<crate::ast::pattern::AstRewriteRule>>,
}

impl AlgebraicLawSuggester {
    /// Try to fold two constant additions
    fn try_fold_add(a: &AstNode, b: &AstNode) -> Option<AstNode> {
        use crate::ast::ConstLiteral;
        match (a, b) {
            (AstNode::Const(ConstLiteral::Isize(x)), AstNode::Const(ConstLiteral::Isize(y))) => {
                Some(AstNode::Const(ConstLiteral::Isize(x + y)))
            }
            (AstNode::Const(ConstLiteral::Usize(x)), AstNode::Const(ConstLiteral::Usize(y))) => {
                Some(AstNode::Const(ConstLiteral::Usize(x + y)))
            }
            (AstNode::Const(ConstLiteral::F32(x)), AstNode::Const(ConstLiteral::F32(y))) => {
                Some(AstNode::Const(ConstLiteral::F32(x + y)))
            }
            _ => None,
        }
    }

    /// Try to fold two constant multiplications
    fn try_fold_mul(a: &AstNode, b: &AstNode) -> Option<AstNode> {
        use crate::ast::ConstLiteral;
        match (a, b) {
            (AstNode::Const(ConstLiteral::Isize(x)), AstNode::Const(ConstLiteral::Isize(y))) => {
                Some(AstNode::Const(ConstLiteral::Isize(x * y)))
            }
            (AstNode::Const(ConstLiteral::Usize(x)), AstNode::Const(ConstLiteral::Usize(y))) => {
                Some(AstNode::Const(ConstLiteral::Usize(x * y)))
            }
            (AstNode::Const(ConstLiteral::F32(x)), AstNode::Const(ConstLiteral::F32(y))) => {
                Some(AstNode::Const(ConstLiteral::F32(x * y)))
            }
            _ => None,
        }
    }

    pub fn new() -> Self {
        let rules = vec![
            // ===== Identity elimination =====
            // x + 0 -> x
            ast_pattern!(|a| a.clone() + AstNode::from(0isize) => a.clone()),
            // 0 + x -> x
            ast_pattern!(|a| AstNode::from(0isize) + a.clone() => a.clone()),
            // x * 1 -> x
            ast_pattern!(|a| a.clone() * AstNode::from(1isize) => a.clone()),
            // 1 * x -> x
            ast_pattern!(|a| AstNode::from(1isize) * a.clone() => a.clone()),
            // x * 0 -> 0
            ast_pattern!(|_a| _a.clone() * AstNode::from(0isize) => AstNode::from(0isize)),
            // 0 * x -> 0
            ast_pattern!(|_a| AstNode::from(0isize) * _a.clone() => AstNode::from(0isize)),
            // ===== Constant folding =====
            // c1 + c2 -> c1+c2
            ast_pattern!(|a, b| a.clone() + b.clone(), if Self::try_fold_add(a, b).is_some() => Self::try_fold_add(a, b).unwrap()),
            // c1 * c2 -> c1*c2
            ast_pattern!(|a, b| a.clone() * b.clone(), if Self::try_fold_mul(a, b).is_some() => Self::try_fold_mul(a, b).unwrap()),
            // -(-x) -> x
            ast_pattern!(|a| -(-a.clone()) => a.clone()),
            // Constant negation: -c -> -c (folded)
            ast_pattern!(|a| -a.clone(), if matches!(a, AstNode::Const(_)) => {
                match a {
                    AstNode::Const(crate::ast::ConstLiteral::Isize(x)) => AstNode::Const(crate::ast::ConstLiteral::Isize(-x)),
                    AstNode::Const(crate::ast::ConstLiteral::F32(x)) => AstNode::Const(crate::ast::ConstLiteral::F32(-x)),
                    _ => -a.clone(),
                }
            }),
            // Constant reciprocal folding
            ast_pattern!(|a| AstNode::Recip(Box::new(a.clone())), if matches!(a, AstNode::Const(crate::ast::ConstLiteral::F32(_))) => {
                match a {
                    AstNode::Const(crate::ast::ConstLiteral::F32(x)) if *x != 0.0 => {
                        AstNode::Const(crate::ast::ConstLiteral::F32(1.0 / x))
                    }
                    _ => AstNode::Recip(Box::new(a.clone())),
                }
            }),
            // Constant max folding
            ast_pattern!(|a, b| AstNode::Max(Box::new(a.clone()), Box::new(b.clone())),
                if matches!((a, b), (AstNode::Const(_), AstNode::Const(_))) => {
                    use crate::ast::ConstLiteral;
                    match (a, b) {
                        (AstNode::Const(ConstLiteral::Isize(x)), AstNode::Const(ConstLiteral::Isize(y))) => {
                            AstNode::Const(ConstLiteral::Isize(*x.max(y)))
                        }
                        (AstNode::Const(ConstLiteral::Usize(x)), AstNode::Const(ConstLiteral::Usize(y))) => {
                            AstNode::Const(ConstLiteral::Usize(*x.max(y)))
                        }
                        (AstNode::Const(ConstLiteral::F32(x)), AstNode::Const(ConstLiteral::F32(y))) => {
                            AstNode::Const(ConstLiteral::F32(x.max(*y)))
                        }
                        _ => AstNode::Max(Box::new(a.clone()), Box::new(b.clone())),
                    }
                }
            ),
            // Constant rem folding
            ast_pattern!(|a, b| AstNode::Rem(Box::new(a.clone()), Box::new(b.clone())),
                if matches!((a, b), (AstNode::Const(_), AstNode::Const(_))) => {
                    use crate::ast::ConstLiteral;
                    match (a, b) {
                        (AstNode::Const(ConstLiteral::Isize(x)), AstNode::Const(ConstLiteral::Isize(y))) if *y != 0 => {
                            AstNode::Const(ConstLiteral::Isize(x % y))
                        }
                        (AstNode::Const(ConstLiteral::Usize(x)), AstNode::Const(ConstLiteral::Usize(y))) if *y != 0 => {
                            AstNode::Const(ConstLiteral::Usize(x % y))
                        }
                        _ => AstNode::Rem(Box::new(a.clone()), Box::new(b.clone())),
                    }
                }
            ),
            // ===== Distributive laws =====
            // a * (b + c) -> a*b + a*c
            ast_pattern!(|a, b, c| a.clone() * (b.clone() + c.clone()) => a.clone() * b.clone() + a.clone() * c.clone()),
            // (a + b) * c -> a*c + b*c
            ast_pattern!(|a, b, c| (a.clone() + b.clone()) * c.clone() => a.clone() * c.clone() + b.clone() * c.clone()),
            // Reverse distributive: a*b + a*c -> a * (b + c)
            ast_pattern!(|a, b, c| a.clone() * b.clone() + a.clone() * c.clone() => a.clone() * (b.clone() + c.clone())),
            // ===== Associative laws =====
            // (a + b) + c -> a + (b + c)
            ast_pattern!(|a, b, c| (a.clone() + b.clone()) + c.clone() => a.clone() + (b.clone() + c.clone())),
            // a + (b + c) -> (a + b) + c
            ast_pattern!(|a, b, c| a.clone() + (b.clone() + c.clone()) => (a.clone() + b.clone()) + c.clone()),
            // (a * b) * c -> a * (b * c)
            ast_pattern!(|a, b, c| (a.clone() * b.clone()) * c.clone() => a.clone() * (b.clone() * c.clone())),
            // a * (b * c) -> (a * b) * c
            ast_pattern!(|a, b, c| a.clone() * (b.clone() * c.clone()) => (a.clone() * b.clone()) * c.clone()),
        ];

        Self { rules }
    }
}

impl Default for AlgebraicLawSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteSuggester for AlgebraicLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();
        for rule in &self.rules {
            suggestions.extend(rule.get_possible_rewrites(node));
        }
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> AstNode {
        AstNode::Var(name.to_string())
    }

    #[test]
    fn test_algebraic_law_suggester() {
        let suggester = AlgebraicLawSuggester::new();

        // Distributive: a * (b + c) -> a*b + a*c
        let a = var("a");
        let b = var("b");
        let c = var("c");
        let ast = AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Add(Box::new(b.clone()), Box::new(c.clone()))),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(c.clone()))),
        );
        assert!(suggestions.contains(&expected));

        // Reverse distributive: a*b + a*c -> a * (b + c)
        let ast = AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(c.clone()))),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Add(Box::new(b.clone()), Box::new(c.clone()))),
        );
        assert!(suggestions.contains(&expected));

        // Associative: (a + b) + c -> a + (b + c)
        let ast = AstNode::Add(
            Box::new(AstNode::Add(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(c.clone()),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(a.clone()),
            Box::new(AstNode::Add(Box::new(b.clone()), Box::new(c.clone()))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_constant_folding() {
        let suggester = AlgebraicLawSuggester::new();

        // 2 + 3 -> 5
        let ast = AstNode::from(2isize) + AstNode::from(3isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(5isize)));

        // 2 * 3 -> 6
        let ast = AstNode::from(2isize) * AstNode::from(3isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(6isize)));
    }

    #[test]
    fn test_identity_elimination() {
        let suggester = AlgebraicLawSuggester::new();
        let x = var("x");

        // x + 0 -> x
        let ast = x.clone() + AstNode::from(0isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&x));

        // x * 1 -> x
        let ast = x.clone() * AstNode::from(1isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&x));

        // x * 0 -> 0
        let ast = x.clone() * AstNode::from(0isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(0isize)));
    }
}
