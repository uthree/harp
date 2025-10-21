use crate::ast::AstNode;
use crate::opt::ast::RewriteSuggester;

/// A suggester for logarithm, exponential, and trigonometric laws
///
/// Logarithm laws:
/// - log2(a) + log2(b) = log2(a * b)
/// - log2(a * b) = log2(a) + log2(b)
/// - log2(a) - log2(b) = log2(a / b)
/// - log2(a / b) = log2(a) - log2(b)
///
/// Exponential laws:
/// - exp2(a) * exp2(b) = exp2(a + b)
/// - exp2(a + b) = exp2(a) * exp2(b)
/// - exp2(a) / exp2(b) = exp2(a - b)
/// - exp2(a - b) = exp2(a) / exp2(b)
///
/// Trigonometric laws:
/// - sin(-x) = -sin(x)
/// - -sin(x) = sin(-x)
pub struct LogExpLawSuggester;

impl RewriteSuggester for LogExpLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // log2(a) + log2(b) -> log2(a * b)
        if let AstNode::Add(left, right) = node {
            if let (AstNode::Log2(a), AstNode::Log2(b)) = (&**left, &**right) {
                suggestions.push(AstNode::Log2(Box::new(AstNode::Mul(a.clone(), b.clone()))));
            }
        }

        // log2(a * b) -> log2(a) + log2(b)
        if let AstNode::Log2(inner) = node {
            if let AstNode::Mul(a, b) = &**inner {
                suggestions.push(AstNode::Add(
                    Box::new(AstNode::Log2(a.clone())),
                    Box::new(AstNode::Log2(b.clone())),
                ));
            }
        }

        // exp2(a) * exp2(b) -> exp2(a + b)
        if let AstNode::Mul(left, right) = node {
            if let (AstNode::Exp2(a), AstNode::Exp2(b)) = (&**left, &**right) {
                suggestions.push(AstNode::Exp2(Box::new(AstNode::Add(a.clone(), b.clone()))));
            }
        }

        // exp2(a + b) -> exp2(a) * exp2(b)
        if let AstNode::Exp2(inner) = node {
            if let AstNode::Add(a, b) = &**inner {
                suggestions.push(AstNode::Mul(
                    Box::new(AstNode::Exp2(a.clone())),
                    Box::new(AstNode::Exp2(b.clone())),
                ));
            }
        }

        // Additional laws:
        // log2(a) - log2(b) -> log2(a / b)
        if let AstNode::Add(left, right) = node {
            if let (AstNode::Log2(a), AstNode::Neg(neg_inner)) = (&**left, &**right) {
                if let AstNode::Log2(b) = &**neg_inner {
                    // log2(a) + (-log2(b)) = log2(a) - log2(b) = log2(a / b)
                    suggestions.push(AstNode::Log2(Box::new(AstNode::Mul(
                        a.clone(),
                        Box::new(AstNode::Recip(b.clone())),
                    ))));
                }
            }
        }

        // log2(a / b) -> log2(a) - log2(b)
        if let AstNode::Log2(inner) = node {
            if let AstNode::Mul(a, recip) = &**inner {
                if let AstNode::Recip(b) = &**recip {
                    suggestions.push(AstNode::Add(
                        Box::new(AstNode::Log2(a.clone())),
                        Box::new(AstNode::Neg(Box::new(AstNode::Log2(b.clone())))),
                    ));
                }
            }
        }

        // exp2(a - b) -> exp2(a) / exp2(b)
        if let AstNode::Exp2(inner) = node {
            if let AstNode::Add(a, neg) = &**inner {
                if let AstNode::Neg(b) = &**neg {
                    suggestions.push(AstNode::Mul(
                        Box::new(AstNode::Exp2(a.clone())),
                        Box::new(AstNode::Recip(Box::new(AstNode::Exp2(b.clone())))),
                    ));
                }
            }
        }

        // exp2(a) / exp2(b) -> exp2(a - b)
        if let AstNode::Mul(left, right) = node {
            if let (AstNode::Exp2(a), AstNode::Recip(recip_inner)) = (&**left, &**right) {
                if let AstNode::Exp2(b) = &**recip_inner {
                    suggestions.push(AstNode::Exp2(Box::new(AstNode::Add(
                        a.clone(),
                        Box::new(AstNode::Neg(b.clone())),
                    ))));
                }
            }
        }

        // Trigonometric identities
        // sin(-x) -> -sin(x)
        if let AstNode::Sin(inner) = node {
            if let AstNode::Neg(x) = &**inner {
                suggestions.push(AstNode::Neg(Box::new(AstNode::Sin(x.clone()))));
            }
        }

        // -sin(x) -> sin(-x)
        if let AstNode::Neg(inner) = node {
            if let AstNode::Sin(x) = &**inner {
                suggestions.push(AstNode::Sin(Box::new(AstNode::Neg(x.clone()))));
            }
        }

        // Recursively suggest in children
        for (i, child) in node.children().iter().enumerate() {
            for suggested_child in self.suggest(child) {
                let mut new_children: Vec<AstNode> =
                    node.children().iter().map(|c| (*c).clone()).collect();
                new_children[i] = suggested_child;
                suggestions.push(node.clone().replace_children(new_children));
            }
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
    fn test_log_addition_law() {
        let suggester = LogExpLawSuggester;

        // log2(a) + log2(b) -> log2(a * b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Add(
            Box::new(AstNode::Log2(Box::new(a.clone()))),
            Box::new(AstNode::Log2(Box::new(b.clone()))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Log2(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_log_multiplication_law() {
        let suggester = LogExpLawSuggester;

        // log2(a * b) -> log2(a) + log2(b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Log2(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(AstNode::Log2(Box::new(a.clone()))),
            Box::new(AstNode::Log2(Box::new(b.clone()))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_exp_multiplication_law() {
        let suggester = LogExpLawSuggester;

        // exp2(a) * exp2(b) -> exp2(a + b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Mul(
            Box::new(AstNode::Exp2(Box::new(a.clone()))),
            Box::new(AstNode::Exp2(Box::new(b.clone()))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Exp2(Box::new(AstNode::Add(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_exp_addition_law() {
        let suggester = LogExpLawSuggester;

        // exp2(a + b) -> exp2(a) * exp2(b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Exp2(Box::new(AstNode::Add(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Mul(
            Box::new(AstNode::Exp2(Box::new(a.clone()))),
            Box::new(AstNode::Exp2(Box::new(b.clone()))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_log_subtraction_law() {
        let suggester = LogExpLawSuggester;

        // log2(a) - log2(b) -> log2(a / b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Add(
            Box::new(AstNode::Log2(Box::new(a.clone()))),
            Box::new(AstNode::Neg(Box::new(AstNode::Log2(Box::new(b.clone()))))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Log2(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Recip(Box::new(b.clone()))),
        )));
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_log_division_law() {
        let suggester = LogExpLawSuggester;

        // log2(a / b) -> log2(a) - log2(b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Log2(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Recip(Box::new(b.clone()))),
        )));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(AstNode::Log2(Box::new(a.clone()))),
            Box::new(AstNode::Neg(Box::new(AstNode::Log2(Box::new(b.clone()))))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_exp_subtraction_law() {
        let suggester = LogExpLawSuggester;

        // exp2(a - b) -> exp2(a) / exp2(b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Exp2(Box::new(AstNode::Add(
            Box::new(a.clone()),
            Box::new(AstNode::Neg(Box::new(b.clone()))),
        )));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Mul(
            Box::new(AstNode::Exp2(Box::new(a.clone()))),
            Box::new(AstNode::Recip(Box::new(AstNode::Exp2(Box::new(b.clone()))))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_exp_division_law() {
        let suggester = LogExpLawSuggester;

        // exp2(a) / exp2(b) -> exp2(a - b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Mul(
            Box::new(AstNode::Exp2(Box::new(a.clone()))),
            Box::new(AstNode::Recip(Box::new(AstNode::Exp2(Box::new(b.clone()))))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Exp2(Box::new(AstNode::Add(
            Box::new(a.clone()),
            Box::new(AstNode::Neg(Box::new(b.clone()))),
        )));
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_sin_negation_law() {
        let suggester = LogExpLawSuggester;

        // sin(-x) -> -sin(x)
        let x = var("x");
        let ast = AstNode::Sin(Box::new(AstNode::Neg(Box::new(x.clone()))));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Neg(Box::new(AstNode::Sin(Box::new(x.clone()))));
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_neg_sin_law() {
        let suggester = LogExpLawSuggester;

        // -sin(x) -> sin(-x)
        let x = var("x");
        let ast = AstNode::Neg(Box::new(AstNode::Sin(Box::new(x.clone()))));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Sin(Box::new(AstNode::Neg(Box::new(x.clone()))));
        assert!(suggestions.contains(&expected));
    }
}
