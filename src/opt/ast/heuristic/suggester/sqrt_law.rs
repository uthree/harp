use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester for square root laws
///
/// Laws:
/// - sqrt(a*b) = sqrt(a) * sqrt(b)
/// - sqrt(a) * sqrt(b) = sqrt(a*b)
/// - sqrt(a/b) = sqrt(a) / sqrt(b)
/// - sqrt(a) / sqrt(b) = sqrt(a/b)
pub struct SqrtLawSuggester;

impl RewriteSuggester for SqrtLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // sqrt(a*b) -> sqrt(a) * sqrt(b)
        if let AstNode::Sqrt(inner) = node {
            if let AstNode::Mul(a, b) = &**inner {
                suggestions.push(AstNode::Mul(
                    Box::new(AstNode::Sqrt(a.clone())),
                    Box::new(AstNode::Sqrt(b.clone())),
                ));
            }
        }

        // sqrt(a) * sqrt(b) -> sqrt(a*b)
        if let AstNode::Mul(left, right) = node {
            if let (AstNode::Sqrt(a), AstNode::Sqrt(b)) = (&**left, &**right) {
                suggestions.push(AstNode::Sqrt(Box::new(AstNode::Mul(a.clone(), b.clone()))));
            }
        }

        // sqrt(a * Recip(b)) -> sqrt(a) * Recip(sqrt(b))
        // This represents sqrt(a/b) -> sqrt(a) / sqrt(b)
        if let AstNode::Sqrt(inner) = node {
            if let AstNode::Mul(a, recip) = &**inner {
                if let AstNode::Recip(b) = &**recip {
                    suggestions.push(AstNode::Mul(
                        Box::new(AstNode::Sqrt(a.clone())),
                        Box::new(AstNode::Recip(Box::new(AstNode::Sqrt(b.clone())))),
                    ));
                }
            }
        }

        // sqrt(a) * Recip(sqrt(b)) -> sqrt(a * Recip(b))
        // This represents sqrt(a) / sqrt(b) -> sqrt(a/b)
        if let AstNode::Mul(left, right) = node {
            if let (AstNode::Sqrt(a), AstNode::Recip(recip_inner)) = (&**left, &**right) {
                if let AstNode::Sqrt(b) = &**recip_inner {
                    suggestions.push(AstNode::Sqrt(Box::new(AstNode::Mul(
                        a.clone(),
                        Box::new(AstNode::Recip(b.clone())),
                    ))));
                }
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
    fn test_sqrt_product_law() {
        let suggester = SqrtLawSuggester;

        // sqrt(a*b) -> sqrt(a) * sqrt(b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Sqrt(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Mul(
            Box::new(AstNode::Sqrt(Box::new(a.clone()))),
            Box::new(AstNode::Sqrt(Box::new(b.clone()))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_product_of_sqrts_law() {
        let suggester = SqrtLawSuggester;

        // sqrt(a) * sqrt(b) -> sqrt(a*b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Mul(
            Box::new(AstNode::Sqrt(Box::new(a.clone()))),
            Box::new(AstNode::Sqrt(Box::new(b.clone()))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Sqrt(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_sqrt_division_law() {
        let suggester = SqrtLawSuggester;

        // sqrt(a/b) -> sqrt(a) / sqrt(b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Sqrt(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Recip(Box::new(b.clone()))),
        )));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Mul(
            Box::new(AstNode::Sqrt(Box::new(a.clone()))),
            Box::new(AstNode::Recip(Box::new(AstNode::Sqrt(Box::new(b.clone()))))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_division_of_sqrts_law() {
        let suggester = SqrtLawSuggester;

        // sqrt(a) / sqrt(b) -> sqrt(a/b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Mul(
            Box::new(AstNode::Sqrt(Box::new(a.clone()))),
            Box::new(AstNode::Recip(Box::new(AstNode::Sqrt(Box::new(b.clone()))))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Sqrt(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Recip(Box::new(b.clone()))),
        )));
        assert!(suggestions.contains(&expected));
    }
}
