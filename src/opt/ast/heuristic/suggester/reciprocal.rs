use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester for reciprocal laws
///
/// Laws:
/// - 1/(a*b) = 1/a * 1/b
/// - 1/a * 1/b = 1/(a*b)
pub struct ReciprocalLawSuggester;

impl RewriteSuggester for ReciprocalLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Recip(a*b) -> Recip(a) * Recip(b)
        if let AstNode::Recip(inner) = node {
            if let AstNode::Mul(a, b) = &**inner {
                suggestions.push(AstNode::Mul(
                    Box::new(AstNode::Recip(a.clone())),
                    Box::new(AstNode::Recip(b.clone())),
                ));
            }
        }

        // Recip(a) * Recip(b) -> Recip(a*b)
        if let AstNode::Mul(left, right) = node {
            if let (AstNode::Recip(a), AstNode::Recip(b)) = (&**left, &**right) {
                suggestions.push(AstNode::Recip(Box::new(AstNode::Mul(a.clone(), b.clone()))));
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
    fn test_recip_product_to_product_of_recips() {
        let suggester = ReciprocalLawSuggester;

        // Recip(a*b) -> Recip(a) * Recip(b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Recip(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Mul(
            Box::new(AstNode::Recip(Box::new(a.clone()))),
            Box::new(AstNode::Recip(Box::new(b.clone()))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_product_of_recips_to_recip_product() {
        let suggester = ReciprocalLawSuggester;

        // Recip(a) * Recip(b) -> Recip(a*b)
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Mul(
            Box::new(AstNode::Recip(Box::new(a.clone()))),
            Box::new(AstNode::Recip(Box::new(b.clone()))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Recip(Box::new(AstNode::Mul(
            Box::new(a.clone()),
            Box::new(b.clone()),
        )));
        assert!(suggestions.contains(&expected));
    }
}
