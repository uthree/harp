use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester for factorization and expansion
/// (a + b) * (a - b) <-> a*a - b*b
pub struct FactorizationSuggester;

impl RewriteSuggester for FactorizationSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // (a + b) * (a - b) -> a*a - b*b
        if let AstNode::Mul(left, right) = node {
            // Check if left is (a + b) and right is (a - b)
            if let (AstNode::Add(a1, b1), AstNode::Add(a2, b2_neg)) = (&**left, &**right) {
                if let AstNode::Neg(b2) = &**b2_neg {
                    if a1 == a2 && b1 == b2 {
                        // (a + b) * (a - b) -> a*a - b*b
                        suggestions.push(
                            AstNode::Mul(a1.clone(), a1.clone())
                                + AstNode::Neg(Box::new(AstNode::Mul(b1.clone(), b1.clone()))),
                        );
                    }
                }
            }
        }

        // a*a - b*b -> (a + b) * (a - b)
        if let AstNode::Add(left, right) = node {
            if let AstNode::Neg(b_squared) = &**right {
                if let (AstNode::Mul(a1, a2), AstNode::Mul(b1, b2)) = (&**left, &**b_squared) {
                    if a1 == a2 && b1 == b2 {
                        // a*a - b*b -> (a + b) * (a - b)
                        suggestions.push(AstNode::Mul(
                            Box::new(AstNode::Add(a1.clone(), b1.clone())),
                            Box::new(AstNode::Add(a1.clone(), Box::new(AstNode::Neg(b1.clone())))),
                        ));
                    }
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
    fn test_factorization_suggester() {
        let suggester = FactorizationSuggester;

        // (a + b) * (a - b) -> a*a - b*b
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Mul(
            Box::new(AstNode::Add(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(AstNode::Add(
                Box::new(a.clone()),
                Box::new(AstNode::Neg(Box::new(b.clone()))),
            )),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(a.clone()))),
            Box::new(AstNode::Neg(Box::new(AstNode::Mul(
                Box::new(b.clone()),
                Box::new(b.clone()),
            )))),
        );
        assert!(suggestions.contains(&expected));
    }
}
