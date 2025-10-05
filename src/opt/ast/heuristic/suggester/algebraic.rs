use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester for distributive and associative laws
pub struct AlgebraicLawSuggester;

impl RewriteSuggester for AlgebraicLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Distributive law: a * (b + c) <-> a*b + a*c
        if let AstNode::Mul(a, bc) = node {
            if let AstNode::Add(b, c) = &**bc {
                // a * (b + c) -> a*b + a*c
                suggestions
                    .push(AstNode::Mul(a.clone(), b.clone()) + AstNode::Mul(a.clone(), c.clone()));
            }
        }

        // Reverse distributive: a*b + a*c -> a * (b + c)
        if let AstNode::Add(left, right) = node {
            if let (AstNode::Mul(a1, b), AstNode::Mul(a2, c)) = (&**left, &**right) {
                if a1 == a2 {
                    // a*b + a*c -> a * (b + c)
                    suggestions.push(AstNode::Mul(
                        a1.clone(),
                        Box::new(AstNode::Add(b.clone(), c.clone())),
                    ));
                }
            }
        }

        // Associative law for addition: (a + b) + c <-> a + (b + c)
        if let AstNode::Add(ab, c) = node {
            if let AstNode::Add(a, b) = &**ab {
                // (a + b) + c -> a + (b + c)
                suggestions.push(AstNode::Add(
                    a.clone(),
                    Box::new(AstNode::Add(b.clone(), c.clone())),
                ));
            }
        }

        if let AstNode::Add(a, bc) = node {
            if let AstNode::Add(b, c) = &**bc {
                // a + (b + c) -> (a + b) + c
                suggestions.push(AstNode::Add(
                    Box::new(AstNode::Add(a.clone(), b.clone())),
                    c.clone(),
                ));
            }
        }

        // Associative law for multiplication: (a * b) * c <-> a * (b * c)
        if let AstNode::Mul(ab, c) = node {
            if let AstNode::Mul(a, b) = &**ab {
                // (a * b) * c -> a * (b * c)
                suggestions.push(AstNode::Mul(
                    a.clone(),
                    Box::new(AstNode::Mul(b.clone(), c.clone())),
                ));
            }
        }

        if let AstNode::Mul(a, bc) = node {
            if let AstNode::Mul(b, c) = &**bc {
                // a * (b * c) -> (a * b) * c
                suggestions.push(AstNode::Mul(
                    Box::new(AstNode::Mul(a.clone(), b.clone())),
                    c.clone(),
                ));
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
    fn test_algebraic_law_suggester() {
        let suggester = AlgebraicLawSuggester;

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
}
