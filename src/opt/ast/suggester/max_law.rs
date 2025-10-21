use crate::ast::AstNode;
use crate::opt::ast::RewriteSuggester;

/// A suggester for max operation laws
///
/// Laws:
/// - max(a, a) = a (idempotent)
/// - max(a, max(b, c)) = max(max(a, b), c) (associative)
/// - max(max(a, b), c) = max(a, max(b, c)) (associative)
pub struct MaxLawSuggester;

impl RewriteSuggester for MaxLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // max(a, a) -> a (idempotent)
        if let AstNode::Max(a, b) = node {
            if a == b {
                suggestions.push((**a).clone());
            }
        }

        // max(a, max(b, c)) -> max(max(a, b), c) (associative)
        if let AstNode::Max(a, bc) = node {
            if let AstNode::Max(b, c) = &**bc {
                suggestions.push(AstNode::Max(
                    Box::new(AstNode::Max(a.clone(), b.clone())),
                    c.clone(),
                ));
            }
        }

        // max(max(a, b), c) -> max(a, max(b, c)) (associative)
        if let AstNode::Max(ab, c) = node {
            if let AstNode::Max(a, b) = &**ab {
                suggestions.push(AstNode::Max(
                    a.clone(),
                    Box::new(AstNode::Max(b.clone(), c.clone())),
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
    fn test_max_idempotent() {
        let suggester = MaxLawSuggester;

        // max(a, a) -> a
        let a = var("a");
        let ast = AstNode::Max(Box::new(a.clone()), Box::new(a.clone()));

        let suggestions = suggester.suggest(&ast);

        assert!(suggestions.contains(&a));
    }

    #[test]
    fn test_max_associative_left_to_right() {
        let suggester = MaxLawSuggester;

        // max(max(a, b), c) -> max(a, max(b, c))
        let a = var("a");
        let b = var("b");
        let c = var("c");
        let ast = AstNode::Max(
            Box::new(AstNode::Max(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(c.clone()),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Max(
            Box::new(a.clone()),
            Box::new(AstNode::Max(Box::new(b.clone()), Box::new(c.clone()))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_max_associative_right_to_left() {
        let suggester = MaxLawSuggester;

        // max(a, max(b, c)) -> max(max(a, b), c)
        let a = var("a");
        let b = var("b");
        let c = var("c");
        let ast = AstNode::Max(
            Box::new(a.clone()),
            Box::new(AstNode::Max(Box::new(b.clone()), Box::new(c.clone()))),
        );

        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Max(
            Box::new(AstNode::Max(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(c.clone()),
        );
        assert!(suggestions.contains(&expected));
    }
}
