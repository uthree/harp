use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester that removes inverse operations like log2(exp2(x)) -> x
pub struct InverseOperationSuggester;

impl RewriteSuggester for InverseOperationSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // log2(exp2(x)) -> x
        if let AstNode::Log2(inner) = node {
            if let AstNode::Exp2(x) = &**inner {
                suggestions.push((**x).clone());
            }
        }

        // exp2(log2(x)) -> x
        if let AstNode::Exp2(inner) = node {
            if let AstNode::Log2(x) = &**inner {
                suggestions.push((**x).clone());
            }
        }

        // sqrt(x * x) -> x (assuming x >= 0)
        if let AstNode::Sqrt(inner) = node {
            if let AstNode::Mul(a, b) = &**inner {
                if a == b {
                    suggestions.push((**a).clone());
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

    #[test]
    fn test_inverse_operation_suggester() {
        let suggester = InverseOperationSuggester;

        // log2(exp2(x)) -> x
        let x = AstNode::Var("x".to_string());
        let ast = AstNode::Log2(Box::new(AstNode::Exp2(Box::new(x.clone()))));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&x));

        // exp2(log2(x)) -> x
        let ast = AstNode::Exp2(Box::new(AstNode::Log2(Box::new(x.clone()))));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&x));

        // sqrt(x * x) -> x
        let ast = AstNode::Sqrt(Box::new(AstNode::Mul(
            Box::new(x.clone()),
            Box::new(x.clone()),
        )));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&x));
    }
}
