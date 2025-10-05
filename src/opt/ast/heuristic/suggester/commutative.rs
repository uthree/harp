use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester for commutative operations (Add, Mul, Max).
/// Suggests swapping the operands of commutative operations.
pub struct CommutativeSuggester;

impl RewriteSuggester for CommutativeSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Try swapping commutative operations at the current node
        match node {
            AstNode::Add(a, b) => {
                suggestions.push(AstNode::Add(b.clone(), a.clone()));
            }
            AstNode::Mul(a, b) => {
                suggestions.push(AstNode::Mul(b.clone(), a.clone()));
            }
            AstNode::Max(a, b) => {
                suggestions.push(AstNode::Max(b.clone(), a.clone()));
            }
            _ => {}
        }

        // Recursively suggest swaps in children
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
