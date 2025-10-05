use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester that proposes removing redundant operations.
/// For example, suggests removing a Store operation that is never read.
pub struct RedundancyRemovalSuggester;

impl RewriteSuggester for RedundancyRemovalSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // For Block nodes, suggest removing each statement
        if let AstNode::Block { scope, statements } = node {
            for i in 0..statements.len() {
                let mut new_statements = statements.clone();
                new_statements.remove(i);
                suggestions.push(AstNode::Block {
                    scope: scope.clone(),
                    statements: new_statements,
                });
            }
        }

        // Recursively suggest removals in children
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
