use crate::ast::AstNode;
use crate::ast_pattern;
use crate::opt::ast::heuristic::RewriteSuggester;
use std::rc::Rc;

/// A suggester for commutative operations (Add, Mul, Max).
/// Suggests swapping the operands of commutative operations.
pub struct CommutativeSuggester {
    rules: Vec<Rc<crate::ast::pattern::AstRewriteRule>>,
}

impl CommutativeSuggester {
    pub fn new() -> Self {
        let rules = vec![
            // a + b -> b + a
            ast_pattern!(|a, b| a.clone() + b.clone() => b.clone() + a.clone()),
            // a * b -> b * a
            ast_pattern!(|a, b| a.clone() * b.clone() => b.clone() * a.clone()),
            // Max(a, b) -> Max(b, a)
            ast_pattern!(|a, b| AstNode::Max(Box::new(a.clone()), Box::new(b.clone())) => AstNode::Max(Box::new(b.clone()), Box::new(a.clone()))),
        ];

        Self { rules }
    }
}

impl Default for CommutativeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteSuggester for CommutativeSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();
        for rule in &self.rules {
            suggestions.extend(rule.get_possible_rewrites(node));
        }
        suggestions
    }
}
