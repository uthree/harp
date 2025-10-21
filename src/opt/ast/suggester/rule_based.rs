use crate::ast::{pattern::AstRewriteRule, AstNode};
use crate::opt::ast::RewriteSuggester;
use std::rc::Rc;

/// A suggester that uses rewrite rules to propose alternative ASTs.
#[derive(Clone)]
pub struct RuleBasedSuggester {
    rules: Vec<Rc<AstRewriteRule>>,
}

impl RuleBasedSuggester {
    pub fn new(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self { rules }
    }
}

impl RewriteSuggester for RuleBasedSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();
        for rule in &self.rules {
            suggestions.extend(rule.get_possible_rewrites(node));
        }
        suggestions
    }
}
