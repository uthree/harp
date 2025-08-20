use crate::ast::AstNode;
use crate::ast::pattern::AstRewriter;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A simple rewrite suggester that holds a predefined AstRewriter.
#[derive(Clone)]
pub struct RuleBasedRewriteSuggester {
    rewriter: AstRewriter,
}

impl RuleBasedRewriteSuggester {
    /// Creates a new suggester from an existing AstRewriter.
    pub fn new(rewriter: AstRewriter) -> Self {
        Self { rewriter }
    }
}

impl RewriteSuggester for RuleBasedRewriteSuggester {
    /// Returns the underlying AstRewriter.
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        self.rewriter.get_possible_rewrites(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{AstNode, DType},
        ast_rewriter, astpat,
    };

    #[test]
    #[allow(unused_variables)]
    fn test_rule_based_suggester() {
        // 1. Create some rules and an AstRewriter
        let add_zero_rule = astpat!(|a, b| a + b, if *b == AstNode::from(0isize) => a.clone());
        let mul_one_rule = astpat!(|a, b| a * b, if *b == AstNode::from(1isize) => a.clone());
        let rewriter = ast_rewriter!("Optimizer", add_zero_rule, mul_one_rule);

        // 2. Create the suggester
        let suggester = RuleBasedRewriteSuggester::new(rewriter);

        // 3. Get the suggested rewrites
        let a = AstNode::var("a", DType::Isize);
        let expr = (a.clone() + AstNode::from(0isize)) * AstNode::from(1isize);
        let suggestions = suggester.suggest(&expr);

        // 4. Verify the suggestions
        let expected1 = a.clone() * AstNode::from(1isize); // from a + 0 => a
        let expected2 = a.clone() + AstNode::from(0isize); // from (a + 0) * 1 => a + 0
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.iter().any(|r| *r == expected1));
        assert!(suggestions.iter().any(|r| *r == expected2));
    }
}
