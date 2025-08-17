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
    fn suggest(&self) -> AstRewriter {
        self.rewriter.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{AstNode, pattern::RewriteRule},
        ast_rewriter, astpat,
    };

    #[test]
    fn test_rule_based_suggester() {
        // 1. Create some rules and an AstRewriter
        let add_zero_rule = astpat!(|a, b| a + b, if b == AstNode::from(0isize) => a);
        let mul_one_rule = astpat!(|a, b| a * b, if b == AstNode::from(1isize) => a);
        let rewriter = ast_rewriter!("Optimizer", add_zero_rule, mul_one_rule);

        // 2. Create the suggester
        let suggester = RuleBasedRewriteSuggester::new(rewriter);

        // 3. Get the suggested rewriter
        let suggested_rewriter = suggester.suggest();

        // 4. Verify that the suggested rewriter works
        let expr1 = AstNode::from(5isize) + AstNode::from(0isize);
        assert_eq!(suggested_rewriter.apply(&expr1), AstNode::from(5isize));

        let expr2 = AstNode::from(5isize) * AstNode::from(1isize);
        assert_eq!(suggested_rewriter.apply(&expr2), AstNode::from(5isize));

        let expr3 = AstNode::from(5isize) + AstNode::from(1isize); // Should not change
        assert_eq!(suggested_rewriter.apply(&expr3), expr3.clone());
    }
}
