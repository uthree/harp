pub mod beam_search;
pub mod handcode;
pub mod rule_based_suggester;
pub mod unroll;

use crate::ast::AstNode;
use std::collections::HashSet;

/// A trait for suggesting rewrites to an AST.
pub trait RewriteSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode>;
}

/// A trait for estimating the cost of an AST.
pub trait CostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32;
}

/// Combines multiple `RewriteSuggester`s into one.
pub struct CombinedRewriteSuggester {
    suggesters: Vec<Box<dyn RewriteSuggester>>,
}

impl CombinedRewriteSuggester {
    pub fn new(suggesters: Vec<Box<dyn RewriteSuggester>>) -> Self {
        Self { suggesters }
    }
}

impl RewriteSuggester for CombinedRewriteSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut all_suggestions = HashSet::new();
        for suggester in &self.suggesters {
            let suggestions = suggester.suggest(node);
            all_suggestions.extend(suggestions);
        }
        all_suggestions.into_iter().collect()
    }
}

/// Combines multiple `CostEstimator`s into one.
pub struct CombinedCostEstimator {
    estimators: Vec<Box<dyn CostEstimator>>,
}

impl CombinedCostEstimator {
    pub fn new(estimators: Vec<Box<dyn CostEstimator>>) -> Self {
        Self { estimators }
    }
}

impl CostEstimator for CombinedCostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32 {
        self.estimators
            .iter()
            .map(|estimator| estimator.estimate_cost(ast))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, AstOp};

    struct MockSuggester {
        suggestions: Vec<AstNode>,
    }

    impl RewriteSuggester for MockSuggester {
        fn suggest(&self, _node: &AstNode) -> Vec<AstNode> {
            self.suggestions.clone()
        }
    }

    struct MockCostEstimator {
        cost: f32,
    }

    impl CostEstimator for MockCostEstimator {
        fn estimate_cost(&self, _ast: &AstNode) -> f32 {
            self.cost
        }
    }

    #[test]
    fn test_combined_rewrite_suggester() {
        let suggester1 = MockSuggester {
            suggestions: vec![AstNode::_new(AstOp::Add, vec![], crate::ast::DType::Any)],
        };
        let suggester2 = MockSuggester {
            suggestions: vec![AstNode::_new(AstOp::Sub, vec![], crate::ast::DType::Any)],
        };
        let combined =
            CombinedRewriteSuggester::new(vec![Box::new(suggester1), Box::new(suggester2)]);
        let suggestions = combined.suggest(&AstNode::_new(
            AstOp::Const(crate::ast::Const::Isize(0)),
            vec![],
            crate::ast::DType::Any,
        ));
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_combined_cost_estimator() {
        let estimator1 = MockCostEstimator { cost: 10.0 };
        let estimator2 = MockCostEstimator { cost: 5.0 };
        let combined = CombinedCostEstimator::new(vec![Box::new(estimator1), Box::new(estimator2)]);
        let cost = combined.estimate_cost(&AstNode::_new(
            AstOp::Const(crate::ast::Const::Isize(0)),
            vec![],
            crate::ast::DType::Any,
        ));
        assert_eq!(cost, 15.0);
    }
}
