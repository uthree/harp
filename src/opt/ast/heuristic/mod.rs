use crate::ast::{AstNode, pattern::AstRewriter};
pub mod beam_search;
pub mod handcode;
pub mod rule_based;

// ASTの実行時のコストを評価する機能
// TODO: 動的Shape変数を正しく処理できるようにする
pub trait CostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32;
}

// 複数のコスト推定の重み付き和
pub struct CombinedCostEstimator {
    weights: Vec<f32>,
    estimators: Vec<Box<dyn CostEstimator>>,
}

impl CombinedCostEstimator {
    pub fn new(estimators: Vec<(f32, Box<dyn CostEstimator>)>) -> Self {
        let (weights, estimators) = estimators.into_iter().unzip();
        Self {
            weights,
            estimators,
        }
    }
}

impl CostEstimator for CombinedCostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32 {
        self.estimators
            .iter()
            .zip(&self.weights)
            .map(|(estimator, weight)| estimator.estimate_cost(ast) * weight)
            .sum()
    }
}

// ASTの書き換えの候補を提案する機能
pub trait RewriteSuggester {
    fn suggest(&self) -> AstRewriter;
}

// 複数の候補提案をまとめる
pub struct CombinedRewriteSuggester {
    suggesters: Vec<Box<dyn RewriteSuggester>>,
}

impl CombinedRewriteSuggester {
    pub fn new(suggesters: Vec<Box<dyn RewriteSuggester>>) -> Self {
        Self { suggesters }
    }
}

impl RewriteSuggester for CombinedRewriteSuggester {
    fn suggest(&self) -> AstRewriter {
        self.suggesters
            .iter()
            .map(|s| s.suggest())
            .reduce(|a, b| a + b)
            .unwrap_or_else(|| AstRewriter::with_rules("Empty", vec![]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{AstNode, pattern::RewriteRule},
        ast_rewriter, astpat,
    };

    // Mock CostEstimator for testing
    struct MockCostEstimator {
        cost: f32,
    }
    impl CostEstimator for MockCostEstimator {
        fn estimate_cost(&self, _ast: &AstNode) -> f32 {
            self.cost
        }
    }

    #[test]
    fn test_combined_cost_estimator() {
        let estimator1 = MockCostEstimator { cost: 10.0 };
        let estimator2 = MockCostEstimator { cost: 5.0 };
        let combined = CombinedCostEstimator::new(vec![
            (0.5, Box::new(estimator1)),
            (2.0, Box::new(estimator2)),
        ]);
        let dummy_ast = AstNode::from(0isize);
        let total_cost = combined.estimate_cost(&dummy_ast);
        assert_eq!(total_cost, 10.0 * 0.5 + 5.0 * 2.0);
    }

    // Mock RewriteSuggester for testing
    struct MockRewriteSuggester {
        rewriter: AstRewriter,
    }
    impl RewriteSuggester for MockRewriteSuggester {
        fn suggest(&self) -> AstRewriter {
            self.rewriter.clone()
        }
    }

    #[test]
    fn test_combined_rewrite_suggester() {
        let rule1 = astpat!(|a| a + 0isize => a);
        let rewriter1 = ast_rewriter!("AddZero", rule1);
        let suggester1 = MockRewriteSuggester {
            rewriter: rewriter1,
        };

        let rule2 = astpat!(|a| a * 1isize => a);
        let rewriter2 = ast_rewriter!("MulOne", rule2);
        let suggester2 = MockRewriteSuggester {
            rewriter: rewriter2,
        };

        let combined =
            CombinedRewriteSuggester::new(vec![Box::new(suggester1), Box::new(suggester2)]);
        let combined_rewriter = combined.suggest();

        // Check that the combined rewriter can apply both rules
        let expr1 = AstNode::from(5isize) + AstNode::from(0isize);
        assert_eq!(combined_rewriter.apply(&expr1), AstNode::from(5isize));

        let expr2 = AstNode::from(5isize) * AstNode::from(1isize);
        assert_eq!(combined_rewriter.apply(&expr2), AstNode::from(5isize));
    }
}
