use crate::ast::AstNode;

pub mod cost_estimator;
pub mod suggester;

pub use cost_estimator::{NodeCountCostEstimator, OperationCostEstimator};
pub use suggester::{
    AlgebraicLawSuggester, CommutativeSuggester, FactorizationSuggester, InverseOperationSuggester,
    LoopTransformSuggester, RedundancyRemovalSuggester, RuleBasedSuggester,
};

/// A trait for suggesting rewrites to an AST.
pub trait RewriteSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode>;
}

/// A trait for estimating the cost of an AST.
pub trait CostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32;
}

/// Combines multiple `RewriteSuggester`s into one.
#[allow(dead_code)]
pub struct CombinedRewriteSuggester {
    suggesters: Vec<Box<dyn RewriteSuggester>>,
}

impl CombinedRewriteSuggester {
    #[allow(dead_code)]
    pub fn new(suggesters: Vec<Box<dyn RewriteSuggester>>) -> Self {
        Self { suggesters }
    }
}

impl RewriteSuggester for CombinedRewriteSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut all_suggestions = Vec::new();
        for suggester in &self.suggesters {
            all_suggestions.extend(suggester.suggest(node));
        }
        all_suggestions
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
