use crate::ast::AstNode;

pub mod cost_estimator;
pub mod optimizer;
pub mod suggester;

pub use cost_estimator::{NodeCountCostEstimator, OperationCostEstimator};
pub use optimizer::{BeamSearchOptimizer, CostBasedOptimizer};
pub use suggester::{
    AlgebraicLawSuggester, BitwiseLawSuggester, CommutativeSuggester, FactorizationSuggester,
    InverseOperationSuggester, LogExpLawSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
    LoopTransformSuggester, LoopUnrollSuggester, MaxLawSuggester, ReciprocalLawSuggester,
    RuleBasedSuggester, SqrtLawSuggester, UnrollHintSuggester,
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

/// Creates a combined suggester with all available suggesters
pub fn all_suggesters() -> CombinedRewriteSuggester {
    CombinedRewriteSuggester::new(vec![
        Box::new(AlgebraicLawSuggester),
        Box::new(BitwiseLawSuggester::new()),
        Box::new(CommutativeSuggester),
        Box::new(FactorizationSuggester),
        Box::new(InverseOperationSuggester),
        Box::new(LogExpLawSuggester),
        Box::new(MaxLawSuggester),
        Box::new(ReciprocalLawSuggester),
        Box::new(SqrtLawSuggester),
        // LoopInterchangeSuggester: swaps nested loop order for better cache locality
        // Only applies to simple nested loops loop(loop(body)), not loop(body, loop(body))
        Box::new(LoopInterchangeSuggester),
        // LoopTilingSuggester: try multiple tile sizes (powers of 2 from 2^1 to 2^8)
        Box::new(LoopTilingSuggester::new(2)),
        Box::new(LoopTilingSuggester::new(4)),
        Box::new(LoopTilingSuggester::new(8)),
        Box::new(LoopTilingSuggester::new(16)),
        Box::new(LoopTilingSuggester::new(32)),
        Box::new(LoopTilingSuggester::new(64)),
        Box::new(LoopTilingSuggester::new(128)),
        Box::new(LoopTilingSuggester::new(256)),
        Box::new(LoopTransformSuggester),
        // LoopUnrollSuggester: fully unrolls loops with constant iteration counts
        Box::new(LoopUnrollSuggester::new()),
    ])
}
