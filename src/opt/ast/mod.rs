pub mod estimator;
pub mod history;
pub mod optimizer;
pub mod rules;
pub mod suggesters;
pub mod transforms;

use crate::ast::AstNode;

/// ASTを最適化するトレイト
pub trait Optimizer {
    /// ASTを最適化して返す
    fn optimize(&self, ast: AstNode) -> AstNode;
}

/// 複数の書き換え候補を提案するトレイト（ビームサーチ用）
pub trait Suggester {
    /// 現在のASTから書き換え可能な候補をすべて提案
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode>;
}

/// ASTの実行コストを推定するトレイト
pub trait CostEstimator {
    /// ASTの実行コストを推定
    fn estimate(&self, ast: &AstNode) -> f32;
}

// Re-export commonly used types
pub use estimator::SimpleCostEstimator;
pub use history::{OptimizationHistory, OptimizationSnapshot};
pub use optimizer::{BeamSearchOptimizer, RuleBaseOptimizer};
pub use suggesters::{
    CompositeSuggester, CseSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseSuggester,
};
pub use transforms::{inline_small_loop, tile_loop};
