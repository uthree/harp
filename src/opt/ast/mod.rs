pub mod estimator;
pub mod history;
pub mod optimizer;
pub mod rules;
pub mod selector;
pub mod suggesters;
pub mod transforms;

use crate::ast::AstNode;

/// ASTを最適化するトレイト
pub trait AstOptimizer {
    /// ASTを最適化して返す
    fn optimize(&self, ast: AstNode) -> AstNode;
}

/// 複数の書き換え候補を提案するトレイト（ビームサーチ用）
pub trait AstSuggester {
    /// Suggesterの名前を取得
    fn name(&self) -> &str;

    /// 現在のASTから書き換え可能な候補をすべて提案
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode>;

    /// 候補とともにSuggester名を返す（デフォルト実装）
    fn suggest_named(&self, ast: &AstNode) -> Vec<(AstNode, String)> {
        let name = self.name().to_string();
        self.suggest(ast)
            .into_iter()
            .map(|ast| (ast, name.clone()))
            .collect()
    }
}

/// ASTの実行コストを推定するトレイト
pub trait AstCostEstimator {
    /// ASTの実行コストを推定
    fn estimate(&self, ast: &AstNode) -> f32;
}

// Re-export commonly used types
pub use estimator::SimpleCostEstimator;
pub use history::{AlternativeCandidate, OptimizationHistory, OptimizationSnapshot};
pub use optimizer::{BeamSearchOptimizer, RuleBaseOptimizer};
pub use selector::{AstCostSelector, AstMultiStageSelector, AstSelector};
pub use suggesters::{
    CompositeSuggester, CseSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseSuggester,
    VariableExpansionSuggester,
};
pub use transforms::{inline_small_loop, tile_loop};
