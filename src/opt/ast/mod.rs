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

/// Suggesterによる提案結果
#[derive(Clone, Debug)]
pub struct AstSuggestResult {
    /// 提案されたAST
    pub ast: AstNode,
    /// 提案したSuggesterの名前
    pub suggester_name: String,
    /// 提案の説明（どのような変換を行ったか）
    pub description: String,
}

impl AstSuggestResult {
    /// 新しいAstSuggestResultを作成（説明なし）
    pub fn new(ast: AstNode, suggester_name: impl Into<String>) -> Self {
        Self {
            ast,
            suggester_name: suggester_name.into(),
            description: String::new(),
        }
    }

    /// 新しいAstSuggestResultを作成（説明付き）
    pub fn with_description(
        ast: AstNode,
        suggester_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            ast,
            suggester_name: suggester_name.into(),
            description: description.into(),
        }
    }
}

/// 複数の書き換え候補を提案するトレイト（ビームサーチ用）
pub trait AstSuggester {
    /// Suggesterの名前を取得
    fn name(&self) -> &str;

    /// 現在のASTから書き換え可能な候補をすべて提案（説明付き）
    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult>;
}

/// ASTの実行コストを推定するトレイト
pub trait AstCostEstimator {
    /// ASTの実行コストを推定
    fn estimate(&self, ast: &AstNode) -> f32;
}

// Re-export commonly used types
pub use estimator::SimpleCostEstimator;
pub use history::{AlternativeCandidate, OptimizationHistory, OptimizationSnapshot};
// AstSuggestResult is already defined in this module
pub use optimizer::{BeamSearchOptimizer, RuleBaseOptimizer};
pub use selector::{AstCostSelector, AstMultiStageSelector, AstSelector};
pub use suggesters::{
    CompositeSuggester, CseSuggester, FunctionInliningSuggester, GlobalParallelizationSuggester,
    LocalParallelizationSuggester, LoopFusionSuggester, LoopInliningSuggester,
    LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseSuggester, VariableExpansionSuggester,
};
pub use transforms::{inline_small_loop, tile_loop};
