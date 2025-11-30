pub mod estimator;
pub mod history;
pub mod optimizer;
pub mod suggesters;

use crate::graph::Graph;

/// グラフを最適化するトレイト
pub trait GraphOptimizer {
    /// グラフを最適化して返す
    fn optimize(&self, graph: Graph) -> Graph;
}

/// Suggesterによる提案結果
#[derive(Clone, Debug)]
pub struct SuggestResult {
    /// 提案されたグラフ
    pub graph: Graph,
    /// 提案したSuggesterの名前
    pub suggester_name: String,
}

impl SuggestResult {
    /// 新しいSuggestResultを作成
    pub fn new(graph: Graph, suggester_name: impl Into<String>) -> Self {
        Self {
            graph,
            suggester_name: suggester_name.into(),
        }
    }
}

/// 複数の書き換え候補を提案するトレイト（ビームサーチ用）
pub trait GraphSuggester {
    /// Suggesterの名前を返す
    fn name(&self) -> &'static str;

    /// 現在のグラフから書き換え可能な候補をすべて提案
    fn suggest(&self, graph: &Graph) -> Vec<Graph>;

    /// 現在のグラフから書き換え可能な候補をSuggester名付きで提案
    fn suggest_named(&self, graph: &Graph) -> Vec<SuggestResult> {
        self.suggest(graph)
            .into_iter()
            .map(|g| SuggestResult::new(g, self.name()))
            .collect()
    }
}

/// グラフの実行コストを推定するトレイト
pub trait GraphCostEstimator {
    /// グラフの実行コストを推定
    fn estimate(&self, graph: &Graph) -> f32;
}

// Re-export commonly used types
pub use estimator::{AstBasedCostEstimator, KernelMergeCostEstimator, SimpleCostEstimator};
pub use history::{OptimizationHistory, OptimizationSnapshot};
pub use optimizer::BeamSearchGraphOptimizer;
pub use suggesters::{
    AstOptimizationSuggester, CompositeSuggester, ContiguousInsertionSuggester, FusionSuggester,
    KernelMergeSuggester, LoweringSuggester, SinkAbsorptionSuggester, TilingSuggester,
    ViewInsertionSuggester, ViewMergeSuggester,
};
