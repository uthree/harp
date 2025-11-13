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

/// 複数の書き換え候補を提案するトレイト（ビームサーチ用）
pub trait GraphSuggester {
    /// 現在のグラフから書き換え可能な候補をすべて提案
    fn suggest(&self, graph: &Graph) -> Vec<Graph>;
}

/// グラフの実行コストを推定するトレイト
pub trait GraphCostEstimator {
    /// グラフの実行コストを推定
    fn estimate(&self, graph: &Graph) -> f32;
}

// Re-export commonly used types
pub use estimator::{AstBasedCostEstimator, SimpleCostEstimator};
pub use history::{OptimizationHistory, OptimizationSnapshot};
pub use optimizer::BeamSearchGraphOptimizer;
pub use suggesters::{
    CompositeSuggester, FusionSuggester, ParallelStrategyChanger, SimdSuggester, TilingSuggester,
    ViewInsertionSuggester,
};
