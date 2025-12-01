pub mod estimator;
pub mod history;
pub mod optimizer;
pub mod suggesters;

use crate::graph::Graph;

/// グラフを最適化するトレイト
pub trait GraphOptimizer {
    /// グラフを最適化して返す
    fn optimize(&self, graph: Graph) -> Graph;

    /// グラフを最適化して、グラフと最適化履歴を返す
    ///
    /// デフォルト実装は `optimize` を呼び出して空の履歴を返します。
    /// 履歴を記録したい場合はこのメソッドをオーバーライドしてください。
    fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        let optimized = self.optimize(graph);
        (optimized, OptimizationHistory::new())
    }

    /// 他のオプティマイザとチェーンする
    ///
    /// フェーズ名は自動的に "Phase 1", "Phase 2", ... と命名されます。
    /// 名前を指定したい場合は `chain_named` を使用してください。
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chained = optimizer1.chain(optimizer2).chain(optimizer3);
    /// let result = chained.optimize(graph);
    /// ```
    fn chain<O: GraphOptimizer + 'static>(self, other: O) -> optimizer::ChainedGraphOptimizer
    where
        Self: Sized + 'static,
    {
        optimizer::ChainedGraphOptimizer::new()
            .add_phase("Phase 1", self)
            .add_phase("Phase 2", other)
    }

    /// 他のオプティマイザと名前付きでチェーンする
    ///
    /// # Arguments
    /// * `self_name` - 自身のフェーズ名
    /// * `other_name` - 追加するオプティマイザのフェーズ名
    /// * `other` - 追加するオプティマイザ
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chained = optimizer1
    ///     .chain_named("Lowering", "Fusion", optimizer2)
    ///     .chain_named("", "Finalize", optimizer3);  // 既存チェーンには空文字で
    /// ```
    fn chain_named<O: GraphOptimizer + 'static>(
        self,
        self_name: impl Into<String>,
        other_name: impl Into<String>,
        other: O,
    ) -> optimizer::ChainedGraphOptimizer
    where
        Self: Sized + 'static,
    {
        optimizer::ChainedGraphOptimizer::new()
            .add_phase(self_name, self)
            .add_phase(other_name, other)
    }
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
pub use optimizer::{BeamSearchGraphOptimizer, ChainedGraphOptimizer};
pub use suggesters::{
    AstOptimizationSuggester, BufferAbsorptionSuggester, CompositeSuggester,
    ContiguousInsertionSuggester, FusionSuggester, KernelMergeSuggester, LoweringSuggester,
    ProgramRootAbsorptionSuggester, ProgramRootBufferAbsorptionSuggester, TilingSuggester,
    ViewInsertionSuggester, ViewMergeSuggester,
};
