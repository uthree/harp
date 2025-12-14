pub mod estimator;
pub mod history;
pub mod optimizer;
pub mod runtime_estimator;
pub mod selector;
pub mod suggesters;

use crate::graph::Graph;

/// グラフを最適化するトレイト
pub trait GraphOptimizer {
    /// オプティマイザの名前を返す（オプション）
    ///
    /// `with_name()`で名前を設定した場合はその名前を返します。
    /// デフォルトはNone（名前なし）です。
    fn name(&self) -> Option<&str> {
        None
    }

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

    /// オプティマイザに名前を付ける
    ///
    /// チェーン時にこの名前がフェーズ名として使用されます。
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chained = preparation_optimizer
    ///     .with_name("Preparation")
    ///     .chain(lowering_optimizer.with_name("Lowering"));
    /// ```
    fn with_name(self, name: impl Into<String>) -> optimizer::NamedOptimizer<Self>
    where
        Self: Sized,
    {
        optimizer::NamedOptimizer::new(self, name)
    }

    /// 他のオプティマイザとチェーンする
    ///
    /// 各オプティマイザの`name()`が設定されている場合はその名前を使用し、
    /// 設定されていない場合は "Phase 1", "Phase 2", ... と自動命名されます。
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 名前付きでチェーン
    /// let chained = optimizer1
    ///     .with_name("Preparation")
    ///     .chain(optimizer2.with_name("Lowering"));
    ///
    /// // 名前なしでチェーン（自動命名）
    /// let chained = optimizer1.chain(optimizer2).chain(optimizer3);
    /// ```
    fn chain<O: GraphOptimizer + 'static>(self, other: O) -> optimizer::ChainedGraphOptimizer
    where
        Self: Sized + 'static,
    {
        let self_name = self
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "Phase 1".to_string());
        let other_name = other
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "Phase 2".to_string());

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
pub use estimator::{
    AstBasedCostEstimator, KernelMergeCostEstimator, LoweringCostEstimator, SimpleCostEstimator,
};
pub use history::{OptimizationHistory, OptimizationSnapshot};
pub use optimizer::{BeamSearchGraphOptimizer, ChainedGraphOptimizer, NamedOptimizer};
pub use runtime_estimator::GraphRuntimeCostEstimator;
pub use selector::{GraphCostSelector, GraphRuntimeSelector, GraphSelector};
pub use suggesters::{
    BufferAbsorptionSuggester, CompositeSuggester, ContiguousInsertionSuggester, FusionSuggester,
    KernelMergeSuggester, LoweringSuggester, ProgramRootAbsorptionSuggester,
    ProgramRootBufferAbsorptionSuggester, TilingSuggester, ViewInsertionSuggester,
    ViewMergeSuggester,
};
