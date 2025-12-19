pub mod estimator;
pub mod factory;
pub mod history;
pub mod optimizer;
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
    /// ```
    /// use harp_core::backend::IdentityOptimizer;
    /// use harp_core::opt::graph::GraphOptimizer;
    ///
    /// let preparation_optimizer = IdentityOptimizer::new("prep");
    /// let lowering_optimizer = IdentityOptimizer::new("lower");
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
    /// ```
    /// use harp_core::backend::IdentityOptimizer;
    /// use harp_core::opt::graph::GraphOptimizer;
    ///
    /// let optimizer1 = IdentityOptimizer::new("opt1");
    /// let optimizer2 = IdentityOptimizer::new("opt2");
    /// let optimizer3 = IdentityOptimizer::new("opt3");
    ///
    /// // 名前付きでチェーン
    /// let chained = optimizer1
    ///     .with_name("Preparation")
    ///     .chain(optimizer2.with_name("Lowering"));
    ///
    /// // 名前なしでチェーン（自動命名）
    /// let optimizer1 = IdentityOptimizer::new("opt1");
    /// let optimizer2 = IdentityOptimizer::new("opt2");
    /// let optimizer3 = IdentityOptimizer::new("opt3");
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
    /// 提案の説明（どのような変換を行ったか）
    pub description: String,
}

impl SuggestResult {
    /// 新しいSuggestResultを作成（説明なし）
    pub fn new(graph: Graph, suggester_name: impl Into<String>) -> Self {
        Self {
            graph,
            suggester_name: suggester_name.into(),
            description: String::new(),
        }
    }

    /// 新しいSuggestResultを作成（説明付き）
    pub fn with_description(
        graph: Graph,
        suggester_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            graph,
            suggester_name: suggester_name.into(),
            description: description.into(),
        }
    }
}

/// 複数の書き換え候補を提案するトレイト（ビームサーチ用）
pub trait GraphSuggester {
    /// Suggesterの名前を返す
    fn name(&self) -> &'static str;

    /// 現在のグラフから書き換え可能な候補をすべて提案（説明付き）
    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult>;
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
pub use factory::{
    IdentityOptimizer, MultiPhaseConfig, SubgraphMode, create_ast_loop_suggester,
    create_fusion_suggester, create_graph_optimization_suggester,
    create_greedy_lowering_only_suggester, create_greedy_optimizer,
    create_kernel_partition_suggester, create_lowering_only_suggester,
    create_lowering_only_suggester_with_simd, create_multi_phase_optimizer,
    create_multi_phase_optimizer_with_selector, create_subgraph_inlining_suggester,
    create_view_merge_only_suggester, optimize_graph_greedy, optimize_graph_multi_phase,
};
pub use history::{AlternativeCandidate, OptimizationHistory, OptimizationSnapshot};
pub use optimizer::{
    BeamSearchGraphOptimizer, ChainedGraphOptimizer, GreedyGraphOptimizer, NamedOptimizer,
};
pub use selector::{GraphCostSelector, GraphMultiStageSelector, GraphSelector};
pub use suggesters::{
    BufferAbsorptionSuggester, CompositeSuggester, ContiguousInsertionSuggester, FusionSuggester,
    KernelMergeSuggester, KernelPartitionSuggester, LoweringSuggester, SubgraphInliningSuggester,
    TilingSuggester, ViewInsertionSuggester, ViewMergeSuggester,
};
