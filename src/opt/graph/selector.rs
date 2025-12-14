//! Graph最適化用のSelector
//!
//! ビームサーチでの候補選択を抽象化します。
//! 多段階のフィルタリングを可能にし、異なるコスト推定器を
//! 組み合わせて候補を絞り込むパイプラインを構築できます。
//!
//! # Example
//!
//! ```ignore
//! use harp::opt::graph::{GraphMultiStageSelector, SimpleCostEstimator};
//!
//! // 2段階の選択パイプライン
//! let selector = GraphMultiStageSelector::new()
//!     .then(SimpleCostEstimator::new(), 50)  // 第1段階: 50件に
//!     .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 10); // 第2段階: 10件に
//! ```

use std::cmp::Ordering;
use std::sync::Arc;

use crate::graph::Graph;

use super::{GraphCostEstimator, SimpleCostEstimator};

/// Graph最適化用のSelector trait
///
/// Graph最適化のビームサーチにおいて、候補の評価と選択を抽象化します。
/// 候補は`(Graph, String)`のタプルで、Stringは生成元のSuggester名です。
pub trait GraphSelector {
    /// 単一候補のコストを推定
    fn estimate(&self, candidate: &(Graph, String)) -> f32;

    /// 候補リストを評価し、上位n件を選択
    fn select(&self, candidates: Vec<(Graph, String)>, n: usize) -> Vec<((Graph, String), f32)>;
}

/// Graph用の静的コストベース選択器
///
/// GraphCostEstimatorを内包し、静的コストで候補をソートして上位n件を選択します。
/// Graph最適化のデフォルトの選択器として使用されます。
#[derive(Clone, Debug)]
pub struct GraphCostSelector<E = SimpleCostEstimator>
where
    E: GraphCostEstimator,
{
    estimator: E,
}

impl Default for GraphCostSelector<SimpleCostEstimator> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCostSelector<SimpleCostEstimator> {
    /// 新しいGraphCostSelectorを作成（デフォルトのSimpleCostEstimatorを使用）
    pub fn new() -> Self {
        Self {
            estimator: SimpleCostEstimator::new(),
        }
    }
}

impl<E> GraphCostSelector<E>
where
    E: GraphCostEstimator,
{
    /// カスタムのCostEstimatorでGraphCostSelectorを作成
    pub fn with_estimator(estimator: E) -> Self {
        Self { estimator }
    }

    /// 内部のCostEstimatorへの参照を取得
    pub fn estimator(&self) -> &E {
        &self.estimator
    }
}

impl<E> GraphSelector for GraphCostSelector<E>
where
    E: GraphCostEstimator,
{
    fn estimate(&self, candidate: &(Graph, String)) -> f32 {
        self.estimator.estimate(&candidate.0)
    }

    fn select(&self, candidates: Vec<(Graph, String)>, n: usize) -> Vec<((Graph, String), f32)> {
        let mut with_cost: Vec<((Graph, String), f32)> = candidates
            .into_iter()
            .map(|c| {
                let cost = self.estimator.estimate(&c.0);
                (c, cost)
            })
            .collect();
        with_cost.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        with_cost.into_iter().take(n).collect()
    }
}

/// 選択ステージ
///
/// 多段階選択の1ステップを表します。
struct SelectionStage {
    /// このステージで使用するコスト推定器
    estimator: Arc<dyn GraphCostEstimator + Send + Sync>,
    /// このステージで残す候補数
    keep: usize,
}

impl Clone for SelectionStage {
    fn clone(&self) -> Self {
        Self {
            estimator: Arc::clone(&self.estimator),
            keep: self.keep,
        }
    }
}

/// 多段階候補選択器
///
/// 複数のステージを順次適用し、各ステージで異なるコスト推定器を
/// 使用して候補を絞り込みます。
///
/// # Example
///
/// ```ignore
/// use harp::opt::graph::{GraphMultiStageSelector, SimpleCostEstimator};
///
/// // 2段階の選択パイプライン
/// let selector = GraphMultiStageSelector::new()
///     .then(SimpleCostEstimator::new(), 50)  // 第1段階: 50件に絞り込み
///     .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 10); // 第2段階: 10件に
/// ```
#[derive(Clone)]
pub struct GraphMultiStageSelector {
    stages: Vec<SelectionStage>,
}

impl Default for GraphMultiStageSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphMultiStageSelector {
    /// 新しい多段階セレクターを作成
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// コスト推定ステージを追加
    ///
    /// # Arguments
    /// * `estimator` - このステージで使用するコスト推定器
    /// * `keep` - このステージで残す候補数
    pub fn then<E: GraphCostEstimator + Send + Sync + 'static>(
        mut self,
        estimator: E,
        keep: usize,
    ) -> Self {
        self.stages.push(SelectionStage {
            estimator: Arc::new(estimator),
            keep,
        });
        self
    }

    /// ステージ数を取得
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl GraphSelector for GraphMultiStageSelector {
    fn estimate(&self, candidate: &(Graph, String)) -> f32 {
        // 最初のステージのestimatorでコストを推定
        self.stages
            .first()
            .map(|s| s.estimator.estimate(&candidate.0))
            .unwrap_or(0.0)
    }

    fn select(&self, candidates: Vec<(Graph, String)>, n: usize) -> Vec<((Graph, String), f32)> {
        if candidates.is_empty() {
            return vec![];
        }

        if self.stages.is_empty() {
            // ステージがない場合はそのまま返す
            return candidates.into_iter().take(n).map(|c| (c, 0.0)).collect();
        }

        let mut current: Vec<((Graph, String), f32)> =
            candidates.into_iter().map(|c| (c, 0.0)).collect();

        for (i, stage) in self.stages.iter().enumerate() {
            // コストを再計算
            for ((graph, _), cost) in current.iter_mut() {
                *cost = stage.estimator.estimate(graph);
            }

            // ソートして足切り
            current.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            // 最終ステージはnを使用、それ以外はstage.keepを使用
            let keep = if i == self.stages.len() - 1 {
                n.min(stage.keep)
            } else {
                stage.keep
            };
            current.truncate(keep);
        }

        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    #[test]
    fn test_graph_cost_selector_basic() {
        let selector = GraphCostSelector::new();

        // 空のグラフを複数作成（コストはノード数に基づく）
        let mut graph1 = Graph::new();
        let a1 = graph1.input("a", DType::F32, vec![10, 10]);
        let b1 = graph1.input("b", DType::F32, vec![10, 10]);
        let _ = a1 + b1;

        let mut graph2 = Graph::new();
        let a2 = graph2.input("a", DType::F32, vec![10, 10]);
        let _ = a2.clone() + a2; // 同じノードを再利用

        let candidates = vec![
            (graph1, "suggester1".to_string()),
            (graph2, "suggester2".to_string()),
        ];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_graph_cost_selector_empty_candidates() {
        let selector = GraphCostSelector::new();
        let candidates: Vec<(Graph, String)> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_multi_stage_selector_single_stage() {
        let selector = GraphMultiStageSelector::new().then(SimpleCostEstimator::new(), 3);

        let mut graph1 = Graph::new();
        let a1 = graph1.input("a", DType::F32, vec![10, 10]);
        let _ = a1.clone() + a1;

        let mut graph2 = Graph::new();
        let a2 = graph2.input("a", DType::F32, vec![10, 10]);
        let b2 = graph2.input("b", DType::F32, vec![10, 10]);
        let _ = a2 + b2;

        let candidates = vec![
            (graph1.clone(), "s1".to_string()),
            (graph2.clone(), "s2".to_string()),
            (graph1.clone(), "s3".to_string()),
            (graph2.clone(), "s4".to_string()),
            (graph1.clone(), "s5".to_string()),
        ];

        let selected = selector.select(candidates, 3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_multi_stage_selector_two_stages() {
        let selector = GraphMultiStageSelector::new()
            .then(SimpleCostEstimator::new(), 4)
            .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 2);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 10]);
        let _ = a.clone() + a;

        let candidates = vec![
            (graph.clone(), "s1".to_string()),
            (graph.clone(), "s2".to_string()),
            (graph.clone(), "s3".to_string()),
            (graph.clone(), "s4".to_string()),
            (graph.clone(), "s5".to_string()),
        ];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_multi_stage_selector_respects_final_k() {
        let selector = GraphMultiStageSelector::new()
            .then(SimpleCostEstimator::new(), 10)
            .then(SimpleCostEstimator::new(), 10);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 10]);
        let _ = a.clone() + a;

        let candidates = vec![
            (graph.clone(), "s1".to_string()),
            (graph.clone(), "s2".to_string()),
            (graph.clone(), "s3".to_string()),
        ];

        // kが3より小さい場合
        let selected = selector.select(candidates, 1);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_multi_stage_selector_empty_candidates() {
        let selector = GraphMultiStageSelector::new().then(SimpleCostEstimator::new(), 10);

        let candidates: Vec<(Graph, String)> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_multi_stage_selector_stage_count() {
        let selector = GraphMultiStageSelector::new()
            .then(SimpleCostEstimator::new(), 100)
            .then(SimpleCostEstimator::new(), 50)
            .then(SimpleCostEstimator::new(), 10);

        assert_eq!(selector.stage_count(), 3);
    }
}
