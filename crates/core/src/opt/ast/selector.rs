//! AST最適化用のSelector
//!
//! ビームサーチでの候補選択を抽象化します。
//! 多段階のフィルタリングを可能にし、異なるコスト推定器を
//! 組み合わせて候補を絞り込むパイプラインを構築できます。
//!
//! # Example
//!
//! ```
//! use harp::opt::ast::{AstMultiStageSelector, SimpleCostEstimator};
//!
//! // 2段階の選択パイプライン
//! let selector = AstMultiStageSelector::new()
//!     .then(SimpleCostEstimator::new(), 100)  // 第1段階: 100件に
//!     .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 10); // 第2段階: 10件に
//! ```

use std::cmp::Ordering;
use std::sync::Arc;

use crate::ast::AstNode;

use super::{AstCostEstimator, SimpleCostEstimator};

/// AST最適化用のSelector trait
///
/// AST最適化のビームサーチにおいて、候補の評価と選択を抽象化します。
pub trait AstSelector {
    /// 単一候補のコストを推定
    fn estimate(&self, candidate: &AstNode) -> f32;

    /// 候補リストを評価し、上位n件を選択
    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)>;

    /// 候補リストを評価し、上位n件を選択（元のインデックスを保持）
    ///
    /// 返り値は (AST, コスト, 元のインデックス) のタプル
    fn select_with_indices(&self, candidates: Vec<AstNode>, n: usize)
    -> Vec<(AstNode, f32, usize)>;
}

/// AST用の静的コストベース選択器
///
/// CostEstimatorを内包し、静的コストで候補をソートして上位n件を選択します。
/// AST最適化のデフォルトの選択器として使用されます。
#[derive(Clone, Debug)]
pub struct AstCostSelector<E = SimpleCostEstimator>
where
    E: AstCostEstimator,
{
    estimator: E,
}

impl Default for AstCostSelector<SimpleCostEstimator> {
    fn default() -> Self {
        Self::new()
    }
}

impl AstCostSelector<SimpleCostEstimator> {
    /// 新しいAstCostSelectorを作成（デフォルトのSimpleCostEstimatorを使用）
    pub fn new() -> Self {
        Self {
            estimator: SimpleCostEstimator::new(),
        }
    }
}

impl<E> AstCostSelector<E>
where
    E: AstCostEstimator,
{
    /// カスタムのCostEstimatorでAstCostSelectorを作成
    pub fn with_estimator(estimator: E) -> Self {
        Self { estimator }
    }

    /// 内部のCostEstimatorへの参照を取得
    pub fn estimator(&self) -> &E {
        &self.estimator
    }
}

impl<E> AstSelector for AstCostSelector<E>
where
    E: AstCostEstimator,
{
    fn estimate(&self, candidate: &AstNode) -> f32 {
        self.estimator.estimate(candidate)
    }

    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)> {
        self.select_with_indices(candidates, n)
            .into_iter()
            .map(|(ast, cost, _)| (ast, cost))
            .collect()
    }

    fn select_with_indices(
        &self,
        candidates: Vec<AstNode>,
        n: usize,
    ) -> Vec<(AstNode, f32, usize)> {
        let mut with_cost_and_index: Vec<(AstNode, f32, usize)> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, c)| {
                let cost = self.estimator.estimate(&c);
                (c, cost, idx)
            })
            .collect();
        with_cost_and_index.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        with_cost_and_index.into_iter().take(n).collect()
    }
}

/// 選択ステージ
///
/// 多段階選択の1ステップを表します。
struct SelectionStage {
    /// このステージで使用するコスト推定器
    estimator: Arc<dyn AstCostEstimator + Send + Sync>,
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
/// ```
/// use harp::opt::ast::{AstMultiStageSelector, SimpleCostEstimator};
///
/// // 2段階の選択パイプライン
/// let selector = AstMultiStageSelector::new()
///     .then(SimpleCostEstimator::new(), 100)  // 第1段階: 100件に絞り込み
///     .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 10); // 第2段階: 10件に
/// ```
#[derive(Clone)]
pub struct AstMultiStageSelector {
    stages: Vec<SelectionStage>,
}

impl Default for AstMultiStageSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl AstMultiStageSelector {
    /// 新しい多段階セレクターを作成
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// コスト推定ステージを追加
    ///
    /// # Arguments
    /// * `estimator` - このステージで使用するコスト推定器
    /// * `keep` - このステージで残す候補数
    pub fn then<E: AstCostEstimator + Send + Sync + 'static>(
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

impl AstSelector for AstMultiStageSelector {
    fn estimate(&self, candidate: &AstNode) -> f32 {
        // 最初のステージのestimatorでコストを推定
        self.stages
            .first()
            .map(|s| s.estimator.estimate(candidate))
            .unwrap_or(0.0)
    }

    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)> {
        self.select_with_indices(candidates, n)
            .into_iter()
            .map(|(ast, cost, _)| (ast, cost))
            .collect()
    }

    fn select_with_indices(
        &self,
        candidates: Vec<AstNode>,
        n: usize,
    ) -> Vec<(AstNode, f32, usize)> {
        if candidates.is_empty() {
            return vec![];
        }

        if self.stages.is_empty() {
            // ステージがない場合はそのまま返す
            return candidates
                .into_iter()
                .enumerate()
                .take(n)
                .map(|(idx, c)| (c, 0.0, idx))
                .collect();
        }

        let mut current: Vec<(AstNode, f32, usize)> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, c)| (c, 0.0, idx))
            .collect();

        for (i, stage) in self.stages.iter().enumerate() {
            // コストを再計算
            for (ast, cost, _) in current.iter_mut() {
                *cost = stage.estimator.estimate(ast);
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
    use crate::ast::Literal;

    #[test]
    fn test_ast_cost_selector_basic() {
        let selector = AstCostSelector::new();

        // シンプルなASTノードを作成
        let ast1 = AstNode::Const(Literal::I64(42));
        let ast2 = AstNode::Const(Literal::F32(2.5));

        let candidates = vec![ast1, ast2];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_ast_cost_selector_empty_candidates() {
        let selector = AstCostSelector::new();
        let candidates: Vec<AstNode> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_ast_cost_selector_limit() {
        let selector = AstCostSelector::new();

        let candidates = vec![
            AstNode::Const(Literal::I64(1)),
            AstNode::Const(Literal::I64(2)),
            AstNode::Const(Literal::I64(3)),
            AstNode::Const(Literal::I64(4)),
            AstNode::Const(Literal::I64(5)),
        ];

        let selected = selector.select(candidates, 3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_multi_stage_selector_single_stage() {
        let selector = AstMultiStageSelector::new().then(SimpleCostEstimator::new(), 3);

        let candidates = vec![
            AstNode::Const(Literal::I64(1)),
            AstNode::Const(Literal::I64(2)),
            AstNode::Const(Literal::I64(3)),
            AstNode::Const(Literal::I64(4)),
            AstNode::Const(Literal::I64(5)),
        ];

        let selected = selector.select(candidates, 3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_multi_stage_selector_two_stages() {
        let selector = AstMultiStageSelector::new()
            .then(SimpleCostEstimator::new(), 4)
            .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 2);

        let candidates = vec![
            AstNode::Const(Literal::I64(1)),
            AstNode::Const(Literal::I64(2)),
            AstNode::Const(Literal::I64(3)),
            AstNode::Const(Literal::I64(4)),
            AstNode::Const(Literal::I64(5)),
        ];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_multi_stage_selector_respects_final_k() {
        let selector = AstMultiStageSelector::new()
            .then(SimpleCostEstimator::new(), 10)
            .then(SimpleCostEstimator::new(), 10);

        let candidates = vec![
            AstNode::Const(Literal::I64(1)),
            AstNode::Const(Literal::I64(2)),
            AstNode::Const(Literal::I64(3)),
        ];

        // kが3より小さい場合
        let selected = selector.select(candidates, 1);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_multi_stage_selector_empty_candidates() {
        let selector = AstMultiStageSelector::new().then(SimpleCostEstimator::new(), 10);

        let candidates: Vec<AstNode> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_multi_stage_selector_stage_count() {
        let selector = AstMultiStageSelector::new()
            .then(SimpleCostEstimator::new(), 100)
            .then(SimpleCostEstimator::new(), 50)
            .then(SimpleCostEstimator::new(), 10);

        assert_eq!(selector.stage_count(), 3);
    }
}
