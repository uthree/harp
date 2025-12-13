//! 候補選択のトレイトと実装
//!
//! ビームサーチなどの最適化アルゴリズムで、
//! コスト付き候補から上位n件を選択する処理を抽象化します。
//!
//! # 設計意図
//!
//! tinygradのような多段階評価を可能にするための抽象化です：
//! 1. 静的評価で明らかに悪い候補を足切り
//! 2. 中間的なヒューリスティクスで絞り込み
//! 3. 実行時間の実測値で精密に評価
//!
//! # Example
//!
//! ```ignore
//! use harp::opt::selector::{Selector, MultiStageSelector};
//!
//! // 3段階選択: 静的コスト→メモリ推定→実測
//! let selector = MultiStageSelector::new()
//!     .then(|c| estimate_static_cost(c), 1000)
//!     .then(|c| estimate_memory(c), 100)
//!     .then(|c| measure_runtime(c), 10);
//!
//! let selected = selector.select(candidates, 5);
//! ```

use std::cmp::Ordering;

use crate::ast::AstNode;
use crate::backend::{Compiler, KernelSignature, Renderer};
use crate::graph::Graph;
use crate::opt::ast::{CostEstimator, RuntimeCostEstimator, SimpleCostEstimator};

/// 候補選択のトレイト
///
/// ビームサーチなどの最適化アルゴリズムにおいて、
/// コスト付き候補から上位n件を選択する処理を抽象化します。
///
/// # Type Parameters
///
/// * `T` - 候補の型
pub trait Selector<T> {
    /// コスト付き候補リストから上位n件を選択
    ///
    /// # Arguments
    ///
    /// * `candidates` - (候補, コスト) のベクタ
    /// * `n` - 選択する最大件数
    ///
    /// # Returns
    ///
    /// 選択された (候補, コスト) のベクタ（最大n件）
    fn select(&self, candidates: Vec<(T, f32)>, n: usize) -> Vec<(T, f32)>;
}

/// 静的コストベースの選択器
///
/// 入力されたコストで昇順ソートして上位n件を選択。
/// デフォルトの選択器として使用されます。
#[derive(Default, Clone, Debug)]
pub struct StaticCostSelector;

impl StaticCostSelector {
    /// 新しいStaticCostSelectorを作成
    pub fn new() -> Self {
        Self
    }
}

impl<T> Selector<T> for StaticCostSelector {
    fn select(&self, candidates: Vec<(T, f32)>, n: usize) -> Vec<(T, f32)> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }
}

/// 選択ステージ
///
/// 各ステージは評価関数と残す候補数を持ちます。
struct SelectionStage<T> {
    /// 評価関数（候補からコストを計算）
    evaluator: Box<dyn Fn(&T) -> f32>,
    /// このステージで残す候補数
    keep_count: usize,
}

/// 多段階選択器
///
/// メソッドチェーンで複数のステージを構築し、段階的に候補を絞り込みます。
/// 各ステージでは評価関数によりコストを再計算し、上位keep_count件を残します。
///
/// # Example
///
/// ```ignore
/// use harp::opt::selector::{Selector, MultiStageSelector};
///
/// // 3段階選択
/// let selector = MultiStageSelector::new()
///     .then(|c| static_cost(c), 1000)   // 静的コストで1000件に足切り
///     .then(|c| memory_cost(c), 100)    // メモリコストで100件に絞り込み
///     .then(|c| runtime(c), 10);        // 実測で10件を最終選択
///
/// let selected = selector.select(candidates, 5);
/// ```
///
/// # 設計
///
/// [dagopt](https://github.com/uthree/dagopt)の設計を参考にしています。
pub struct MultiStageSelector<T> {
    stages: Vec<SelectionStage<T>>,
}

impl<T> Default for MultiStageSelector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MultiStageSelector<T> {
    /// 新しいMultiStageSelectorを作成
    pub fn new() -> Self {
        Self { stages: vec![] }
    }

    /// 選択ステージを追加
    ///
    /// # Arguments
    ///
    /// * `evaluator` - 候補からコストを計算する評価関数
    /// * `keep_count` - このステージで残す候補数
    ///
    /// # Example
    ///
    /// ```ignore
    /// let selector = MultiStageSelector::new()
    ///     .then(|c| c.node_count() as f32, 100)
    ///     .then(|c| measure_runtime(c), 10);
    /// ```
    pub fn then<F>(mut self, evaluator: F, keep_count: usize) -> Self
    where
        F: Fn(&T) -> f32 + 'static,
    {
        self.stages.push(SelectionStage {
            evaluator: Box::new(evaluator),
            keep_count,
        });
        self
    }

    /// ステージ数を取得
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl<T> Selector<T> for MultiStageSelector<T> {
    fn select(&self, candidates: Vec<(T, f32)>, n: usize) -> Vec<(T, f32)> {
        if candidates.is_empty() {
            return vec![];
        }

        // ステージがない場合は入力のコストでソートして返す
        if self.stages.is_empty() {
            let mut sorted = candidates;
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            return sorted.into_iter().take(n).collect();
        }

        // 最初のステージでコストを再計算
        let first_stage = &self.stages[0];
        let mut candidates_with_cost: Vec<(T, f32)> = candidates
            .into_iter()
            .map(|(candidate, _old_cost)| {
                let cost = (first_stage.evaluator)(&candidate);
                (candidate, cost)
            })
            .collect();
        candidates_with_cost.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        candidates_with_cost.truncate(first_stage.keep_count);

        // 2番目以降のステージ
        for (i, stage) in self.stages.iter().enumerate().skip(1) {
            // 各候補のコストを再計算
            candidates_with_cost = candidates_with_cost
                .into_iter()
                .map(|(candidate, _old_cost)| {
                    let new_cost = (stage.evaluator)(&candidate);
                    (candidate, new_cost)
                })
                .collect();

            // コストでソート
            candidates_with_cost.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            // 最後のステージでは n を考慮、それ以外は keep_count で截断
            let limit = if i == self.stages.len() - 1 {
                n.min(stage.keep_count)
            } else {
                stage.keep_count
            };

            candidates_with_cost.truncate(limit);
        }

        // 最終的にnで切る
        candidates_with_cost.truncate(n);
        candidates_with_cost
    }
}

/// ランタイムコストベースの選択器
///
/// 静的コストで足切りした後、実行時間を計測して最終選択を行います。
/// AST最適化のビームサーチにおいて、より正確なコスト評価を提供します。
///
/// # 2段階評価
///
/// 1. **Stage 1**: 静的コスト（SimpleCostEstimator）で`pre_filter_count`件に足切り
/// 2. **Stage 2**: 実行時間計測（RuntimeCostEstimator）で`n`件を最終選択
///
/// # Example
///
/// ```ignore
/// use harp::opt::selector::RuntimeSelector;
/// use harp::backend::c::{CRenderer, CCompiler};
///
/// let selector = RuntimeSelector::new(
///     CRenderer::new(),
///     CCompiler::new(),
///     signature,
///     |sig| create_buffers(sig),
/// )
/// .with_pre_filter_count(10)
/// .with_measurement_count(5);
///
/// let optimizer = BeamSearchOptimizer::new(suggester)
///     .with_selector(selector);
/// ```
///
/// # Type Parameters
///
/// * `R` - レンダラーの型
/// * `C` - コンパイラの型
pub struct RuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 静的コスト推定器（足切り用）
    static_estimator: SimpleCostEstimator,
    /// ランタイムコスト評価器
    runtime_estimator: RuntimeCostEstimator<R, C>,
    /// 足切り候補数（デフォルト: 10）
    pre_filter_count: usize,
}

impl<R, C> RuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 新しいRuntimeSelectorを作成
    ///
    /// # Arguments
    ///
    /// * `renderer` - ASTをソースコードに変換するレンダラー
    /// * `compiler` - ソースコードをカーネルにコンパイルするコンパイラ
    /// * `signature` - カーネルシグネチャ（入出力バッファの形状情報）
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    pub fn new<F>(renderer: R, compiler: C, signature: KernelSignature, buffer_factory: F) -> Self
    where
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + 'static,
    {
        Self {
            static_estimator: SimpleCostEstimator::new(),
            runtime_estimator: RuntimeCostEstimator::new(
                renderer,
                compiler,
                signature,
                buffer_factory,
            ),
            pre_filter_count: 10,
        }
    }

    /// 足切り候補数を設定
    ///
    /// 静的コストで上位何件を残すかを指定します。
    /// デフォルトは10件です。
    pub fn with_pre_filter_count(mut self, count: usize) -> Self {
        self.pre_filter_count = count.max(1);
        self
    }

    /// 計測回数を設定
    ///
    /// 実行時間計測の回数を指定します。
    /// デフォルトは10回です。
    pub fn with_measurement_count(mut self, count: usize) -> Self {
        self.runtime_estimator = self.runtime_estimator.with_measurement_count(count);
        self
    }
}

impl<R, C> Selector<AstNode> for RuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    fn select(&self, candidates: Vec<(AstNode, f32)>, n: usize) -> Vec<(AstNode, f32)> {
        if candidates.is_empty() {
            return vec![];
        }

        // Stage 1: 静的コストで足切り（入力のコストを再計算して使用）
        let mut stage1_candidates: Vec<(AstNode, f32)> = candidates
            .into_iter()
            .map(|(ast, _old_cost)| {
                let cost = self.static_estimator.estimate(&ast);
                (ast, cost)
            })
            .collect();

        stage1_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        stage1_candidates.truncate(self.pre_filter_count);

        // Stage 2: 実行時間計測で最終選択
        let mut stage2_candidates: Vec<(AstNode, f32)> = stage1_candidates
            .into_iter()
            .map(|(ast, _)| {
                let runtime_cost = self.runtime_estimator.measure(&ast);
                (ast, runtime_cost)
            })
            .collect();

        stage2_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        stage2_candidates.truncate(n);

        stage2_candidates
    }
}

use crate::opt::graph::{
    GraphCostEstimator, GraphRuntimeCostEstimator, SimpleCostEstimator as GraphSimpleCostEstimator,
};

/// グラフ用ランタイムコストベースの選択器
///
/// 静的コストで足切りした後、Lowering→コンパイル→実行時間計測を行って最終選択します。
/// グラフ最適化のビームサーチにおいて、より正確なコスト評価を提供します。
///
/// # 2段階評価
///
/// 1. **Stage 1**: 静的コスト（SimpleCostEstimator）で`pre_filter_count`件に足切り
/// 2. **Stage 2**: 実行時間計測（GraphRuntimeCostEstimator）で`n`件を最終選択
///
/// # Example
///
/// ```ignore
/// use harp::opt::selector::GraphRuntimeSelector;
/// use harp::backend::c::{CRenderer, CCompiler};
///
/// let selector = GraphRuntimeSelector::new(
///     CRenderer::new(),
///     CCompiler::new(),
///     |sig| create_buffers(sig),
/// )
/// .with_pre_filter_count(5)
/// .with_measurement_count(5);
///
/// let optimizer = BeamSearchGraphOptimizer::new(suggester)
///     .with_selector(selector);
/// ```
///
/// # Type Parameters
///
/// * `R` - レンダラーの型
/// * `C` - コンパイラの型
pub struct GraphRuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 静的コスト推定器（足切り用）
    static_estimator: GraphSimpleCostEstimator,
    /// ランタイムコスト評価器
    runtime_estimator: GraphRuntimeCostEstimator<R, C>,
    /// 足切り候補数（デフォルト: 5）
    pre_filter_count: usize,
}

impl<R, C> GraphRuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 新しいGraphRuntimeSelectorを作成
    ///
    /// # Arguments
    ///
    /// * `renderer` - ASTをソースコードに変換するレンダラー
    /// * `compiler` - ソースコードをカーネルにコンパイルするコンパイラ
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    pub fn new<F>(renderer: R, compiler: C, buffer_factory: F) -> Self
    where
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + 'static,
    {
        Self {
            static_estimator: GraphSimpleCostEstimator::new(),
            runtime_estimator: GraphRuntimeCostEstimator::new(renderer, compiler, buffer_factory),
            pre_filter_count: 5, // グラフ用はデフォルトを低めに
        }
    }

    /// 足切り候補数を設定
    ///
    /// 静的コストで上位何件を残すかを指定します。
    /// デフォルトは5件です（AST用より少なめ）。
    pub fn with_pre_filter_count(mut self, count: usize) -> Self {
        self.pre_filter_count = count.max(1);
        self
    }

    /// 計測回数を設定
    ///
    /// 実行時間計測の回数を指定します。
    /// デフォルトは5回です（AST用より少なめ）。
    pub fn with_measurement_count(mut self, count: usize) -> Self {
        self.runtime_estimator = self.runtime_estimator.with_measurement_count(count);
        self
    }

    /// Loweringの最大ステップ数を設定
    ///
    /// 簡易Loweringで使用する最大ステップ数を指定します。
    /// デフォルトは1000です。
    pub fn with_lowering_max_steps(mut self, steps: usize) -> Self {
        self.runtime_estimator = self.runtime_estimator.with_lowering_max_steps(steps);
        self
    }
}

impl<R, C> Selector<(Graph, String)> for GraphRuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    fn select(
        &self,
        candidates: Vec<((Graph, String), f32)>,
        n: usize,
    ) -> Vec<((Graph, String), f32)> {
        if candidates.is_empty() {
            return vec![];
        }

        // Stage 1: 静的コストで足切り（入力のコストを再計算して使用）
        let mut stage1_candidates: Vec<((Graph, String), f32)> = candidates
            .into_iter()
            .map(|((graph, name), _old_cost)| {
                let cost = self.static_estimator.estimate(&graph);
                ((graph, name), cost)
            })
            .collect();

        stage1_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        stage1_candidates.truncate(self.pre_filter_count);

        // Stage 2: 実行時間計測で最終選択
        let mut stage2_candidates: Vec<((Graph, String), f32)> = stage1_candidates
            .into_iter()
            .map(|((graph, name), _)| {
                let runtime_cost = self.runtime_estimator.measure(&graph);
                ((graph, name), runtime_cost)
            })
            .collect();

        stage2_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        stage2_candidates.truncate(n);

        stage2_candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_cost_selector_basic() {
        let selector = StaticCostSelector::new();
        let candidates = vec![("a", 3.0), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "b"); // コスト1.0
        assert_eq!(selected[1].0, "c"); // コスト2.0
    }

    #[test]
    fn test_static_cost_selector_empty() {
        let selector = StaticCostSelector::new();
        let candidates: Vec<(&str, f32)> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_static_cost_selector_zero_n() {
        let selector = StaticCostSelector::new();
        let candidates = vec![("a", 1.0)];

        let selected = selector.select(candidates, 0);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_static_cost_selector_with_nan() {
        let selector = StaticCostSelector::new();
        let candidates = vec![("a", f32::NAN), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        // NaNの扱いは未定義だが、パニックしないことを確認
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_multi_stage_selector_single_stage() {
        // 単一ステージ: 入力コストを無視して評価関数でコストを再計算
        let evaluator = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0, // 入力コスト3.0だが評価では最良
                "b" => 2.0, // 入力コスト1.0だが評価では2番目
                "c" => 3.0, // 入力コスト2.0だが評価では最悪
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new().then(evaluator, 3);
        let candidates = vec![("a", 3.0), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "a"); // 評価コスト1.0
        assert_eq!(selected[1].0, "b"); // 評価コスト2.0
    }

    #[test]
    fn test_multi_stage_selector_two_stages() {
        // 2段階選択
        let stage1_eval = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                "d" => 4.0,
                "e" => 5.0,
                _ => f32::MAX,
            }
        };
        let stage2_eval = |s: &&str| -> f32 {
            match *s {
                "a" => 3.0,
                "b" => 2.0,
                "c" => 1.0,
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new()
            .then(stage1_eval, 3)
            .then(stage2_eval, 2);

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0), ("e", 0.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "c"); // Stage2コスト1.0
        assert_eq!(selected[1].0, "b"); // Stage2コスト2.0
    }

    #[test]
    fn test_multi_stage_selector_three_stages() {
        let stage1 = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                "d" => 4.0,
                _ => f32::MAX,
            }
        };
        let stage2 = |s: &&str| -> f32 {
            match *s {
                "a" => 3.0,
                "b" => 1.0,
                "c" => 2.0,
                _ => f32::MAX,
            }
        };
        let stage3 = |s: &&str| -> f32 {
            match *s {
                "b" => 2.0,
                "c" => 1.0,
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new()
            .then(stage1, 3)
            .then(stage2, 2)
            .then(stage3, 1);

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0)];

        let selected = selector.select(candidates, 1);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, "c");
    }

    #[test]
    fn test_multi_stage_selector_no_stages() {
        // ステージがない場合は入力のコストでソートして返す
        let selector: MultiStageSelector<&str> = MultiStageSelector::new();
        let candidates = vec![("a", 3.0), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "b"); // コスト1.0
        assert_eq!(selected[1].0, "c"); // コスト2.0
    }

    #[test]
    fn test_multi_stage_selector_cutoff() {
        let stage1 = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                "d" => 10.0,
                _ => f32::MAX,
            }
        };
        let stage2 = |s: &&str| -> f32 {
            match *s {
                "d" => 0.0,
                "a" => 3.0,
                "b" => 2.0,
                "c" => 1.0,
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new().then(stage1, 3).then(stage2, 2);

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "c");
        assert_eq!(selected[1].0, "b");
    }

    #[test]
    fn test_multi_stage_selector_n_smaller_than_keep() {
        let eval = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new().then(eval, 10);

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "a");
        assert_eq!(selected[1].0, "b");
    }

    #[test]
    fn test_multi_stage_selector_stage_count() {
        let selector: MultiStageSelector<i32> = MultiStageSelector::new()
            .then(|_| 0.0, 10)
            .then(|_| 0.0, 5)
            .then(|_| 0.0, 2);

        assert_eq!(selector.stage_count(), 3);
    }

    #[test]
    fn test_multi_stage_selector_empty_candidates() {
        let selector = MultiStageSelector::new().then(|_: &&str| 0.0, 10);
        let candidates: Vec<(&str, f32)> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }
}
