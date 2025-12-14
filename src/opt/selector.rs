//! 候補選択のトレイトと実装
//!
//! ビームサーチなどの最適化アルゴリズムで、
//! 候補の評価と選択を抽象化します。
//!
//! # 設計
//!
//! ```text
//! Optimizer
//!   └── Selector
//!         └── CostEstimator(s)
//! ```
//!
//! - **CostEstimator**: `estimate` メソッドでコストを計算
//! - **Selector**: CostEstimatorを内包し、候補のソート・選択ロジックを担当
//! - **Optimizer**: Selectorのみを持つ（CostEstimatorは直接持たない）
//!
//! # Graph用とAST用のSelector
//!
//! 型安全性のため、Graph用とAST用で別のtraitを提供しています：
//! - `GraphSelector`: Graph最適化用（候補は`(Graph, String)`のタプル）
//! - `AstSelector`: AST最適化用（候補は`AstNode`）

use std::cmp::Ordering;

use crate::ast::AstNode;
use crate::backend::{Compiler, KernelSignature, Renderer};
use crate::graph::Graph;
use crate::opt::ast::{CostEstimator, RuntimeCostEstimator, SimpleCostEstimator};
use crate::opt::graph::{GraphCostEstimator, SimpleCostEstimator as GraphSimpleCostEstimator};

// =============================================================================
// Graph用Selector
// =============================================================================

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

// =============================================================================
// AST用Selector
// =============================================================================

/// AST最適化用のSelector trait
///
/// AST最適化のビームサーチにおいて、候補の評価と選択を抽象化します。
pub trait AstSelector {
    /// 単一候補のコストを推定
    fn estimate(&self, candidate: &AstNode) -> f32;

    /// 候補リストを評価し、上位n件を選択
    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)>;
}

/// Graph用の静的コストベース選択器
///
/// GraphCostEstimatorを内包し、静的コストで候補をソートして上位n件を選択します。
/// Graph最適化のデフォルトの選択器として使用されます。
#[derive(Clone, Debug)]
pub struct GraphCostSelector<E = GraphSimpleCostEstimator>
where
    E: GraphCostEstimator,
{
    estimator: E,
}

impl Default for GraphCostSelector<GraphSimpleCostEstimator> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCostSelector<GraphSimpleCostEstimator> {
    /// 新しいGraphCostSelectorを作成（デフォルトのSimpleCostEstimatorを使用）
    pub fn new() -> Self {
        Self {
            estimator: GraphSimpleCostEstimator::new(),
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

/// AST用の静的コストベース選択器
///
/// CostEstimatorを内包し、静的コストで候補をソートして上位n件を選択します。
/// AST最適化のデフォルトの選択器として使用されます。
#[derive(Clone, Debug)]
pub struct AstCostSelector<E = SimpleCostEstimator>
where
    E: CostEstimator,
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
    E: CostEstimator,
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
    E: CostEstimator,
{
    fn estimate(&self, candidate: &AstNode) -> f32 {
        self.estimator.estimate(candidate)
    }

    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)> {
        let mut with_cost: Vec<(AstNode, f32)> = candidates
            .into_iter()
            .map(|c| {
                let cost = self.estimator.estimate(&c);
                (c, cost)
            })
            .collect();
        with_cost.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        with_cost.into_iter().take(n).collect()
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
/// use harp::backend::opencl::{OpenCLRenderer, OpenCLCompiler};
///
/// let selector = RuntimeSelector::new(
///     OpenCLRenderer::new(),
///     OpenCLCompiler::new(),
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

impl<R, C> AstSelector for RuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    fn estimate(&self, candidate: &AstNode) -> f32 {
        self.static_estimator.estimate(candidate)
    }

    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)> {
        if candidates.is_empty() {
            return vec![];
        }

        // Stage 1: 静的コストで足切り
        let mut stage1_candidates: Vec<(AstNode, f32)> = candidates
            .into_iter()
            .map(|ast| {
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

use crate::opt::graph::GraphRuntimeCostEstimator;

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
/// use harp::backend::opencl::{OpenCLRenderer, OpenCLCompiler};
///
/// let selector = GraphRuntimeSelector::new(
///     OpenCLRenderer::new(),
///     OpenCLCompiler::new(),
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

impl<R, C> Clone for GraphRuntimeSelector<R, C>
where
    R: Renderer + Clone,
    C: Compiler<CodeRepr = R::CodeRepr> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            static_estimator: self.static_estimator,
            runtime_estimator: self.runtime_estimator.clone(),
            pre_filter_count: self.pre_filter_count,
        }
    }
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
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + Send + Sync + 'static,
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

impl<R, C> GraphSelector for GraphRuntimeSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    fn estimate(&self, candidate: &(Graph, String)) -> f32 {
        self.static_estimator.estimate(&candidate.0)
    }

    fn select(&self, candidates: Vec<(Graph, String)>, n: usize) -> Vec<((Graph, String), f32)> {
        if candidates.is_empty() {
            return vec![];
        }

        // Stage 1: 静的コストで足切り
        let mut stage1_candidates: Vec<((Graph, String), f32)> = candidates
            .into_iter()
            .map(|(graph, name)| {
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
    fn test_graph_cost_selector_basic() {
        let selector = GraphCostSelector::new();

        // 空のグラフを複数作成（コストはノード数に基づく）
        let mut graph1 = Graph::new();
        let a1 = graph1.input("a", crate::graph::DType::F32, vec![10, 10]);
        let b1 = graph1.input("b", crate::graph::DType::F32, vec![10, 10]);
        let _ = a1 + b1;

        let mut graph2 = Graph::new();
        let a2 = graph2.input("a", crate::graph::DType::F32, vec![10, 10]);
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
    fn test_ast_cost_selector_basic() {
        use crate::ast::Literal;

        let selector = AstCostSelector::new();

        // シンプルなASTノードを作成
        let ast1 = AstNode::Const(Literal::Int(42));
        let ast2 = AstNode::Const(Literal::F32(3.14));

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
        use crate::ast::Literal;

        let selector = AstCostSelector::new();

        let candidates = vec![
            AstNode::Const(Literal::Int(1)),
            AstNode::Const(Literal::Int(2)),
            AstNode::Const(Literal::Int(3)),
            AstNode::Const(Literal::Int(4)),
            AstNode::Const(Literal::Int(5)),
        ];

        let selected = selector.select(candidates, 3);
        assert_eq!(selected.len(), 3);
    }
}
