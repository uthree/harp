//! Graph最適化用のSelector
//!
//! ビームサーチでの候補選択を抽象化します。
//! 多段階のフィルタリングを可能にし、tinygradのように
//! 静的ヒューリスティクスで足切りしてから実行時間で評価するような
//! パイプラインを構築できます。
//!
//! # Example
//!
//! ```ignore
//! use harp::opt::graph::{GraphMultiStageSelector, SimpleCostEstimator};
//! use harp::opt::graph::GraphRuntimeCostEstimator;
//!
//! // 2段階の選択パイプライン
//! let selector = GraphMultiStageSelector::new()
//!     .then(SimpleCostEstimator::new(), 50)  // 静的コストで50件に
//!     .then_runtime(renderer, compiler, buffer_factory, 5); // 実行時間で5件に
//! ```

use std::cmp::Ordering;

use crate::backend::{Compiler, KernelSignature, Renderer};
use crate::graph::Graph;

use super::{GraphCostEstimator, GraphRuntimeCostEstimator, SimpleCostEstimator};

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

/// 選択ステージの種類
enum StageKind<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 静的コスト推定器
    Static(Box<dyn GraphCostEstimator>),
    /// ランタイムコスト推定器
    Runtime(GraphRuntimeCostEstimator<R, C>),
}

/// 選択ステージ
///
/// 多段階選択の1ステップを表します。
struct SelectionStage<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// このステージで使用するコスト推定器
    kind: StageKind<R, C>,
    /// このステージで残す候補数
    keep: usize,
}

impl<R, C> SelectionStage<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 静的コスト推定器を使用するステージを作成
    fn new_static<E: GraphCostEstimator + 'static>(estimator: E, keep: usize) -> Self {
        Self {
            kind: StageKind::Static(Box::new(estimator)),
            keep,
        }
    }

    /// ランタイムコスト推定器を使用するステージを作成
    fn new_runtime(estimator: GraphRuntimeCostEstimator<R, C>, keep: usize) -> Self {
        Self {
            kind: StageKind::Runtime(estimator),
            keep,
        }
    }

    /// コストを推定
    fn estimate(&self, graph: &Graph) -> f32 {
        match &self.kind {
            StageKind::Static(e) => e.estimate(graph),
            StageKind::Runtime(e) => e.measure(graph),
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
/// // 静的コストのみで2段階
/// let selector = GraphMultiStageSelector::new()
///     .then(SimpleCostEstimator::new(), 50)  // 第1段階: 50件に絞り込み
///     .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 10); // 第2段階: 10件に
///
/// // 静的コスト→実行時間の2段階
/// let selector = GraphMultiStageSelector::new()
///     .then(SimpleCostEstimator::new(), 20)  // 静的コストで20件に
///     .then_runtime(renderer, compiler, buffer_factory, 5); // 実行時間で5件に
/// ```
pub struct GraphMultiStageSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    stages: Vec<SelectionStage<R, C>>,
}

/// ダミーのRenderer/Compiler型（ランタイムステージなしの場合に使用）
mod dummy {
    use crate::ast::{AstNode, DType};
    use crate::backend::{Buffer, Compiler, Kernel, KernelSignature, Renderer};

    #[derive(Clone)]
    pub struct DummyRenderer;

    impl Renderer for DummyRenderer {
        type CodeRepr = String;
        type Option = ();

        fn render(&self, _ast: &AstNode) -> Self::CodeRepr {
            String::new()
        }

        fn is_available(&self) -> bool {
            true
        }
    }

    pub struct DummyBuffer;

    impl Buffer for DummyBuffer {
        fn allocate(_shape: Vec<usize>, _dtype: DType) -> Self {
            DummyBuffer
        }

        fn shape(&self) -> Vec<usize> {
            vec![]
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn to_bytes(&self) -> Vec<u8> {
            vec![]
        }

        fn from_bytes(&mut self, _bytes: &[u8]) -> Result<(), String> {
            Ok(())
        }

        fn byte_len(&self) -> usize {
            0
        }
    }

    pub struct DummyKernel;

    impl Kernel for DummyKernel {
        type Buffer = DummyBuffer;

        fn signature(&self) -> KernelSignature {
            KernelSignature::empty()
        }

        unsafe fn execute(&self, _buffers: &mut [&mut Self::Buffer]) -> Result<(), String> {
            Ok(())
        }
    }

    #[derive(Clone)]
    pub struct DummyCompiler;

    impl Compiler for DummyCompiler {
        type CodeRepr = String;
        type Kernel = DummyKernel;
        type Buffer = DummyBuffer;
        type Option = ();

        fn new() -> Self {
            DummyCompiler
        }

        fn is_available(&self) -> bool {
            true
        }

        fn compile(&mut self, _code: &Self::CodeRepr, _signature: KernelSignature) -> Self::Kernel {
            DummyKernel
        }

        fn create_buffer(&self, _shape: Vec<usize>, _element_size: usize) -> Self::Buffer {
            DummyBuffer
        }
    }
}

use dummy::{DummyCompiler, DummyRenderer};

impl Default for GraphMultiStageSelector<DummyRenderer, DummyCompiler> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphMultiStageSelector<DummyRenderer, DummyCompiler> {
    /// 新しい多段階セレクターを作成
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// 静的コスト推定ステージを追加
    ///
    /// # Arguments
    /// * `estimator` - このステージで使用するコスト推定器
    /// * `keep` - このステージで残す候補数
    pub fn then<E: GraphCostEstimator + 'static>(
        mut self,
        estimator: E,
        keep: usize,
    ) -> GraphMultiStageSelector<DummyRenderer, DummyCompiler> {
        self.stages
            .push(SelectionStage::new_static(estimator, keep));
        self
    }

    /// ランタイムコスト推定ステージを追加し、型を変換
    ///
    /// # Arguments
    /// * `renderer` - ASTをソースコードに変換するレンダラー
    /// * `compiler` - ソースコードをカーネルにコンパイルするコンパイラ
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    /// * `keep` - このステージで残す候補数
    pub fn then_runtime<R, C, F>(
        self,
        renderer: R,
        compiler: C,
        buffer_factory: F,
        keep: usize,
    ) -> GraphMultiStageSelector<R, C>
    where
        R: Renderer,
        C: Compiler<CodeRepr = R::CodeRepr>,
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + Send + Sync + 'static,
    {
        let runtime_estimator = GraphRuntimeCostEstimator::new(renderer, compiler, buffer_factory);

        // 既存のステージを新しい型に変換
        let mut new_stages: Vec<SelectionStage<R, C>> = self
            .stages
            .into_iter()
            .map(|stage| SelectionStage {
                kind: match stage.kind {
                    StageKind::Static(e) => StageKind::Static(e),
                    StageKind::Runtime(_) => {
                        unreachable!("DummyRenderer stages should not have runtime")
                    }
                },
                keep: stage.keep,
            })
            .collect();

        new_stages.push(SelectionStage::new_runtime(runtime_estimator, keep));

        GraphMultiStageSelector { stages: new_stages }
    }

    /// ステージ数を取得
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl<R, C> GraphMultiStageSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 静的コスト推定ステージを追加
    ///
    /// # Arguments
    /// * `estimator` - このステージで使用するコスト推定器
    /// * `keep` - このステージで残す候補数
    pub fn then_static<E: GraphCostEstimator + 'static>(
        mut self,
        estimator: E,
        keep: usize,
    ) -> Self {
        self.stages
            .push(SelectionStage::new_static(estimator, keep));
        self
    }

    /// ランタイムコスト推定ステージを追加
    ///
    /// 既にランタイムステージが設定されている場合に追加のランタイムステージを追加できます。
    /// 計測回数を変えたい場合などに使用します。
    ///
    /// # Arguments
    /// * `renderer` - ASTをソースコードに変換するレンダラー
    /// * `compiler` - ソースコードをカーネルにコンパイルするコンパイラ
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    /// * `keep` - このステージで残す候補数
    pub fn then_runtime_stage<F>(
        mut self,
        renderer: R,
        compiler: C,
        buffer_factory: F,
        keep: usize,
    ) -> Self
    where
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + Send + Sync + 'static,
    {
        let runtime_estimator = GraphRuntimeCostEstimator::new(renderer, compiler, buffer_factory);
        self.stages
            .push(SelectionStage::new_runtime(runtime_estimator, keep));
        self
    }

    /// ステージ数を取得
    pub fn stages(&self) -> usize {
        self.stages.len()
    }
}

impl<R, C> GraphSelector for GraphMultiStageSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    fn estimate(&self, candidate: &(Graph, String)) -> f32 {
        // 最初のステージのestimatorでコストを推定
        self.stages
            .first()
            .map(|s| s.estimate(&candidate.0))
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
                *cost = stage.estimate(graph);
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

/// グラフ用ランタイムコストベースの選択器（後方互換性のためのエイリアス）
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
/// use harp::opt::graph::GraphRuntimeSelector;
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
    static_estimator: SimpleCostEstimator,
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
            static_estimator: SimpleCostEstimator::new(),
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
