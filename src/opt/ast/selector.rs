//! AST最適化用のSelector
//!
//! ビームサーチでの候補選択を抽象化します。
//! 多段階のフィルタリングを可能にし、tinygradのように
//! 静的ヒューリスティクスで足切りしてから実行時間で評価するような
//! パイプラインを構築できます。
//!
//! # Example
//!
//! ```ignore
//! use harp::opt::ast::{AstMultiStageSelector, SimpleCostEstimator};
//! use harp::opt::ast::RuntimeCostEstimator;
//!
//! // 2段階の選択パイプライン
//! let selector = AstMultiStageSelector::new()
//!     .then(SimpleCostEstimator::new(), 100)  // 静的コストで100件に
//!     .then_runtime(renderer, compiler, signature, buffer_factory, 10); // 実行時間で10件に
//! ```

use std::cmp::Ordering;

use crate::ast::AstNode;
use crate::backend::{Compiler, KernelSignature, Renderer};

use super::{CostEstimator, RuntimeCostEstimator, SimpleCostEstimator};

/// AST最適化用のSelector trait
///
/// AST最適化のビームサーチにおいて、候補の評価と選択を抽象化します。
pub trait AstSelector {
    /// 単一候補のコストを推定
    fn estimate(&self, candidate: &AstNode) -> f32;

    /// 候補リストを評価し、上位n件を選択
    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)>;
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

/// 選択ステージの種類
enum StageKind<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 静的コスト推定器
    Static(Box<dyn CostEstimator>),
    /// ランタイムコスト推定器
    Runtime(RuntimeCostEstimator<R, C>),
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
    fn new_static<E: CostEstimator + 'static>(estimator: E, keep: usize) -> Self {
        Self {
            kind: StageKind::Static(Box::new(estimator)),
            keep,
        }
    }

    /// ランタイムコスト推定器を使用するステージを作成
    fn new_runtime(estimator: RuntimeCostEstimator<R, C>, keep: usize) -> Self {
        Self {
            kind: StageKind::Runtime(estimator),
            keep,
        }
    }

    /// コストを推定
    fn estimate(&self, ast: &AstNode) -> f32 {
        match &self.kind {
            StageKind::Static(e) => e.estimate(ast),
            StageKind::Runtime(e) => e.measure(ast),
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
/// use harp::opt::ast::{AstMultiStageSelector, SimpleCostEstimator};
///
/// // 静的コストのみで2段階
/// let selector = AstMultiStageSelector::new()
///     .then(SimpleCostEstimator::new(), 100)  // 第1段階: 100件に絞り込み
///     .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 10); // 第2段階: 10件に
///
/// // 静的コスト→実行時間の2段階
/// let selector = AstMultiStageSelector::new()
///     .then(SimpleCostEstimator::new(), 50)  // 静的コストで50件に
///     .then_runtime(renderer, compiler, signature, buffer_factory, 5); // 実行時間で5件に
/// ```
pub struct AstMultiStageSelector<R, C>
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

impl Default for AstMultiStageSelector<DummyRenderer, DummyCompiler> {
    fn default() -> Self {
        Self::new()
    }
}

impl AstMultiStageSelector<DummyRenderer, DummyCompiler> {
    /// 新しい多段階セレクターを作成
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// 静的コスト推定ステージを追加
    ///
    /// # Arguments
    /// * `estimator` - このステージで使用するコスト推定器
    /// * `keep` - このステージで残す候補数
    pub fn then<E: CostEstimator + 'static>(
        mut self,
        estimator: E,
        keep: usize,
    ) -> AstMultiStageSelector<DummyRenderer, DummyCompiler> {
        self.stages
            .push(SelectionStage::new_static(estimator, keep));
        self
    }

    /// ランタイムコスト推定ステージを追加し、型を変換
    ///
    /// # Arguments
    /// * `renderer` - ASTをソースコードに変換するレンダラー
    /// * `compiler` - ソースコードをカーネルにコンパイルするコンパイラ
    /// * `signature` - カーネルシグネチャ
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    /// * `keep` - このステージで残す候補数
    pub fn then_runtime<R, C, F>(
        self,
        renderer: R,
        compiler: C,
        signature: KernelSignature,
        buffer_factory: F,
        keep: usize,
    ) -> AstMultiStageSelector<R, C>
    where
        R: Renderer,
        C: Compiler<CodeRepr = R::CodeRepr>,
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + 'static,
    {
        let runtime_estimator =
            RuntimeCostEstimator::new(renderer, compiler, signature, buffer_factory);

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

        AstMultiStageSelector { stages: new_stages }
    }

    /// ステージ数を取得
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl<R, C> AstMultiStageSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 静的コスト推定ステージを追加
    ///
    /// # Arguments
    /// * `estimator` - このステージで使用するコスト推定器
    /// * `keep` - このステージで残す候補数
    pub fn then_static<E: CostEstimator + 'static>(mut self, estimator: E, keep: usize) -> Self {
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
    /// * `signature` - カーネルシグネチャ
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    /// * `keep` - このステージで残す候補数
    pub fn then_runtime_stage<F>(
        mut self,
        renderer: R,
        compiler: C,
        signature: KernelSignature,
        buffer_factory: F,
        keep: usize,
    ) -> Self
    where
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + 'static,
    {
        let runtime_estimator =
            RuntimeCostEstimator::new(renderer, compiler, signature, buffer_factory);
        self.stages
            .push(SelectionStage::new_runtime(runtime_estimator, keep));
        self
    }

    /// ステージ数を取得
    pub fn stages(&self) -> usize {
        self.stages.len()
    }
}

impl<R, C> AstSelector for AstMultiStageSelector<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    fn estimate(&self, candidate: &AstNode) -> f32 {
        // 最初のステージのestimatorでコストを推定
        self.stages
            .first()
            .map(|s| s.estimate(candidate))
            .unwrap_or(0.0)
    }

    fn select(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32)> {
        if candidates.is_empty() {
            return vec![];
        }

        if self.stages.is_empty() {
            // ステージがない場合はそのまま返す
            return candidates.into_iter().take(n).map(|c| (c, 0.0)).collect();
        }

        let mut current: Vec<(AstNode, f32)> = candidates.into_iter().map(|c| (c, 0.0)).collect();

        for (i, stage) in self.stages.iter().enumerate() {
            // コストを再計算
            for (ast, cost) in current.iter_mut() {
                *cost = stage.estimate(ast);
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

/// ランタイムコストベースの選択器（後方互換性のためのエイリアス）
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
/// use harp::opt::ast::RuntimeSelector;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_ast_cost_selector_basic() {
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

    #[test]
    fn test_multi_stage_selector_single_stage() {
        let selector = AstMultiStageSelector::new().then(SimpleCostEstimator::new(), 3);

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

    #[test]
    fn test_multi_stage_selector_two_stages() {
        let selector = AstMultiStageSelector::new()
            .then(SimpleCostEstimator::new(), 4)
            .then(SimpleCostEstimator::new().with_node_count_penalty(0.1), 2);

        let candidates = vec![
            AstNode::Const(Literal::Int(1)),
            AstNode::Const(Literal::Int(2)),
            AstNode::Const(Literal::Int(3)),
            AstNode::Const(Literal::Int(4)),
            AstNode::Const(Literal::Int(5)),
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
            AstNode::Const(Literal::Int(1)),
            AstNode::Const(Literal::Int(2)),
            AstNode::Const(Literal::Int(3)),
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
