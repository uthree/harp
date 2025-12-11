/// Pipeline実装のための共通ヘルパー関数
///
/// 各バックエンドのPipeline実装で使用される共通機能を提供します。
use crate::ast::AstNode;
use crate::graph::Graph;
use crate::opt::ast::rules::all_rules_with_search;
use crate::opt::ast::{
    BeamSearchOptimizer as AstBeamSearchOptimizer, CompositeSuggester as AstCompositeSuggester,
    FunctionInliningSuggester, LoopFusionSuggester, LoopInliningSuggester,
    LoopInterchangeSuggester, LoopTilingSuggester, OptimizationHistory as AstOptimizationHistory,
    RuleBaseSuggester, SimpleCostEstimator as AstSimpleCostEstimator,
};
use crate::opt::graph::{
    BeamSearchGraphOptimizer, BufferAbsorptionSuggester, ChainedGraphOptimizer, CompositeSuggester,
    ContiguousInsertionSuggester, FusionSuggester, GraphCostEstimator, GraphOptimizer,
    KernelMergeSuggester, LoweringCostEstimator, LoweringSuggester, ProgramRootAbsorptionSuggester,
    ProgramRootBufferAbsorptionSuggester, SimpleCostEstimator, TilingSuggester,
    ViewInsertionSuggester, ViewMergeSuggester,
};

/// Suggesterの種類を指定するフラグ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SuggesterFlags {
    /// KernelMergeSuggesterを含めるかどうか
    ///
    /// trueの場合、単一ステージでCustom(Function)のマージも行います。
    /// これにより、部分的にloweringされた状態でも増分マージが可能になります。
    pub include_kernel_merge: bool,
}

impl SuggesterFlags {
    /// デフォルトのSuggesterフラグを作成
    ///
    /// デフォルトではKernelMergeSuggesterは含まれません。
    pub fn new() -> Self {
        Self {
            include_kernel_merge: false,
        }
    }

    /// KernelMergeSuggesterを含む単一ステージ最適化用のフラグを作成
    ///
    /// Custom(Program)の増分マージをサポートし、
    /// 単一のビームサーチでloweringからマージまで行います。
    pub fn single_stage() -> Self {
        Self {
            include_kernel_merge: true,
        }
    }

    /// KernelMergeSuggesterを含めるかどうかを設定
    pub fn with_kernel_merge(mut self, include: bool) -> Self {
        self.include_kernel_merge = include;
        self
    }
}

/// グラフ最適化用のSuggesterを作成
///
/// `flags.include_kernel_merge`がtrueの場合、KernelMergeSuggesterも含まれ、
/// 単一ステージでloweringからマージまで行います。
pub fn create_graph_suggester(flags: SuggesterFlags) -> CompositeSuggester {
    let mut suggesters: Vec<Box<dyn crate::opt::graph::GraphSuggester>> = vec![
        Box::new(ViewInsertionSuggester::new()),
        // ViewMergeSuggesterはView(Const)パターンもマージする
        Box::new(ViewMergeSuggester::new()),
        Box::new(TilingSuggester::with_default_tile_sizes()),
        Box::new(ContiguousInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
        // LoweringSuggesterは他の最適化後にlowering
        Box::new(LoweringSuggester::new()),
        // BufferAbsorptionSuggesterはCustomの入力Bufferをinput_buffersフィールドに取り込む
        Box::new(BufferAbsorptionSuggester::new()),
        // ProgramRootAbsorptionSuggesterはCustom(Function)をSinkに吸収
        Box::new(ProgramRootAbsorptionSuggester::new()),
        // ProgramRootBufferAbsorptionSuggesterはSinkの入力Bufferを除去
        Box::new(ProgramRootBufferAbsorptionSuggester::new()),
    ];

    // 単一ステージモードの場合、KernelMergeSuggesterも含める
    // これにより、Custom(Program)の増分マージが可能になる
    // 注: ProgramRootAbsorptionSuggesterが優先され、KernelMergeSuggesterは
    // 既存のCustom(Program)マージにのみ使用される
    if flags.include_kernel_merge {
        suggesters.push(Box::new(KernelMergeSuggester::new()));
    }

    CompositeSuggester::new(suggesters)
}

/// カーネルマージ用のSuggesterを作成
///
/// KernelMergeSuggesterのみを含むSuggesterを作成します。
/// グラフ最適化の第2フェーズで使用します。
pub fn create_kernel_merge_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(KernelMergeSuggester::new())])
}

// =============================================================================
// マルチフェーズ最適化用の関数
// =============================================================================

/// グラフ準備フェーズ用のSuggesterを作成
///
/// Phase 1: グラフ構造の最適化（View挿入、融合、タイリングなど）
/// Lowering前にグラフ構造を整理するために使用します。
pub fn create_graph_preparation_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(ViewMergeSuggester::new()),
        Box::new(TilingSuggester::with_default_tile_sizes()),
        Box::new(ContiguousInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
    ])
}

/// Loweringフェーズ用のSuggesterを作成
///
/// Phase 2: グラフノードをCustomノードに変換し、単一のProgramRootに集約
/// LoweringSuggester、BufferAbsorption、ProgramRootAbsorptionを使用します。
///
/// # 設計方針
/// - ViewMergeSuggesterを含むことで、Lowering後に残るViewをCustomに吸収
/// - ProgramRootAbsorptionSuggesterはSinkの直接の子のCustomのみを吸収
/// - この順序により、View -> Custom が Custom[view適用] になってからSinkに吸収される
pub fn create_lowering_phase_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        // LoweringSuggesterでGraphOp -> Custom(Function)に変換
        Box::new(LoweringSuggester::new()),
        // ViewMergeSuggesterでViewをCustomに吸収（Sink -> View -> Custom を Sink -> Custom[view適用] に変換）
        Box::new(ViewMergeSuggester::new()),
        // BufferAbsorptionでCustomノードに入力Bufferを取り込む
        Box::new(BufferAbsorptionSuggester::new()),
        // ProgramRootAbsorptionでCustom(Function)をSinkに吸収
        Box::new(ProgramRootAbsorptionSuggester::new()),
        // ProgramRootBufferAbsorptionでSinkの入力Bufferを除去
        Box::new(ProgramRootBufferAbsorptionSuggester::new()),
        // KernelMergeSuggesterで複数のCustom(Function)をマージ
        Box::new(KernelMergeSuggester::new()),
    ])
}

/// AST最適化用のSuggesterを作成
pub fn create_ast_suggester() -> AstCompositeSuggester {
    AstCompositeSuggester::new(vec![
        Box::new(RuleBaseSuggester::new(all_rules_with_search())),
        Box::new(LoopTilingSuggester::with_default_sizes()),
        Box::new(LoopInliningSuggester::with_default_limit()),
        Box::new(LoopInterchangeSuggester::new()),
        Box::new(LoopFusionSuggester::new()),
        Box::new(FunctionInliningSuggester::with_default_limit()),
    ])
}

/// グラフ最適化用のOptimizerを作成・設定
pub fn create_graph_optimizer<E>(
    suggester: CompositeSuggester,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
) -> BeamSearchGraphOptimizer<CompositeSuggester, E>
where
    E: GraphCostEstimator,
{
    BeamSearchGraphOptimizer::new(suggester, estimator)
        .with_beam_width(beam_width)
        .with_max_steps(max_steps)
        .with_progress(show_progress)
}

/// AST最適化用のOptimizerを作成・設定
pub fn create_ast_optimizer<E>(
    suggester: AstCompositeSuggester,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
) -> AstBeamSearchOptimizer<AstCompositeSuggester, E>
where
    E: crate::opt::ast::CostEstimator,
{
    AstBeamSearchOptimizer::new(suggester, estimator)
        .with_beam_width(beam_width)
        .with_max_steps(max_steps)
        .with_progress(show_progress)
}

/// グラフ最適化を実行（履歴付き）
pub fn optimize_graph_with_history<E>(
    graph: Graph,
    flags: SuggesterFlags,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
) -> (Graph, crate::opt::graph::OptimizationHistory)
where
    E: GraphCostEstimator,
{
    let suggester = create_graph_suggester(flags);
    let optimizer =
        create_graph_optimizer(suggester, estimator, beam_width, max_steps, show_progress);
    optimizer.optimize_with_history(graph)
}

/// AST最適化を実行（履歴付き）
///
/// ビームサーチ最適化を適用します。
/// RuleBaseSuggesterがビームサーチ内に含まれているため、
/// 代数的簡約などのルールベース最適化も統合的に探索されます。
pub fn optimize_ast_with_history(
    program: AstNode,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
) -> (AstNode, AstOptimizationHistory) {
    let suggester = create_ast_suggester();
    let estimator = AstSimpleCostEstimator::new();
    let optimizer =
        create_ast_optimizer(suggester, estimator, beam_width, max_steps, show_progress);

    optimizer.optimize_with_history(program)
}

/// 単一ステージグラフ最適化を実行（履歴付き）
///
/// KernelMergeSuggesterを含む単一のビームサーチで、
/// fusion, lowering, カーネルマージを統合的に最適化します。
///
/// Kernel(Program)の増分マージをサポートするため、
/// 従来の2ステージ最適化よりも柔軟な最適化が可能です。
///
/// # Arguments
/// * `graph` - 最適化対象のグラフ
/// * `estimator` - コスト推定器
/// * `beam_width` - ビームサーチの幅
/// * `max_steps` - 最大ステップ数
/// * `show_progress` - 進捗表示フラグ
///
/// # Example
/// ```ignore
/// use harp::backend::pipeline::optimize_graph_single_stage;
/// use harp::opt::graph::SimpleCostEstimator;
///
/// let (optimized, history) = optimize_graph_single_stage(
///     graph,
///     SimpleCostEstimator::new(),
///     8,    // beam_width
///     200,  // max_steps
///     true, // show_progress
/// );
/// ```
pub fn optimize_graph_single_stage<E>(
    graph: Graph,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
) -> (Graph, crate::opt::graph::OptimizationHistory)
where
    E: GraphCostEstimator,
{
    let flags = SuggesterFlags::single_stage();
    optimize_graph_with_history(
        graph,
        flags,
        estimator,
        beam_width,
        max_steps,
        show_progress,
    )
}

/// マルチフェーズグラフ最適化の設定
#[derive(Debug, Clone)]
pub struct MultiPhaseConfig {
    /// ビーム幅
    pub beam_width: usize,
    /// 各フェーズの最大ステップ数
    pub max_steps_per_phase: usize,
    /// 進捗表示フラグ
    pub show_progress: bool,
    /// ログ収集を有効にするか
    pub collect_logs: bool,
}

impl Default for MultiPhaseConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_steps_per_phase: 5000,
            show_progress: false,
            collect_logs: cfg!(debug_assertions),
        }
    }
}

impl MultiPhaseConfig {
    /// 新しい設定を作成
    pub fn new() -> Self {
        Self::default()
    }

    /// ビーム幅を設定
    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    /// 各フェーズの最大ステップ数を設定
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps_per_phase = steps;
        self
    }

    /// 進捗表示を設定
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// ログ収集を設定
    pub fn with_collect_logs(mut self, collect: bool) -> Self {
        self.collect_logs = collect;
        self
    }
}

/// マルチフェーズグラフ最適化を作成
///
/// 2つのフェーズで段階的にグラフを最適化します：
///
/// 1. **Preparation** (グラフ準備): View挿入、融合、タイリングなど
///    - コスト推定器: SimpleCostEstimator
///    - 目的: グラフ構造の最適化
///
/// 2. **Lowering** (Lowering): Custom変換、ProgramRoot集約
///    - コスト推定器: LoweringCostEstimator
///    - 目的: 単一のProgramRootノードへの変換
///
/// AST最適化はグラフ最適化とは別のフェーズとして、GenericPipelineで
/// Lowering完了後に独立して実行されます。
///
/// # Arguments
/// * `config` - 最適化の設定
///
/// # Returns
/// ChainedGraphOptimizer（各フェーズの名前付き）
///
/// # Example
/// ```ignore
/// use harp::backend::pipeline::{create_multi_phase_optimizer, MultiPhaseConfig};
/// use harp::opt::graph::GraphOptimizer;
///
/// let config = MultiPhaseConfig::new()
///     .with_beam_width(8)
///     .with_max_steps(3000)
///     .with_progress(true);
///
/// let optimizer = create_multi_phase_optimizer(config);
/// let (optimized, history) = optimizer.optimize_with_history(graph);
/// ```
pub fn create_multi_phase_optimizer(config: MultiPhaseConfig) -> ChainedGraphOptimizer {
    // Phase 1: グラフ準備（View挿入、融合など）
    let preparation_suggester = create_graph_preparation_suggester();
    let preparation_estimator = SimpleCostEstimator::new();
    let preparation_optimizer =
        BeamSearchGraphOptimizer::new(preparation_suggester, preparation_estimator)
            .with_beam_width(config.beam_width)
            .with_max_steps(config.max_steps_per_phase)
            .with_progress(config.show_progress)
            .with_collect_logs(config.collect_logs);

    // Phase 2: Lowering（Custom変換、ProgramRoot集約）
    let lowering_suggester = create_lowering_phase_suggester();
    let lowering_estimator = LoweringCostEstimator::new();
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester, lowering_estimator)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs);

    // チェーンを構築（各オプティマイザに名前を付けてからchainする）
    preparation_optimizer
        .with_name("Preparation")
        .chain(lowering_optimizer.with_name("Lowering"))
}

/// マルチフェーズグラフ最適化を実行（履歴付き）
///
/// 2つのフェーズで段階的にグラフを最適化し、各フェーズの履歴を結合して返します。
///
/// # Arguments
/// * `graph` - 最適化対象のグラフ
/// * `config` - 最適化の設定
///
/// # Returns
/// (最適化されたグラフ, 結合された最適化履歴)
///
/// # Example
/// ```ignore
/// use harp::backend::pipeline::{optimize_graph_multi_phase, MultiPhaseConfig};
///
/// let config = MultiPhaseConfig::new()
///     .with_beam_width(8)
///     .with_progress(true);
///
/// let (optimized, history) = optimize_graph_multi_phase(graph, config);
///
/// // 履歴には各フェーズの名前がプレフィックスとして付いている
/// for snapshot in history.snapshots() {
///     println!("{}: cost={}", snapshot.description, snapshot.cost);
/// }
/// ```
pub fn optimize_graph_multi_phase(
    graph: Graph,
    config: MultiPhaseConfig,
) -> (Graph, crate::opt::graph::OptimizationHistory) {
    let optimizer = create_multi_phase_optimizer(config);
    optimizer.optimize_with_history(graph)
}
