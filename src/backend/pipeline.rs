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
    Optimizer, RuleBaseOptimizer, RuleBaseSuggester, SimpleCostEstimator as AstSimpleCostEstimator,
};
use crate::opt::graph::{
    AstOptimizationSuggester, BeamSearchGraphOptimizer, CompositeSuggester,
    ContiguousInsertionSuggester, FusionSuggester, GraphCostEstimator, KernelMergeCostEstimator,
    KernelMergeSuggester, LoweringSuggester, TilingSuggester, ViewInsertionSuggester,
    ViewMergeSuggester,
};

/// Suggesterの種類を指定するフラグ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SuggesterFlags {
    /// KernelMergeSuggesterを含めるかどうか
    ///
    /// trueの場合、単一ステージでCustom(Function)のマージも行います。
    /// これにより、部分的にloweringされた状態でも増分マージが可能になります。
    pub include_kernel_merge: bool,

    /// AstOptimizationSuggesterを含めるかどうか
    ///
    /// trueの場合、CustomノードのASTに対してAST最適化を適用します。
    /// これにより、グラフ最適化とAST最適化を統合的に探索できます。
    pub include_ast_optimization: bool,
}

impl SuggesterFlags {
    /// デフォルトのSuggesterフラグを作成
    ///
    /// デフォルトではKernelMergeSuggester、AstOptimizationSuggesterは含まれません。
    pub fn new() -> Self {
        Self {
            include_kernel_merge: false,
            include_ast_optimization: false,
        }
    }

    /// KernelMergeSuggesterを含む単一ステージ最適化用のフラグを作成
    ///
    /// Custom(Program)の増分マージをサポートし、
    /// 単一のビームサーチでloweringからマージまで行います。
    pub fn single_stage() -> Self {
        Self {
            include_kernel_merge: true,
            include_ast_optimization: false,
        }
    }

    /// 統合最適化用のフラグを作成
    ///
    /// KernelMergeSuggesterとAstOptimizationSuggesterの両方を含み、
    /// グラフ最適化とAST最適化を単一のビームサーチで探索します。
    pub fn unified() -> Self {
        Self {
            include_kernel_merge: true,
            include_ast_optimization: true,
        }
    }

    /// KernelMergeSuggesterを含めるかどうかを設定
    pub fn with_kernel_merge(mut self, include: bool) -> Self {
        self.include_kernel_merge = include;
        self
    }

    /// AstOptimizationSuggesterを含めるかどうかを設定
    pub fn with_ast_optimization(mut self, include: bool) -> Self {
        self.include_ast_optimization = include;
        self
    }
}

/// グラフ最適化用のSuggesterを作成
///
/// `flags.include_kernel_merge`がtrueの場合、KernelMergeSuggesterも含まれ、
/// 単一ステージでloweringからマージまで行います。
///
/// `flags.include_ast_optimization`がtrueの場合、AstOptimizationSuggesterも含まれ、
/// CustomノードのASTに対してAST最適化を適用します。
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
    ];

    // 単一ステージモードの場合、KernelMergeSuggesterも含める
    // これにより、Custom(Program)の増分マージが可能になる
    if flags.include_kernel_merge {
        suggesters.push(Box::new(KernelMergeSuggester::new()));
    }

    // AST最適化を含める場合、AstOptimizationSuggesterを追加
    // CustomノードのASTに対してRuleBaseSuggesterなどを適用
    if flags.include_ast_optimization {
        let ast_suggesters = create_ast_suggesters_for_graph();
        suggesters.push(Box::new(
            AstOptimizationSuggester::new(ast_suggesters).with_max_suggestions_per_node(2),
        ));
    }

    CompositeSuggester::new(suggesters)
}

/// グラフ最適化内で使用するAstSuggesterを作成
fn create_ast_suggesters_for_graph() -> Vec<Box<dyn crate::opt::ast::Suggester>> {
    vec![
        // ルールベース最適化（代数的簡約など）
        Box::new(RuleBaseSuggester::new(all_rules_with_search())),
        // ループタイリング
        Box::new(LoopTilingSuggester::with_default_sizes()),
        // ループ融合
        Box::new(LoopFusionSuggester::new()),
    ]
}

/// カーネルマージ用のSuggesterを作成
///
/// KernelMergeSuggesterのみを含むSuggesterを作成します。
/// グラフ最適化の第2フェーズで使用します。
pub fn create_kernel_merge_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(KernelMergeSuggester::new())])
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
pub fn optimize_ast_with_history(
    program: AstNode,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
) -> (AstNode, AstOptimizationHistory) {
    // 1. ルールベース最適化（代数的簡約など）を先に適用
    let rules = all_rules_with_search();
    let rule_optimizer = RuleBaseOptimizer::new(rules).with_max_iterations(100);
    let program = rule_optimizer.optimize(program);

    // 2. ビームサーチ最適化を適用
    let suggester = create_ast_suggester();
    let estimator = AstSimpleCostEstimator::new();
    let optimizer =
        create_ast_optimizer(suggester, estimator, beam_width, max_steps, show_progress);

    optimizer.optimize_with_history(program)
}

/// カーネルマージ最適化を実行（履歴付き）
///
/// 複数のCustom(Function)を1つのCustom(Program)にマージします。
/// グラフ最適化の第2フェーズとして使用します。
pub fn optimize_kernel_merge_with_history(
    graph: Graph,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
) -> (Graph, crate::opt::graph::OptimizationHistory) {
    let suggester = create_kernel_merge_suggester();
    let estimator = KernelMergeCostEstimator::new();
    let optimizer =
        create_graph_optimizer(suggester, estimator, beam_width, max_steps, show_progress);
    optimizer.optimize_with_history(graph)
}

/// 2段階グラフ最適化を実行（履歴付き）
///
/// 1. 第1フェーズ: 一般的なグラフ最適化（fusion, lowering等）
/// 2. 第2フェーズ: カーネルマージ（複数Custom(Function)を1つのCustom(Program)に統合）
///
/// # Arguments
/// * `graph` - 最適化対象のグラフ
/// * `flags` - Suggesterの設定フラグ
/// * `estimator` - 第1フェーズ用のコスト推定器
/// * `beam_width` - ビームサーチの幅
/// * `max_steps` - 各フェーズの最大ステップ数
/// * `show_progress` - 進捗表示フラグ
/// * `enable_kernel_merge` - カーネルマージを有効にするか
///
/// # Note
/// 単一ステージ最適化を使用したい場合は、`optimize_graph_single_stage`を使用してください。
#[deprecated(
    since = "0.2.0",
    note = "Use `optimize_graph_single_stage` for unified optimization, or `optimize_graph_with_history` with `SuggesterFlags::single_stage()` for single-stage mode"
)]
pub fn optimize_graph_two_phase<E>(
    graph: Graph,
    flags: SuggesterFlags,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
    enable_kernel_merge: bool,
) -> (
    Graph,
    crate::opt::graph::OptimizationHistory,
    Option<crate::opt::graph::OptimizationHistory>,
)
where
    E: GraphCostEstimator,
{
    // 第1フェーズ: 一般的なグラフ最適化
    let (phase1_graph, phase1_history) = optimize_graph_with_history(
        graph,
        flags,
        estimator,
        beam_width,
        max_steps,
        show_progress,
    );

    // 第2フェーズ: カーネルマージ（有効な場合）
    if enable_kernel_merge {
        let (phase2_graph, phase2_history) = optimize_kernel_merge_with_history(
            phase1_graph,
            beam_width,
            max_steps / 2, // カーネルマージは少ないステップで十分
            show_progress,
        );
        (phase2_graph, phase1_history, Some(phase2_history))
    } else {
        (phase1_graph, phase1_history, None)
    }
}

/// 単一ステージグラフ最適化を実行（履歴付き）
///
/// KernelMergeSuggesterを含む単一のビームサーチで、
/// fusion, lowering, カーネルマージを統合的に最適化します。
///
/// Custom(Program)の増分マージをサポートするため、
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
