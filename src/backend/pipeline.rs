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
    BeamSearchGraphOptimizer, CompositeSuggester, ConstPropagationSuggester,
    ContiguousInsertionSuggester, CustomFusionSuggester, FusionSuggester, GraphCostEstimator,
    KernelMergeCostEstimator, KernelMergeSuggester, LoweringSuggester, TilingSuggester,
    ViewInsertionSuggester, ViewMergeSuggester,
};

/// Suggesterの種類を指定するフラグ
///
/// 現在は将来の拡張のためのプレースホルダーです。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SuggesterFlags;

impl SuggesterFlags {
    /// デフォルトのSuggesterフラグを作成
    pub fn new() -> Self {
        Self
    }
}

/// グラフ最適化用のSuggesterを作成
pub fn create_graph_suggester(_flags: SuggesterFlags) -> CompositeSuggester {
    let suggesters: Vec<Box<dyn crate::opt::graph::GraphSuggester>> = vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(ViewMergeSuggester::new()),
        Box::new(ConstPropagationSuggester::new()),
        Box::new(TilingSuggester::with_default_tile_sizes()),
        Box::new(ContiguousInsertionSuggester::new()),
        Box::new(CustomFusionSuggester::new()),
        Box::new(FusionSuggester::new()),
        // LoweringSuggesterは他の最適化後にlowering
        Box::new(LoweringSuggester::new()),
        // 注意: KernelMergeSuggesterはここには含めない
        // カーネルマージは別のフェーズ（optimize_graph_with_kernel_merge）で行う
    ];

    CompositeSuggester::new(suggesters)
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
