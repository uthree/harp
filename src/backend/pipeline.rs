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
    ContiguousInsertionSuggester, FusionSuggester, GraphCostEstimator, ParallelStrategyChanger,
    SimdSuggester, TilingSuggester, ViewInsertionSuggester, ViewMergeSuggester,
};

/// Suggesterの種類を指定するフラグ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuggesterFlags {
    /// 並列化Suggesterを含めるか
    pub enable_parallel: bool,
    /// SIMD Suggesterを含めるか
    pub enable_simd: bool,
}

impl SuggesterFlags {
    /// すべてのSuggesterを有効にする
    pub fn all() -> Self {
        Self {
            enable_parallel: true,
            enable_simd: true,
        }
    }

    /// 並列化とSIMDを無効にする（シングルスレッドバックエンド用）
    pub fn single_threaded() -> Self {
        Self {
            enable_parallel: false,
            enable_simd: false,
        }
    }
}

/// グラフ最適化用のSuggesterを作成
pub fn create_graph_suggester(flags: SuggesterFlags) -> CompositeSuggester {
    let mut suggesters: Vec<Box<dyn crate::opt::graph::GraphSuggester>> = vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(ViewMergeSuggester::new()),
        Box::new(ConstPropagationSuggester::new()),
        Box::new(TilingSuggester::with_default_tile_sizes()),
        Box::new(ContiguousInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
    ];

    // 並列化Suggesterを追加（フラグが有効な場合）
    if flags.enable_parallel {
        suggesters.push(Box::new(ParallelStrategyChanger::new()));
    }

    // SIMD Suggesterを追加（フラグが有効な場合）
    if flags.enable_simd {
        suggesters.push(Box::new(SimdSuggester::new()));
    }

    CompositeSuggester::new(suggesters)
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
