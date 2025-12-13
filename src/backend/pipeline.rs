/// Pipeline実装のための共通ヘルパー関数
///
/// 各バックエンドのPipeline実装で使用される共通機能を提供します。
use crate::graph::Graph;
use crate::opt::Selector;
use crate::opt::ast::{
    CompositeSuggester as AstCompositeSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
};
use crate::opt::graph::{
    BeamSearchGraphOptimizer, BufferAbsorptionSuggester, ChainedGraphOptimizer, CompositeSuggester,
    ContiguousInsertionSuggester, FusionSuggester, GraphOptimizer, KernelMergeSuggester,
    LoweringSuggester, ProgramRootAbsorptionSuggester, ProgramRootBufferAbsorptionSuggester,
    TilingSuggester, ViewInsertionSuggester, ViewMergeSuggester,
};

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
/// Phase 2: グラフノードをKernelノードに変換し、単一のProgramRootに集約
/// LoweringSuggester、BufferAbsorption、ProgramRootAbsorptionを使用します。
///
/// # 設計方針
/// - ViewMergeSuggesterを含むことで、Lowering後に残るViewをKernelに吸収
/// - ProgramRootAbsorptionSuggesterはProgramRootの直接の子のKernelのみを吸収
/// - この順序により、View -> Kernel が Kernel[view適用] になってからProgramRootに吸収される
pub fn create_lowering_phase_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        // LoweringSuggesterでGraphOp -> Kernel(Function)に変換
        Box::new(LoweringSuggester::new()),
        // ViewMergeSuggesterでViewをKernelに吸収（ProgramRoot -> View -> Kernel を ProgramRoot -> Kernel[view適用] に変換）
        Box::new(ViewMergeSuggester::new()),
        // BufferAbsorptionでKernelノードに入力Bufferを取り込む
        Box::new(BufferAbsorptionSuggester::new()),
        // ProgramRootAbsorptionでKernel(Function)をProgramRootに吸収
        Box::new(ProgramRootAbsorptionSuggester::new()),
        // ProgramRootBufferAbsorptionでProgramRootの入力Bufferを除去
        Box::new(ProgramRootBufferAbsorptionSuggester::new()),
        // KernelMergeSuggesterで複数のKernel(Function)をマージ
        Box::new(KernelMergeSuggester::new()),
    ])
}

/// 貪欲法Lowering用のSuggesterを作成
///
/// 実行時間の実測など、高速なloweringが必要な場合に使用します。
/// 並列化戦略はSequentialのみに制限され、探索空間を最小限に抑えます。
///
/// # 設計方針
/// - LoweringSuggester::sequential_only()で逐次実行のみに制限
/// - 他のSuggesterは通常通り使用
/// - ビームサーチ幅=1との組み合わせで貪欲法として動作
pub fn create_greedy_lowering_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        // LoweringSuggester::sequential_only()でSequential戦略のみ使用
        Box::new(LoweringSuggester::sequential_only()),
        Box::new(ViewMergeSuggester::new()),
        Box::new(BufferAbsorptionSuggester::new()),
        Box::new(ProgramRootAbsorptionSuggester::new()),
        Box::new(ProgramRootBufferAbsorptionSuggester::new()),
        Box::new(KernelMergeSuggester::new()),
    ])
}

/// ループ最適化用のSuggesterを作成（RuleBaseSuggesterを除く）
///
/// ルールベース最適化を事前に`RuleBaseOptimizer`で実行した後、
/// ループ構造の最適化のみを行う場合に使用します。
///
/// # Example
/// ```ignore
/// use harp::opt::ast::{RuleBaseOptimizer, BeamSearchOptimizer};
/// use harp::opt::ast::rules::all_algebraic_rules;
/// use harp::backend::pipeline::create_ast_loop_suggester;
///
/// // 第1段階: ルールベース最適化（高速、ビームサーチなし）
/// let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules());
/// let rule_optimized = rule_optimizer.optimize(program);
///
/// // 第2段階: ループ最適化（ビームサーチ）
/// let loop_suggester = create_ast_loop_suggester();
/// let beam_optimizer = BeamSearchOptimizer::new(loop_suggester);
/// let final_optimized = beam_optimizer.optimize(rule_optimized);
/// ```
pub fn create_ast_loop_suggester() -> AstCompositeSuggester {
    AstCompositeSuggester::new(vec![
        Box::new(LoopTilingSuggester::with_default_sizes()),
        Box::new(LoopInliningSuggester::with_default_limit()),
        Box::new(LoopInterchangeSuggester::new()),
        Box::new(LoopFusionSuggester::new()),
        Box::new(FunctionInliningSuggester::with_default_limit()),
    ])
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
    /// 早期終了の閾値（改善なしステップ数）
    ///
    /// Some(n): n回連続で改善がなければ終了
    /// None: 早期終了を無効化
    pub early_termination_threshold: Option<usize>,
}

impl Default for MultiPhaseConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_steps_per_phase: 5000,
            show_progress: false,
            collect_logs: cfg!(debug_assertions),
            early_termination_threshold: Some(10), // デフォルト: 10ステップ改善なしで終了
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

    /// 早期終了の閾値を設定
    ///
    /// Some(n): n回連続で改善がなければ終了
    /// None: 早期終了を無効化
    pub fn with_early_termination_threshold(mut self, threshold: Option<usize>) -> Self {
        self.early_termination_threshold = threshold;
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
/// 2. **Lowering** (Lowering): Kernel変換、ProgramRoot集約
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
    let preparation_optimizer = BeamSearchGraphOptimizer::new(preparation_suggester)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 2: Lowering（Kernel変換、ProgramRoot集約）
    let lowering_suggester = create_lowering_phase_suggester();
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // チェーンを構築（各オプティマイザに名前を付けてからchainする）
    preparation_optimizer
        .with_name("Preparation")
        .chain(lowering_optimizer.with_name("Lowering"))
}

/// マルチフェーズグラフ最適化を作成（LoweringフェーズにカスタムSelector使用）
///
/// `create_multi_phase_optimizer`と同様ですが、LoweringフェーズでカスタムSelectorを使用できます。
/// GraphRuntimeSelectorを使用した実測値ベースの最適化に使用します。
///
/// Phase 1（Preparation）は静的コスト推定を使用し、Phase 2（Lowering）で
/// カスタムSelectorを使用します。これは、Lowering済みのグラフは実行可能で
/// 直接計測できるためです。
///
/// # Arguments
/// * `config` - 最適化の設定
/// * `selector` - Loweringフェーズで使用するカスタムSelector（GraphRuntimeSelectorなど）
///
/// # Returns
/// ChainedGraphOptimizer（LoweringフェーズにカスタムSelectorが設定される）
///
/// # Example
/// ```ignore
/// use harp::backend::pipeline::{create_multi_phase_optimizer_with_selector, MultiPhaseConfig};
/// use harp::opt::selector::GraphRuntimeSelector;
/// use harp::opt::graph::GraphOptimizer;
///
/// let selector = GraphRuntimeSelector::new(renderer, compiler, buffer_factory);
/// let config = MultiPhaseConfig::new().with_beam_width(4);
/// let optimizer = create_multi_phase_optimizer_with_selector(config, selector);
/// let (optimized, history) = optimizer.optimize_with_history(graph);
/// ```
pub fn create_multi_phase_optimizer_with_selector<Sel>(
    config: MultiPhaseConfig,
    selector: Sel,
) -> ChainedGraphOptimizer
where
    Sel: Selector<(Graph, String)> + 'static,
{
    // Phase 1: グラフ準備（View挿入、融合など）- 静的コスト推定
    let preparation_suggester = create_graph_preparation_suggester();
    let preparation_optimizer = BeamSearchGraphOptimizer::new(preparation_suggester)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 2: Lowering（Kernel変換、ProgramRoot集約）- カスタムSelector使用
    let lowering_suggester = create_lowering_phase_suggester();
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_selector(selector)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // チェーンを構築
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

/// 貪欲法グラフ最適化用のOptimizerを作成
///
/// 実行時間の実測など、高速なloweringが必要な場合に使用します。
///
/// # 特徴
/// - ビーム幅=1で貪欲法として動作
/// - LoweringSuggesterはSequential戦略のみ使用
/// - 探索空間を最小限に抑え、高速にloweringを完了
///
/// # Arguments
/// * `config` - 最適化の設定（beam_widthは1に強制される）
///
/// # Returns
/// ChainedGraphOptimizer
pub fn create_greedy_optimizer(config: MultiPhaseConfig) -> ChainedGraphOptimizer {
    // Phase 1: グラフ準備（通常通り、ただしbeam_width=1）
    let preparation_suggester = create_graph_preparation_suggester();
    let preparation_optimizer = BeamSearchGraphOptimizer::new(preparation_suggester)
        .with_beam_width(1) // 貪欲法
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 2: Lowering（Sequential戦略のみ、beam_width=1）
    let lowering_suggester = create_greedy_lowering_suggester();
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_beam_width(1) // 貪欲法
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    preparation_optimizer
        .with_name("Preparation (Greedy)")
        .chain(lowering_optimizer.with_name("Lowering (Greedy)"))
}

/// 貪欲法グラフ最適化を実行（履歴付き）
///
/// ビーム幅=1、Sequential戦略のみで高速にloweringを行います。
/// 実行時間の実測など、軽量なloweringが必要な場合に使用します。
///
/// # Arguments
/// * `graph` - 最適化対象のグラフ
/// * `max_steps` - 各フェーズの最大ステップ数
///
/// # Returns
/// (最適化されたグラフ, 結合された最適化履歴)
///
/// # Example
/// ```ignore
/// use harp::backend::pipeline::optimize_graph_greedy;
///
/// let (optimized, history) = optimize_graph_greedy(graph, 5000);
/// ```
pub fn optimize_graph_greedy(
    graph: Graph,
    max_steps: usize,
) -> (Graph, crate::opt::graph::OptimizationHistory) {
    let config = MultiPhaseConfig::new()
        .with_beam_width(1)
        .with_max_steps(max_steps)
        .with_progress(false)
        .with_collect_logs(false);
    let optimizer = create_greedy_optimizer(config);
    optimizer.optimize_with_history(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, GraphOp};
    use crate::opt::graph::GraphSuggester;

    #[test]
    fn test_greedy_lowering_suggester_uses_sequential_only() {
        let suggester = create_greedy_lowering_suggester();

        // Suggesterの最初の要素がLoweringSuggester::sequential_only()であることを確認
        // CompositeSuggesterの内部構造にはアクセスできないため、
        // 実際にsuggestを呼び出して候補数で確認する
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        // 通常のLowering Suggesterを使った場合と比較
        let normal_suggester = create_lowering_phase_suggester();

        let greedy_suggestions = suggester.suggest(&graph);
        let normal_suggestions = normal_suggester.suggest(&graph);

        // 貪欲法用Suggesterは候補が少ない（Sequentialのみ）
        // 通常は複数の並列化戦略が生成される
        assert!(
            greedy_suggestions.len() <= normal_suggestions.len(),
            "Greedy suggester should generate fewer or equal candidates: greedy={}, normal={}",
            greedy_suggestions.len(),
            normal_suggestions.len()
        );
    }

    #[test]
    fn test_optimize_graph_greedy_basic() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        let (optimized, history) = optimize_graph_greedy(graph, 1000);

        // 最適化履歴が存在することを確認
        assert!(
            history.len() > 0,
            "Optimization history should not be empty"
        );

        // 最適化後のグラフがProgramRootを持つことを確認
        // (完全にloweringが完了した場合)
        if let Some(root) = optimized.program_root() {
            assert!(
                matches!(root.op, GraphOp::ProgramRoot { .. }),
                "Root should be ProgramRoot"
            );
        }
    }

    #[test]
    fn test_greedy_optimizer_beam_width_is_one() {
        // 貪欲法オプティマイザのビーム幅が1であることを間接的に確認
        // ビーム幅=1なので、各ステップで1つの候補のみ保持される
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        let config = MultiPhaseConfig::new()
            .with_max_steps(100)
            .with_progress(false);
        let optimizer = create_greedy_optimizer(config);

        let (optimized, _history) = optimizer.optimize_with_history(graph);

        // 最適化が完了することを確認（パニックしない）
        let _outputs = optimized.outputs();
    }

    #[test]
    fn test_multi_phase_vs_greedy() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        // 両方とも最適化が完了することを確認
        let multi_phase_config = MultiPhaseConfig::new()
            .with_beam_width(4)
            .with_max_steps(1000);
        let (_multi_phase_result, _) =
            optimize_graph_multi_phase(graph.clone(), multi_phase_config);

        let (_greedy_result, _) = optimize_graph_greedy(graph, 1000);

        // 両方とも正常に完了（パニックしない）
    }
}
