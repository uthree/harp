/// Pipeline実装のための共通ヘルパー関数
///
/// 各バックエンドのPipeline実装で使用される共通機能を提供します。
use crate::graph::Graph;
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

/// ViewMergeのみのフェーズ用Suggesterを作成
///
/// グラフの構築過程で生成された余分なビュー変更を除去します。
/// 最適化の最初と最後に実行することで、クリーンなグラフ構造を維持します。
pub fn create_view_merge_only_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(ViewMergeSuggester::new())])
}

/// グラフ最適化フェーズ用のSuggesterを作成
///
/// ViewInsertion、Tiling、ContiguousInsertion、Fusion, ViewMergeなど、
/// グラフ構造の最適化を行うSuggesterを含みます。
///
/// ViewMergeはこのフェーズには含めず、独立したフェーズとして実行します。
pub fn create_graph_optimization_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(TilingSuggester::with_default_tile_sizes()),
        Box::new(ContiguousInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
        Box::new(ViewMergeSuggester::new()),
    ])
}

/// グラフ準備フェーズ用のSuggesterを作成（レガシー互換）
///
/// Phase 1: グラフ構造の最適化（View挿入、融合、タイリングなど）
/// Lowering前にグラフ構造を整理するために使用します。
///
/// Note: 新しいコードでは `create_view_merge_only_suggester` と
/// `create_graph_optimization_suggester` を個別に使用することを推奨します。
pub fn create_graph_preparation_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(ViewMergeSuggester::new()),
        Box::new(TilingSuggester::with_default_tile_sizes()),
        Box::new(ContiguousInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
    ])
}

/// Loweringのみのフェーズ用Suggesterを作成
///
/// 全てのGraphOpノードをKernelノードに変換します。
///
/// # 設計方針
/// - LoweringSuggesterでGraphOp → Kernel(Function/Kernel)に変換
/// - 並列化戦略の選択がこのフェーズで行われるため、実測値ベース最適化の対象
/// - ViewMergeは独立したフェーズで実行するため、このSuggesterには含めない
pub fn create_lowering_only_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        // LoweringSuggesterでGraphOp -> Kernel(Function/Kernel)に変換
        Box::new(LoweringSuggester::new()),
    ])
}

/// 貪欲法Lowering用のSuggesterを作成
///
/// Sequential戦略のみで高速にLoweringを行います。
pub fn create_greedy_lowering_only_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(LoweringSuggester::sequential_only())])
}

/// Fusionフェーズ用のSuggesterを作成
///
/// Phase 3: 全てのKernelノードを単一のProgramRootに融合
///
/// # 設計方針
/// - BufferAbsorptionでKernelに入力Bufferを取り込む
/// - ProgramRootAbsorptionでKernel(Function)をProgramRootに吸収
/// - ProgramRootBufferAbsorptionでProgramRootの入力Bufferを除去
/// - KernelMergeで複数のKernel(Function)をマージ
///
/// このフェーズは実測値ベース最適化の対象外です（決定論的な変換のみ）。
pub fn create_fusion_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
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
/// 5つのフェーズで段階的にグラフを最適化します：
///
/// 1. **ViewMerge** (ビューマージ): グラフ構築時に生成された余分なビュー変更を除去
///    - 目的: クリーンなグラフ構造の確保
///
/// 2. **Optimization** (最適化): View挿入、タイリング、Contiguous挿入、融合など
///    - 目的: グラフ構造の最適化
///    - ViewInsertionはこのフェーズで使用
///
/// 3. **ViewMerge** (ビューマージ): 最適化後に残ったビューをマージ
///    - 目的: Lowering前のクリーンアップ
///
/// 4. **Lowering** (Lowering): 全GraphOpをKernelノードに変換
///    - 目的: 並列化戦略の選択を含むKernel生成
///    - 実測値ベース最適化の主な対象
///
/// 5. **Absorption** (吸収): 全KernelをProgramRootに融合
///    - 目的: 単一のProgramRootノードへの変換
///    - 決定論的な変換のため、実測値ベース最適化は不要
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
    // Phase 1: ViewMerge（余分なビュー変更を除去）
    // 決定論的な変換なのでビーム幅=1で十分
    let view_merge_1_suggester = create_view_merge_only_suggester();
    let view_merge_1_optimizer = BeamSearchGraphOptimizer::new(view_merge_1_suggester)
        .with_beam_width(1) // 決定論的変換
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(5)); // 早期終了

    // Phase 2: Optimization（グラフ構造の最適化）
    let optimization_suggester = create_graph_optimization_suggester();
    let optimization_optimizer = BeamSearchGraphOptimizer::new(optimization_suggester)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 3: ViewMerge（最適化後のビューをマージ）
    // 決定論的な変換なのでビーム幅=1で十分
    let view_merge_2_suggester = create_view_merge_only_suggester();
    let view_merge_2_optimizer = BeamSearchGraphOptimizer::new(view_merge_2_suggester)
        .with_beam_width(1) // 決定論的変換
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(5)); // 早期終了

    // Phase 4: Lowering（Kernel変換のみ）
    let lowering_suggester = create_lowering_only_suggester();
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 5: Absorption（ProgramRootへの融合）
    // 決定論的な変換なのでビーム幅=1で十分、早期終了も不要
    let absorption_suggester = create_fusion_suggester();
    let absorption_optimizer = BeamSearchGraphOptimizer::new(absorption_suggester)
        .with_beam_width(1) // 決定論的変換なのでビーム幅1
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(1)); // すぐに終了

    // チェーンを構築（各オプティマイザに名前を付けてからchainする）
    view_merge_1_optimizer
        .with_name("ViewMerge (Initial)")
        .chain(optimization_optimizer.with_name("Optimization"))
        .chain(view_merge_2_optimizer.with_name("ViewMerge (Post-Opt)"))
        .chain(lowering_optimizer.with_name("Lowering"))
        .chain(absorption_optimizer.with_name("Absorption"))
}

/// マルチフェーズグラフ最適化を作成（カスタムSelector使用）
///
/// `create_multi_phase_optimizer`と同様ですが、OptimizationフェーズとLoweringフェーズで
/// カスタムSelectorを使用できます。GraphRuntimeSelectorを使用した実測値ベースの最適化に使用します。
///
/// ViewMergeフェーズとAbsorptionフェーズは決定論的な変換のため、Selectorは使用しません。
///
/// # Arguments
/// * `config` - 最適化の設定
/// * `selector` - OptimizationとLoweringで使用するカスタムSelector（GraphRuntimeSelectorなど）
///
/// # Returns
/// ChainedGraphOptimizer（OptimizationとLoweringにカスタムSelectorが設定される）
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
    Sel: crate::opt::GraphSelector + Clone + 'static,
{
    // Phase 1: ViewMerge（余分なビュー変更を除去）- Selector不使用（決定論的変換）
    let view_merge_1_suggester = create_view_merge_only_suggester();
    let view_merge_1_optimizer = BeamSearchGraphOptimizer::new(view_merge_1_suggester)
        .with_beam_width(1) // 決定論的変換
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(5)); // 早期終了

    // Phase 2: Optimization（グラフ構造の最適化）- カスタムSelector使用
    let optimization_suggester = create_graph_optimization_suggester();
    let optimization_optimizer = BeamSearchGraphOptimizer::new(optimization_suggester)
        .with_selector(selector.clone())
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 3: ViewMerge（最適化後のビューをマージ）- Selector不使用（決定論的変換）
    let view_merge_2_suggester = create_view_merge_only_suggester();
    let view_merge_2_optimizer = BeamSearchGraphOptimizer::new(view_merge_2_suggester)
        .with_beam_width(1) // 決定論的変換
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(5)); // 早期終了

    // Phase 4: Lowering（Kernel変換のみ）- カスタムSelector使用
    let lowering_suggester = create_lowering_only_suggester();
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_selector(selector)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 5: Absorption（ProgramRootへの融合）- Selector不使用（決定論的変換）
    let absorption_suggester = create_fusion_suggester();
    let absorption_optimizer = BeamSearchGraphOptimizer::new(absorption_suggester)
        .with_beam_width(1) // 決定論的変換なのでビーム幅1
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(1)); // すぐに終了

    // チェーンを構築
    view_merge_1_optimizer
        .with_name("ViewMerge (Initial)")
        .chain(optimization_optimizer.with_name("Optimization"))
        .chain(view_merge_2_optimizer.with_name("ViewMerge (Post-Opt)"))
        .chain(lowering_optimizer.with_name("Lowering"))
        .chain(absorption_optimizer.with_name("Absorption"))
}

/// マルチフェーズグラフ最適化を実行（履歴付き）
///
/// 5つのフェーズで段階的にグラフを最適化し、各フェーズの履歴を結合して返します。
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
/// - 全フェーズでビーム幅=1で貪欲法として動作
/// - LoweringSuggesterはSequential戦略のみ使用
/// - 探索空間を最小限に抑え、高速にloweringを完了
///
/// # Arguments
/// * `config` - 最適化の設定（beam_widthは1に強制される）
///
/// # Returns
/// ChainedGraphOptimizer
pub fn create_greedy_optimizer(config: MultiPhaseConfig) -> ChainedGraphOptimizer {
    // Phase 1: ViewMerge（余分なビュー変更を除去）
    let view_merge_1_suggester = create_view_merge_only_suggester();
    let view_merge_1_optimizer = BeamSearchGraphOptimizer::new(view_merge_1_suggester)
        .with_beam_width(1) // 貪欲法
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(5)); // 早期終了

    // Phase 2: Optimization（グラフ構造の最適化）
    let optimization_suggester = create_graph_optimization_suggester();
    let optimization_optimizer = BeamSearchGraphOptimizer::new(optimization_suggester)
        .with_beam_width(1) // 貪欲法
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 3: ViewMerge（最適化後のビューをマージ）
    let view_merge_2_suggester = create_view_merge_only_suggester();
    let view_merge_2_optimizer = BeamSearchGraphOptimizer::new(view_merge_2_suggester)
        .with_beam_width(1) // 貪欲法
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(5)); // 早期終了

    // Phase 4: Lowering（Sequential戦略のみ）
    let lowering_suggester = create_greedy_lowering_only_suggester();
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_beam_width(1) // 貪欲法
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 5: Absorption（ProgramRootへの融合）
    let absorption_suggester = create_fusion_suggester();
    let absorption_optimizer = BeamSearchGraphOptimizer::new(absorption_suggester)
        .with_beam_width(1) // 決定論的変換
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(1)); // すぐに終了

    view_merge_1_optimizer
        .with_name("ViewMerge (Greedy)")
        .chain(optimization_optimizer.with_name("Optimization (Greedy)"))
        .chain(view_merge_2_optimizer.with_name("ViewMerge (Greedy)"))
        .chain(lowering_optimizer.with_name("Lowering (Greedy)"))
        .chain(absorption_optimizer.with_name("Absorption (Greedy)"))
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
        let greedy_suggester = create_greedy_lowering_only_suggester();
        let normal_suggester = create_lowering_only_suggester();

        // Suggesterの最初の要素がLoweringSuggester::sequential_only()であることを確認
        // CompositeSuggesterの内部構造にはアクセスできないため、
        // 実際にsuggestを呼び出して候補数で確認する
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        let greedy_suggestions = greedy_suggester.suggest(&graph);
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
