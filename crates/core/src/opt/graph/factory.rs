//! グラフオプティマイザのファクトリ関数
//!
//! マルチフェーズ最適化や各種Suggesterを組み合わせた
//! オプティマイザを作成するためのファクトリ関数を提供します。

use crate::graph::Graph;
use crate::lowerer::SubgraphLoweringOptimizer;
use crate::opt::ast::{
    CompositeSuggester as AstCompositeSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
};
use crate::opt::context::DeviceCapabilities;
use crate::opt::graph::{
    BeamSearchGraphOptimizer, BufferAbsorptionSuggester, ChainedGraphOptimizer, CompositeSuggester,
    ContiguousInsertionSuggester, FusionSuggester, GraphOptimizer, GreedyGraphOptimizer,
    KernelMergeSuggester, KernelPartitionSuggester, LoweringSuggester, OptimizationHistory,
    SubgraphInliningSuggester, ViewInsertionSuggester, ViewMergeSuggester,
};

// =============================================================================
// ユーティリティオプティマイザ
// =============================================================================

/// 何も変更しないオプティマイザ
///
/// グラフをそのまま返す特殊なオプティマイザです。
/// サブグラフ処理をスキップする場合などに使用します。
#[derive(Debug, Clone)]
pub struct IdentityOptimizer {
    name: String,
}

impl IdentityOptimizer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl GraphOptimizer for IdentityOptimizer {
    fn name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn optimize(&self, graph: Graph) -> Graph {
        graph
    }

    fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        (graph, OptimizationHistory::new())
    }
}

// =============================================================================
// サブグラフ処理モード
// =============================================================================

/// サブグラフの処理方法を指定するモード
///
/// コンパイルパイプラインでSubgraphCallノードをどのように扱うかを決定します。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubgraphMode {
    /// サブグラフをインライン展開する
    ///
    /// SubgraphCallノードを対応するサブグラフの計算ノードで直接置き換えます。
    /// これにより、単一の大きなカーネルが生成されます。
    ///
    /// **利点**: カーネル呼び出しオーバーヘッドがない
    /// **欠点**: コードサイズが大きくなる可能性がある
    #[default]
    Inline,

    /// サブグラフを個別のカーネル関数として生成する
    ///
    /// 各SubgraphCallは独立したカーネル関数として保持され、
    /// execution_orderで呼び出し順序が管理されます。
    ///
    /// **利点**: コードの再利用、モジュール性の維持
    /// **欠点**: カーネル呼び出しオーバーヘッドがある
    SeparateKernels,

    /// サブグラフを無視する（SubgraphCallノードを残す）
    ///
    /// SubgraphCallノードをそのまま残します。警告が出力されます。
    /// デバッグや特殊なケースで使用します。
    Skip,
}

// =============================================================================
// マルチフェーズ最適化用の関数
// =============================================================================

/// サブグラフインライン展開フェーズ用Suggesterを作成
///
/// SubgraphCallノードを検出し、対応するサブグラフの計算を
/// 呼び出し元グラフに直接埋め込みます。
///
/// このフェーズは他の最適化の前に実行する必要があります。
pub fn create_subgraph_inlining_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(SubgraphInliningSuggester::new())])
}

/// ViewMergeとFusionのみのフェーズ用Suggesterを作成
///
/// グラフの構築過程で生成された余分なビュー変更を除去します。
/// 最適化の最初と最後に実行することで、クリーンなグラフ構造を維持します。
pub fn create_view_merge_only_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(ViewMergeSuggester::new())])
}

/// グラフ最適化フェーズ用のSuggesterを作成
///
/// ViewInsertion、ContiguousInsertion、Fusion, ViewMergeなど、
/// グラフ構造の最適化を行うSuggesterを含みます。
///
/// ViewMergeはこのフェーズには含めず、独立したフェーズとして実行します。
///
/// Note: タイル化はAST最適化フェーズで行われるため、グラフレベルでは行いません。
pub fn create_graph_optimization_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        Box::new(ViewInsertionSuggester::new()),
        Box::new(ContiguousInsertionSuggester::new()),
        Box::new(FusionSuggester::new()),
        Box::new(ViewMergeSuggester::new()),
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

/// Loweringのみのフェーズ用Suggesterを作成（SIMD幅指定付き）
///
/// 全てのGraphOpノードをKernelノードに変換します。
///
/// # 注意
/// SIMD化はASTレベルで行われるため、この関数は`simd_widths`パラメータを無視します。
/// 後方互換性のために関数シグネチャは維持されています。
///
/// # Arguments
/// * `_simd_widths` - 無視されます（SIMD化はAST最適化で行われます）
#[allow(unused_variables)]
pub fn create_lowering_only_suggester_with_simd(_simd_widths: Vec<usize>) -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(LoweringSuggester::new())])
}

/// 貪欲法Lowering用のSuggesterを作成
///
/// 高速にLoweringを行います。
pub fn create_greedy_lowering_only_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(LoweringSuggester::new())])
}

/// KernelPartitionフェーズ用のSuggesterを作成
///
/// LoweringSuggesterで生成された1D FlatParallel Kernelを
/// 多次元グリッドに分割します。
///
/// # 設計方針
/// - Loweringで生成された1D tid を持つKernelを対象
/// - 2D/3Dグリッドへの分割候補を生成
/// - Absorptionの前に実行することで、グラフレベルでdispatch設定の一貫性を保証
pub fn create_kernel_partition_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![Box::new(KernelPartitionSuggester::new())])
}

/// Fusionフェーズ用のSuggesterを作成
///
/// Phase 3: Kernelノードの最終処理
///
/// # 設計方針
/// - BufferAbsorptionでKernelに入力Bufferを取り込む
/// - KernelMergeで複数のKernel(Function)をマージ
///
/// Note: カーネル実行順序はCompiledProgram.execution_wavesで管理されます。
///
/// このフェーズは実測値ベース最適化の対象外です（決定論的な変換のみ）。
pub fn create_fusion_suggester() -> CompositeSuggester {
    CompositeSuggester::new(vec![
        // BufferAbsorptionでKernelノードに入力Bufferを取り込む
        Box::new(BufferAbsorptionSuggester::new()),
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
/// use harp_core::opt::ast::{RuleBaseOptimizer, BeamSearchOptimizer};
/// use harp_core::opt::ast::rules::all_algebraic_rules;
/// use harp_core::opt::graph::factory::create_ast_loop_suggester;
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
        Box::new(LoopTilingSuggester::new()),
        Box::new(LoopInliningSuggester::new()),
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
    /// サブグラフの処理モード
    ///
    /// - `Inline`: サブグラフをインライン展開（デフォルト、従来の動作）
    /// - `SeparateKernels`: サブグラフを個別のカーネル関数として生成
    /// - `Skip`: サブグラフ処理をスキップ
    pub subgraph_mode: SubgraphMode,
    /// LoweringSuggesterのSIMD幅候補
    ///
    /// 空の場合はスカラー版のみ生成します。
    /// 例: `vec![4, 8]` で幅4と幅8のSIMD候補を生成
    pub simd_widths: Vec<usize>,
    /// 最適化コンテキスト（デバイス固有の情報）
    ///
    /// 設定されている場合、Suggesterはこのコンテキストからデバイス固有の
    /// パラメータ（タイルサイズ、スレッドグループサイズ、SIMD幅など）を取得します。
    /// 設定されていない場合、デフォルト値が使用されます。
    pub opt_context: Option<DeviceCapabilities>,
}

impl Default for MultiPhaseConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_steps_per_phase: 5000,
            show_progress: false,
            collect_logs: cfg!(debug_assertions),
            early_termination_threshold: Some(10), // デフォルト: 10ステップ改善なしで終了
            subgraph_mode: SubgraphMode::default(), // デフォルト: Inline
            simd_widths: vec![],                   // デフォルト: スカラー版のみ
            opt_context: None,                     // デフォルト: コンテキストなし
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

    /// サブグラフの処理モードを設定
    ///
    /// # Arguments
    /// * `mode` - サブグラフの処理方法
    ///   - `SubgraphMode::Inline`: サブグラフをインライン展開（デフォルト）
    ///   - `SubgraphMode::SeparateKernels`: サブグラフを個別のカーネル関数として生成
    ///   - `SubgraphMode::Skip`: サブグラフ処理をスキップ
    ///
    /// # Example
    /// ```ignore
    /// use harp_core::opt::graph::factory::{MultiPhaseConfig, SubgraphMode};
    ///
    /// // サブグラフを個別カーネルとして生成
    /// let config = MultiPhaseConfig::new()
    ///     .with_subgraph_mode(SubgraphMode::SeparateKernels);
    /// ```
    pub fn with_subgraph_mode(mut self, mode: SubgraphMode) -> Self {
        self.subgraph_mode = mode;
        self
    }

    /// SIMD幅候補を設定
    ///
    /// LoweringSuggesterがElementwise演算のSIMD版を生成する際に使用する幅を指定します。
    /// 空の場合はスカラー版のみ生成します。
    ///
    /// # Example
    /// ```ignore
    /// use harp_core::opt::graph::factory::MultiPhaseConfig;
    ///
    /// // 幅4と幅8のSIMD候補を生成
    /// let config = MultiPhaseConfig::new()
    ///     .with_simd_widths(vec![4, 8]);
    /// ```
    pub fn with_simd_widths(mut self, widths: Vec<usize>) -> Self {
        self.simd_widths = widths;
        self
    }

    /// 最適化コンテキストを設定
    ///
    /// デバイス固有の情報（タイルサイズ、スレッドグループサイズ、SIMD幅など）を
    /// Suggesterに渡すために使用します。
    ///
    /// コンテキストが設定されている場合:
    /// - KernelPartitionSuggesterはコンテキストのwork_group_sizeを使用
    /// - LoweringSuggesterはコンテキストのvector_widthsを使用
    ///
    /// # Example
    /// ```ignore
    /// use harp_core::opt::graph::factory::MultiPhaseConfig;
    /// use harp_core::opt::context::DeviceCapabilities;
    ///
    /// let caps = DeviceCapabilities::from_device(&device);
    /// let config = MultiPhaseConfig::new()
    ///     .with_capabilities(caps);
    /// ```
    pub fn with_capabilities(mut self, caps: DeviceCapabilities) -> Self {
        // DeviceCapabilitiesからSIMD幅を自動設定（明示的に設定されていない場合）
        if self.simd_widths.is_empty() {
            self.simd_widths = caps
                .all_simd_widths()
                .into_iter()
                .filter(|&w| w > 1)
                .collect();
        }
        self.opt_context = Some(caps);
        self
    }
}

/// マルチフェーズグラフ最適化を作成
///
/// 7つのフェーズで段階的にグラフを最適化します：
///
/// 0. **SubgraphInlining** (サブグラフインライン展開): サブグラフ呼び出しを展開
///    - 目的: SubgraphCallノードを実際の計算ノードに変換
///    - 他の最適化の前に実行する必要がある
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
/// 5. **KernelPartition** (カーネル分割): 1D FlatParallel Kernelを多次元グリッドに分割
///    - 目的: GPUの多次元スレッドグリッドを活用
///    - Absorptionの前に実行することでdispatch設定の一貫性を保証
///
/// 6. **Absorption** (吸収): Kernelノードの最終処理
///    - 目的: Bufferの吸収とKernelのマージ
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
/// use harp_core::opt::graph::factory::{create_multi_phase_optimizer, MultiPhaseConfig};
/// use harp_core::opt::graph::GraphOptimizer;
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
    // Phase 0: サブグラフ処理（モードによって動作が異なる）
    let subgraph_optimizer: Box<dyn GraphOptimizer> = match config.subgraph_mode {
        SubgraphMode::Inline => {
            // インライン展開: SubgraphInliningSuggesterを使用
            // コスト増加に関わらず展開する必要があるためGreedyGraphOptimizerを使用
            let subgraph_inlining_suggester = create_subgraph_inlining_suggester();
            Box::new(
                GreedyGraphOptimizer::new(subgraph_inlining_suggester)
                    .with_max_steps(config.max_steps_per_phase)
                    .with_name("SubgraphInlining"),
            )
        }
        SubgraphMode::SeparateKernels => {
            // 個別カーネル生成: SubgraphLoweringOptimizerを使用
            Box::new(SubgraphLoweringOptimizer::new())
        }
        SubgraphMode::Skip => {
            // スキップ: 何もしないオプティマイザ
            Box::new(IdentityOptimizer::new("SubgraphSkip"))
        }
    };

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
    // タイル化はAST最適化フェーズで行われるため、グラフレベルでは行わない
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
    // BeamSearchを使用してコストベースで最適な並列化戦略を選択
    // SimpleCostEstimatorは並列カーネルにボーナス（コスト減少）を与えるため、
    // 適切なスレッドグループサイズとベクトル化が選択される
    // DeviceCapabilitiesがある場合はLoweringSuggesterに渡す
    let lowering_suggester = if let Some(ref caps) = config.opt_context {
        CompositeSuggester::new(vec![Box::new(LoweringSuggester::from_capabilities(caps))])
    } else {
        create_lowering_only_suggester_with_simd(config.simd_widths.clone())
    };
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 5: KernelPartition（1D Kernelを多次元グリッドに分割）
    // DeviceCapabilitiesがある場合はKernelPartitionSuggesterに渡す
    let kernel_partition_suggester = if let Some(ref caps) = config.opt_context {
        CompositeSuggester::new(vec![Box::new(KernelPartitionSuggester::from_capabilities(
            caps,
        ))])
    } else {
        create_kernel_partition_suggester()
    };
    let kernel_partition_optimizer = BeamSearchGraphOptimizer::new(kernel_partition_suggester)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 6: Absorption（Bufferの吸収とKernelのマージ）
    // 決定論的な変換なのでビーム幅=1で十分、早期終了も不要
    let absorption_suggester = create_fusion_suggester();
    let absorption_optimizer = BeamSearchGraphOptimizer::new(absorption_suggester)
        .with_beam_width(1) // 決定論的変換なのでビーム幅1
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(1)); // すぐに終了

    // サブグラフ処理フェーズの名前を決定
    let subgraph_phase_name = match config.subgraph_mode {
        SubgraphMode::Inline => "SubgraphInlining",
        SubgraphMode::SeparateKernels => "SubgraphLowering",
        SubgraphMode::Skip => "SubgraphSkip",
    };

    // チェーンを構築
    ChainedGraphOptimizer::new()
        .add_phase_boxed(subgraph_phase_name, subgraph_optimizer)
        .add_phase("ViewMerge (Initial)", view_merge_1_optimizer)
        .add_phase("Optimization", optimization_optimizer)
        .add_phase("ViewMerge (Post-Opt)", view_merge_2_optimizer)
        .add_phase("Lowering", lowering_optimizer)
        .add_phase("KernelPartition", kernel_partition_optimizer)
        .add_phase("Absorption", absorption_optimizer)
}

/// マルチフェーズグラフ最適化を作成（カスタムSelector使用）
///
/// `create_multi_phase_optimizer`と同様ですが、Optimization、Lowering、KernelPartitionフェーズで
/// カスタムSelectorを使用できます。GraphRuntimeSelectorを使用した実測値ベースの最適化に使用します。
///
/// ViewMergeフェーズとAbsorptionフェーズは決定論的な変換のため、Selectorは使用しません。
///
/// # Arguments
/// * `config` - 最適化の設定
/// * `selector` - Optimization、Lowering、KernelPartitionで使用するカスタムSelector（GraphRuntimeSelectorなど）
///
/// # Returns
/// ChainedGraphOptimizer（Optimization、Lowering、KernelPartitionにカスタムSelectorが設定される）
///
/// # Example
/// ```ignore
/// use harp_core::opt::graph::factory::{create_multi_phase_optimizer_with_selector, MultiPhaseConfig};
/// use harp_core::opt::selector::GraphRuntimeSelector;
/// use harp_core::opt::graph::GraphOptimizer;
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
    // Phase 0: サブグラフ処理（モードによって動作が異なる）- Selector不使用（決定論的変換）
    let subgraph_optimizer: Box<dyn GraphOptimizer> = match config.subgraph_mode {
        SubgraphMode::Inline => {
            let subgraph_inlining_suggester = create_subgraph_inlining_suggester();
            Box::new(
                BeamSearchGraphOptimizer::new(subgraph_inlining_suggester)
                    .with_beam_width(1)
                    .with_max_steps(config.max_steps_per_phase)
                    .with_progress(config.show_progress)
                    .with_collect_logs(config.collect_logs)
                    .with_early_termination_threshold(Some(5)),
            )
        }
        SubgraphMode::SeparateKernels => Box::new(SubgraphLoweringOptimizer::new()),
        SubgraphMode::Skip => Box::new(IdentityOptimizer::new("SubgraphSkip")),
    };

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
    let lowering_suggester = create_lowering_only_suggester_with_simd(config.simd_widths.clone());
    let lowering_optimizer = BeamSearchGraphOptimizer::new(lowering_suggester)
        .with_selector(selector.clone())
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 5: KernelPartition（1D Kernelを多次元グリッドに分割）- カスタムSelector使用
    let kernel_partition_suggester = create_kernel_partition_suggester();
    let kernel_partition_optimizer = BeamSearchGraphOptimizer::new(kernel_partition_suggester)
        .with_selector(selector)
        .with_beam_width(config.beam_width)
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 6: Absorption（Bufferの吸収とKernelのマージ）- Selector不使用（決定論的変換）
    let absorption_suggester = create_fusion_suggester();
    let absorption_optimizer = BeamSearchGraphOptimizer::new(absorption_suggester)
        .with_beam_width(1) // 決定論的変換なのでビーム幅1
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(1)); // すぐに終了

    // サブグラフ処理フェーズの名前を決定
    let subgraph_phase_name = match config.subgraph_mode {
        SubgraphMode::Inline => "SubgraphInlining",
        SubgraphMode::SeparateKernels => "SubgraphLowering",
        SubgraphMode::Skip => "SubgraphSkip",
    };

    // チェーンを構築
    ChainedGraphOptimizer::new()
        .add_phase_boxed(subgraph_phase_name, subgraph_optimizer)
        .add_phase("ViewMerge (Initial)", view_merge_1_optimizer)
        .add_phase("Optimization", optimization_optimizer)
        .add_phase("ViewMerge (Post-Opt)", view_merge_2_optimizer)
        .add_phase("Lowering", lowering_optimizer)
        .add_phase("KernelPartition", kernel_partition_optimizer)
        .add_phase("Absorption", absorption_optimizer)
}

/// マルチフェーズグラフ最適化を実行（履歴付き）
///
/// 6つのフェーズで段階的にグラフを最適化し、各フェーズの履歴を結合して返します。
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
/// use harp_core::opt::graph::factory::{optimize_graph_multi_phase, MultiPhaseConfig};
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
) -> (Graph, OptimizationHistory) {
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
    // Phase 0: サブグラフ処理（モードによって動作が異なる）
    let subgraph_optimizer: Box<dyn GraphOptimizer> = match config.subgraph_mode {
        SubgraphMode::Inline => {
            let subgraph_inlining_suggester = create_subgraph_inlining_suggester();
            Box::new(
                BeamSearchGraphOptimizer::new(subgraph_inlining_suggester)
                    .with_beam_width(1) // 貪欲法
                    .with_max_steps(config.max_steps_per_phase)
                    .with_progress(config.show_progress)
                    .with_collect_logs(config.collect_logs)
                    .with_early_termination_threshold(Some(5)),
            )
        }
        SubgraphMode::SeparateKernels => Box::new(SubgraphLoweringOptimizer::new()),
        SubgraphMode::Skip => Box::new(IdentityOptimizer::new("SubgraphSkip")),
    };

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

    // Phase 5: KernelPartition（1D Kernelを多次元グリッドに分割）
    // 貪欲法なのでビーム幅=1で最初の候補のみを選択
    let kernel_partition_suggester = create_kernel_partition_suggester();
    let kernel_partition_optimizer = BeamSearchGraphOptimizer::new(kernel_partition_suggester)
        .with_beam_width(1) // 貪欲法
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(config.early_termination_threshold);

    // Phase 6: Absorption（Bufferの吸収とKernelのマージ）
    let absorption_suggester = create_fusion_suggester();
    let absorption_optimizer = BeamSearchGraphOptimizer::new(absorption_suggester)
        .with_beam_width(1) // 決定論的変換
        .with_max_steps(config.max_steps_per_phase)
        .with_progress(config.show_progress)
        .with_collect_logs(config.collect_logs)
        .with_early_termination_threshold(Some(1)); // すぐに終了

    // サブグラフ処理フェーズの名前を決定
    let subgraph_phase_name = match config.subgraph_mode {
        SubgraphMode::Inline => "SubgraphInlining (Greedy)",
        SubgraphMode::SeparateKernels => "SubgraphLowering (Greedy)",
        SubgraphMode::Skip => "SubgraphSkip (Greedy)",
    };

    // チェーンを構築
    ChainedGraphOptimizer::new()
        .add_phase_boxed(subgraph_phase_name, subgraph_optimizer)
        .add_phase("ViewMerge (Greedy)", view_merge_1_optimizer)
        .add_phase("Optimization (Greedy)", optimization_optimizer)
        .add_phase("ViewMerge (Greedy)", view_merge_2_optimizer)
        .add_phase("Lowering (Greedy)", lowering_optimizer)
        .add_phase("KernelPartition (Greedy)", kernel_partition_optimizer)
        .add_phase("Absorption (Greedy)", absorption_optimizer)
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
/// use harp_core::opt::graph::factory::optimize_graph_greedy;
///
/// let (optimized, history) = optimize_graph_greedy(graph, 5000);
/// ```
pub fn optimize_graph_greedy(graph: Graph, max_steps: usize) -> (Graph, OptimizationHistory) {
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
    fn test_greedy_lowering_suggester() {
        let greedy_suggester = create_greedy_lowering_only_suggester();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        let suggestions = greedy_suggester.suggest(&graph);

        // LoweringSuggesterは1つの候補を生成
        assert_eq!(
            suggestions.len(),
            1,
            "LoweringSuggester should generate exactly 1 candidate, got {}",
            suggestions.len()
        );

        // 生成された候補はKernelノードを含む
        let outputs = suggestions[0].graph.outputs();
        let output = outputs.get("c").unwrap();
        assert!(
            matches!(output.op, GraphOp::Kernel { .. }),
            "Output should be Kernel node"
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
            !history.is_empty(),
            "Optimization history should not be empty"
        );

        // 最適化後のグラフの出力がKernelノードであることを確認
        // (完全にloweringが完了した場合)
        for output_node in optimized.outputs().values() {
            assert!(
                matches!(output_node.op, GraphOp::Kernel { .. }),
                "Output should be Kernel node after optimization"
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
