use crate::ast::AstNode;
use crate::backend::{Buffer, Compiler, KernelSignature, Pipeline, Renderer};
use crate::graph::Graph;
use crate::opt::ast::rules::all_algebraic_rules;
use crate::opt::ast::{
    BeamSearchOptimizer as AstBeamSearchOptimizer, CompositeSuggester as AstCompositeSuggester,
    FunctionInliningSuggester, LoopFusionSuggester, LoopInliningSuggester,
    LoopInterchangeSuggester, LoopTilingSuggester, OptimizationHistory as AstOptimizationHistory,
    Optimizer as AstOptimizer, RuleBaseOptimizer,
};
use crate::opt::graph::{GraphOptimizer, OptimizationHistory as GraphOptimizationHistory};
use crate::opt::selector::{GraphRuntimeSelector, RuntimeSelector};
use std::collections::HashMap;

/// KernelSignatureからベンチマーク用バッファを生成
///
/// RuntimeSelector内部で使用されるヘルパー関数。
/// 各入出力バッファに対してBuffer::allocateを呼び出して適切なサイズのバッファを確保します。
/// 計測用なのでデータ型はF32で固定しています（実行時間は型に依存しないため）。
fn create_buffers_from_signature<B: Buffer>(sig: &KernelSignature) -> Vec<B> {
    use crate::ast::DType;

    sig.inputs
        .iter()
        .chain(sig.outputs.iter())
        .map(|buf_sig| {
            let shape: Vec<usize> = buf_sig
                .shape
                .iter()
                .map(|d| {
                    d.expect_const("buffer shape must be const for runtime measurement") as usize
                })
                .collect();
            // 計測用なので型はF32で固定
            B::allocate(shape, DType::F32)
        })
        .collect()
}

/// compile_graph_with_all_historiesの戻り値の型
type CompileWithHistoriesResult<K> =
    Result<(K, AstNode, HashMap<String, AstOptimizationHistory>), String>;

/// 最適化の設定（グラフとASTで共通）
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// ビーム幅
    pub beam_width: usize,
    /// 最大ステップ数
    pub max_steps: usize,
    /// プログレスバーを表示するか
    pub show_progress: bool,
    /// 早期終了の閾値（改善なしステップ数）
    ///
    /// Some(n): n回連続で改善がなければ終了
    /// None: 早期終了を無効化
    pub early_termination_threshold: Option<usize>,
    /// RuntimeSelector（実測値ベース最適化）を有効にするか
    ///
    /// `true`に設定すると、実測値ベースの候補選択が有効になります。
    /// バッファは`Buffer::allocate`を使用して自動的に生成されます。
    pub enable_runtime_selector: bool,
    /// 足切り候補数（RuntimeSelector使用時）
    ///
    /// RuntimeSelectorを使用する際、静的コストで上位何件に絞り込むかを指定。
    /// 絞り込み後の候補に対して実行時間を計測する。
    pub pre_filter_count: usize,
    /// 計測回数（RuntimeSelector使用時）
    ///
    /// RuntimeSelectorを使用する際、各候補の実行時間を何回計測して平均を取るか。
    pub measurement_count: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_steps: 10000,
            show_progress: false,
            early_termination_threshold: Some(2), // デフォルト: 2ステップ改善なしで終了
            enable_runtime_selector: false,       // デフォルト: 無効
            pre_filter_count: 4,                  // デフォルト: 4件に足切り
            measurement_count: 10,                // デフォルト: 10回計測して平均
        }
    }
}

/// 最適化履歴を管理する構造体
#[derive(Debug, Clone, Default)]
pub struct OptimizationHistories {
    /// グラフ最適化履歴（Phase 1: 一般最適化）
    pub graph: Option<GraphOptimizationHistory>,
    /// グラフ最適化履歴（Phase 2: カーネルマージ）
    pub graph_phase2: Option<GraphOptimizationHistory>,
    /// AST最適化履歴
    pub ast: Option<AstOptimizationHistory>,
}

impl OptimizationHistories {
    /// 全ての履歴をクリア
    pub fn clear(&mut self) {
        self.graph = None;
        self.graph_phase2 = None;
        self.ast = None;
    }

    /// 2段階のグラフ最適化履歴を結合して取得
    ///
    /// Phase 1とPhase 2の履歴をフェーズ名付きで結合します。
    /// 可視化ツールで1つのタイムラインとして表示するために使用します。
    pub fn combined_graph_history(&self) -> Option<GraphOptimizationHistory> {
        match (&self.graph, &self.graph_phase2) {
            (Some(phase1), Some(phase2)) => {
                let mut combined = phase1.clone();
                combined.extend_with_phase(phase2.clone(), "Kernel Merge");
                Some(combined)
            }
            (Some(phase1), None) => Some(phase1.clone()),
            (None, Some(phase2)) => Some(phase2.clone()),
            (None, None) => None,
        }
    }
}

/// 汎用的なPipeline実装
///
/// 任意のRendererとCompilerを組み合わせて使用でき、
/// コンパイル済みのKernelをキャッシュする機能を提供します。
///
/// 最適化履歴の記録機能を持ち、可視化ツールと統合できます。
///
/// # 使用例
/// ```ignore
/// let mut pipeline = GenericPipeline::new(renderer, compiler);
///
/// // 設定のカスタマイズ
/// pipeline.graph_config.beam_width = 8;
///
/// // コンパイル
/// let kernel = pipeline.compile_graph(graph)?;
/// ```
///
/// # RuntimeSelector（実測値ベース最適化）
///
/// `enable_runtime_selector()`を呼び出すと、グラフ最適化とAST最適化の両方で
/// 実測値ベースの候補選択が有効になります。
/// バッファは`Buffer::allocate`を使用して自動的に生成されます。
///
/// # Note
/// グラフ最適化とAST最適化は常に有効です。
pub struct GenericPipeline<R, C>
where
    R: Renderer + Clone + 'static,
    C: Compiler<CodeRepr = R::CodeRepr> + Clone + 'static,
    C::Buffer: 'static,
{
    renderer: R,
    compiler: C,
    /// コンパイル済みKernelのキャッシュ
    kernel_cache: HashMap<String, C::Kernel>,
    /// 最適化履歴
    pub histories: OptimizationHistories,
    /// グラフ最適化の設定
    ///
    /// RuntimeSelectorを使用する場合は`enable_runtime_selector`を`true`に設定してください。
    pub graph_config: OptimizationConfig,
    /// AST最適化の設定
    ///
    /// RuntimeSelectorを使用する場合は`enable_runtime_selector`を`true`に設定してください。
    pub ast_config: OptimizationConfig,
    /// 最適化履歴を収集するか（DEBUGビルドではデフォルトでtrue、RELEASEビルドではfalse）
    pub collect_histories: bool,
}

impl<R, C> GenericPipeline<R, C>
where
    R: Renderer + Clone + 'static,
    C: Compiler<CodeRepr = R::CodeRepr> + Clone + 'static,
    C::Buffer: 'static,
{
    /// 新しいGenericPipelineを作成
    ///
    /// グラフ最適化とAST最適化は常に有効です。
    ///
    /// 最適化履歴の収集は、DEBUGビルドではデフォルトで有効、RELEASEビルドでは無効です。
    pub fn new(renderer: R, compiler: C) -> Self {
        Self {
            renderer,
            compiler,
            kernel_cache: HashMap::new(),
            histories: OptimizationHistories::default(),
            graph_config: OptimizationConfig::default(),
            ast_config: OptimizationConfig::default(),
            collect_histories: cfg!(debug_assertions),
        }
    }

    /// RuntimeSelector（実測値ベース最適化）を有効化
    ///
    /// グラフ最適化とAST最適化の両方で実測値ベースの候補選択が有効になります。
    /// バッファは`Buffer::allocate`を使用して自動的に生成されます。
    ///
    /// # Example
    ///
    /// ```ignore
    /// use harp::backend::GenericPipeline;
    /// use harp::backend::opencl::{OpenCLRenderer, OpenCLCompiler};
    ///
    /// let mut pipeline = GenericPipeline::new(OpenCLRenderer::new(), OpenCLCompiler::new());
    /// pipeline.enable_runtime_selector();
    /// ```
    pub fn enable_runtime_selector(&mut self) {
        self.graph_config.enable_runtime_selector = true;
        self.ast_config.enable_runtime_selector = true;
    }

    /// キャッシュからKernelを取得
    pub fn get_cached_kernel(&self, key: &str) -> Option<&C::Kernel> {
        self.kernel_cache.get(key)
    }

    /// グラフをコンパイルし、結果をキャッシュに保存
    ///
    /// キーが既に存在する場合は上書きされます。
    pub fn compile_and_cache(&mut self, key: String, graph: Graph) -> Result<&C::Kernel, String> {
        let kernel = self.compile_graph(graph)?;
        self.kernel_cache.insert(key.clone(), kernel);
        Ok(self.kernel_cache.get(&key).unwrap())
    }

    /// 最適化履歴を記録しながらグラフをコンパイル
    ///
    /// 最適化が有効な場合、最適化履歴を内部に保存します。
    /// 複数のAST最適化履歴を取得するには、compile_graph_with_all_histories()を使用してください。
    ///
    /// `enable_runtime_selector()`が呼び出されている場合、グラフ最適化とAST最適化の
    /// 両方で自動的に実測値ベースの候補選択が有効になります。
    pub fn compile_graph_with_history(&mut self, graph: Graph) -> Result<C::Kernel, String> {
        // Signatureを作成（最適化前のGraphから）
        let signature = crate::lowerer::create_signature(&graph);

        // グラフ最適化（Phase 1 + Phase 2、RuntimeSelector自動使用）
        let optimized_graph = self.optimize_graph_internal(graph);

        // グラフからAST Programを抽出（Kernel(Program)があれば直接使用）
        let program = crate::lowerer::extract_program(optimized_graph);

        // AST最適化（RuntimeSelector自動使用）
        let (optimized_program, _history) = self.optimize_ast_internal(program, signature.clone());

        // レンダリングとコンパイル
        let code = self.renderer().render(&optimized_program);
        Ok(self.compiler().compile(&code, signature))
    }

    /// 最適化のみを実行（コンパイルなし、AST履歴を返す）
    ///
    /// 最適化が有効な場合、最適化履歴を内部に保存します。
    /// プログラム全体のAST最適化履歴（キー: "program"）と最適化後のProgramを返します。
    /// コンパイルは行わないため、OpenMPなどのランタイムサポートが不要です。
    ///
    /// `enable_runtime_selector()`が呼び出されている場合、グラフ最適化とAST最適化の
    /// 両方で自動的に実測値ベースの候補選択が有効になります。
    pub fn optimize_graph_with_all_histories(
        &mut self,
        graph: Graph,
    ) -> Result<(AstNode, HashMap<String, AstOptimizationHistory>), String> {
        // Signatureを作成（最適化前のGraphから）
        let signature = crate::lowerer::create_signature(&graph);

        // グラフ最適化（Phase 1 + Phase 2、RuntimeSelector自動使用）
        let optimized_graph = self.optimize_graph_internal(graph);

        // グラフからAST Programを抽出（Kernel(Program)があれば直接使用）
        let program = crate::lowerer::extract_program(optimized_graph);

        // AST最適化（Program全体を最適化、RuntimeSelector自動使用）
        let (program, history) = self.optimize_ast_internal(program, signature);

        let mut all_histories = HashMap::new();
        all_histories.insert("program".to_string(), history);

        Ok((program, all_histories))
    }

    /// 最適化履歴を記録しながらグラフをコンパイル（AST履歴を返す）
    ///
    /// 最適化が有効な場合、最適化履歴を内部に保存します。
    /// プログラム全体のAST最適化履歴（キー: "program"）と最適化後のProgramを返します。
    ///
    /// `enable_runtime_selector()`が呼び出されている場合、グラフ最適化とAST最適化の
    /// 両方で自動的に実測値ベースの候補選択が有効になります。
    pub fn compile_graph_with_all_histories(
        &mut self,
        graph: Graph,
    ) -> CompileWithHistoriesResult<C::Kernel> {
        // Signatureを作成（最適化前のGraphから）
        let signature = crate::lowerer::create_signature(&graph);

        // グラフ最適化（Phase 1 + Phase 2、RuntimeSelector自動使用）
        let optimized_graph = self.optimize_graph_internal(graph);

        // グラフからAST Programを抽出（Kernel(Program)があれば直接使用）
        let program = crate::lowerer::extract_program(optimized_graph);

        // AST最適化（Program全体を最適化、RuntimeSelector自動使用）
        let (optimized_program, history) = self.optimize_ast_internal(program, signature.clone());

        let mut all_histories = HashMap::new();
        all_histories.insert("program".to_string(), history);

        // レンダリングとコンパイル
        let code = self.renderer().render(&optimized_program);
        let kernel = self.compiler().compile(&code, signature);
        Ok((kernel, optimized_program, all_histories))
    }

    /// キャッシュをクリア
    pub fn clear_cache(&mut self) {
        self.kernel_cache.clear();
    }

    /// キャッシュサイズを取得
    pub fn cache_size(&self) -> usize {
        self.kernel_cache.len()
    }

    /// 特定のキャッシュエントリを削除
    pub fn remove_cached_kernel(&mut self, key: &str) -> Option<C::Kernel> {
        self.kernel_cache.remove(key)
    }

    /// ループ最適化用のSuggesterを作成（RuleBaseSuggesterを除く）
    ///
    /// 2段階AST最適化の第2段階で使用。
    /// 第1段階でRuleBaseOptimizerが適用済みのため、ループ最適化のみを行う。
    fn create_loop_suggester() -> AstCompositeSuggester {
        AstCompositeSuggester::new(vec![
            Box::new(LoopTilingSuggester::with_default_sizes()),
            Box::new(LoopInliningSuggester::with_default_limit()),
            Box::new(LoopInterchangeSuggester::new()),
            Box::new(LoopFusionSuggester::new()),
            Box::new(FunctionInliningSuggester::with_default_limit()),
        ])
    }

    /// AST最適化用のOptimizerを作成・設定
    fn create_ast_optimizer(
        &self,
        suggester: AstCompositeSuggester,
    ) -> AstBeamSearchOptimizer<AstCompositeSuggester> {
        AstBeamSearchOptimizer::new(suggester)
            .with_beam_width(self.ast_config.beam_width)
            .with_max_steps(self.ast_config.max_steps)
            .with_progress(self.ast_config.show_progress)
            .with_no_improvement_limit(self.ast_config.early_termination_threshold)
    }

    /// グラフ最適化の内部処理（履歴付き）
    ///
    /// マルチフェーズ最適化を使用:
    /// - Phase 1 (Preparation): グラフ構造の最適化（View挿入、融合など）
    /// - Phase 2 (Lowering): Kernel変換、ProgramRoot集約
    ///
    /// `graph_config.enable_runtime_selector`が`true`の場合、
    /// Phase 2でGraphRuntimeSelectorを使用した実測値ベースの候補選択を行います。
    fn optimize_graph_internal(&mut self, graph: Graph) -> Graph {
        use crate::backend::pipeline::{
            MultiPhaseConfig, create_multi_phase_optimizer,
            create_multi_phase_optimizer_with_selector,
        };

        let config = MultiPhaseConfig::new()
            .with_beam_width(self.graph_config.beam_width)
            .with_max_steps(self.graph_config.max_steps)
            .with_progress(self.graph_config.show_progress)
            .with_collect_logs(self.collect_histories);

        // RuntimeSelectorが有効なら実測値ベース最適化
        let (optimized_graph, history) = if self.graph_config.enable_runtime_selector {
            let graph_runtime_selector = GraphRuntimeSelector::new(
                self.renderer.clone(),
                self.compiler.clone(),
                create_buffers_from_signature::<C::Buffer>,
            )
            .with_pre_filter_count(self.graph_config.pre_filter_count)
            .with_measurement_count(self.graph_config.measurement_count);

            let optimizer =
                create_multi_phase_optimizer_with_selector(config, graph_runtime_selector);
            optimizer.optimize_with_history(graph)
        } else {
            let optimizer = create_multi_phase_optimizer(config);
            optimizer.optimize_with_history(graph)
        };

        if self.collect_histories {
            self.histories.graph = Some(history);
            self.histories.graph_phase2 = None; // マルチフェーズでは履歴が統合される
        }

        optimized_graph
    }

    /// AST最適化の内部処理
    ///
    /// 2段階でAST最適化を適用:
    /// 1. ルールベース最適化（RuleBaseOptimizer）: 代数的簡約など（高速、ビームサーチなし）
    /// 2. ループ最適化（BeamSearch）: ループタイリング、融合など
    ///
    /// `ast_config.enable_runtime_selector`が`true`の場合、
    /// 第2段階でRuntimeSelectorを使用した実測値ベースの選択を行います。
    fn optimize_ast_internal(
        &mut self,
        program: AstNode,
        signature: KernelSignature,
    ) -> (AstNode, AstOptimizationHistory) {
        // 第1段階: ルールベース最適化（高速）
        let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules());
        let rule_optimized = rule_optimizer.optimize(program);

        // 第2段階: ループ最適化（ビームサーチ）
        let loop_suggester = Self::create_loop_suggester();

        let (optimized, history) = if self.ast_config.enable_runtime_selector {
            // RuntimeSelectorを使用した実測値ベース最適化
            let runtime_selector = RuntimeSelector::new(
                self.renderer.clone(),
                self.compiler.clone(),
                signature,
                create_buffers_from_signature::<C::Buffer>,
            )
            .with_pre_filter_count(self.ast_config.pre_filter_count)
            .with_measurement_count(self.ast_config.measurement_count);

            let optimizer = AstBeamSearchOptimizer::new(loop_suggester)
                .with_selector(runtime_selector)
                .with_beam_width(self.ast_config.beam_width)
                .with_max_steps(self.ast_config.max_steps)
                .with_progress(self.ast_config.show_progress)
                .with_no_improvement_limit(self.ast_config.early_termination_threshold);

            optimizer.optimize_with_history(rule_optimized)
        } else {
            // 静的コストベース最適化
            let loop_optimizer = self.create_ast_optimizer(loop_suggester);
            loop_optimizer.optimize_with_history(rule_optimized)
        };

        if self.collect_histories {
            self.histories.ast = Some(history.clone());
        }
        (optimized, history)
    }
}

impl<R, C> Pipeline for GenericPipeline<R, C>
where
    R: Renderer + Clone + 'static,
    C: Compiler<CodeRepr = R::CodeRepr> + Clone + 'static,
    C::Buffer: 'static,
{
    type Compiler = C;
    type Renderer = R;
    type Error = String;

    fn renderer(&self) -> &Self::Renderer {
        &self.renderer
    }

    fn compiler(&mut self) -> &mut Self::Compiler {
        &mut self.compiler
    }

    /// グラフ最適化を実行
    ///
    /// マルチフェーズ最適化を適用（常に有効）：
    /// - Phase 1 (Preparation): View挿入、融合など
    /// - Phase 2 (Lowering): GraphOp → Kernel変換
    fn optimize_graph(&self, graph: Graph) -> Graph {
        use crate::backend::pipeline::{MultiPhaseConfig, create_multi_phase_optimizer};

        let config = MultiPhaseConfig::new()
            .with_beam_width(self.graph_config.beam_width)
            .with_max_steps(self.graph_config.max_steps / 2)
            .with_progress(self.graph_config.show_progress)
            .with_collect_logs(false); // &selfなので履歴保存不可

        let optimizer = create_multi_phase_optimizer(config);
        let (optimized_graph, _history) = optimizer.optimize_with_history(graph);

        optimized_graph
    }

    /// プログラム（AST）最適化を実行
    ///
    /// 有効な場合、Program全体に対して2段階の最適化を適用：
    /// 1. ルールベース最適化（代数的簡約、高速）
    /// 2. ループ最適化（ビームサーチ）
    fn optimize_program(&self, program: AstNode) -> AstNode {
        // 第1段階: ルールベース最適化（高速）
        let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules());
        let rule_optimized = rule_optimizer.optimize(program);

        // 第2段階: ループ最適化（ビームサーチ）
        let loop_suggester = Self::create_loop_suggester();
        let loop_optimizer = self.create_ast_optimizer(loop_suggester);
        let (optimized, _history) = loop_optimizer.optimize_with_history(rule_optimized);

        optimized
    }

    /// グラフをコンパイル
    ///
    /// `enable_runtime_selector()`が呼び出されている場合、グラフ最適化とAST最適化の
    /// 両方で自動的に実測値ベースの候補選択が有効になります。
    fn compile_graph(&mut self, graph: Graph) -> Result<C::Kernel, String> {
        // Signatureを作成（最適化前のGraphから）
        let signature = crate::lowerer::create_signature(&graph);

        // グラフ最適化（Phase 1 + Phase 2、RuntimeSelector自動使用）
        let optimized_graph = self.optimize_graph_internal(graph);

        // グラフからAST Programを抽出
        let program = crate::lowerer::extract_program(optimized_graph);

        // AST最適化（RuntimeSelector自動使用）
        let (optimized_program, _history) = self.optimize_ast_internal(program, signature.clone());

        // レンダリングとコンパイル
        let code = self.renderer().render(&optimized_program);
        Ok(self.compiler().compile(&code, signature))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Buffer, Kernel, KernelSignature};
    use crate::graph::DType;

    // テスト用のダミー実装
    #[derive(Clone)]
    struct DummyRenderer;

    impl Renderer for DummyRenderer {
        type CodeRepr = String;
        type Option = ();

        fn render(&self, _program: &crate::ast::AstNode) -> Self::CodeRepr {
            "dummy code".to_string()
        }

        fn is_available(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Clone)]
    struct DummyBuffer;

    impl Buffer for DummyBuffer {
        fn allocate(_shape: Vec<usize>, _dtype: crate::ast::DType) -> Self {
            Self
        }

        fn shape(&self) -> Vec<usize> {
            vec![]
        }

        fn dtype(&self) -> crate::ast::DType {
            crate::ast::DType::F32
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

    #[derive(Debug, Clone)]
    struct DummyKernel;

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
    struct DummyCompiler;

    impl Compiler for DummyCompiler {
        type CodeRepr = String;
        type Buffer = DummyBuffer;
        type Kernel = DummyKernel;
        type Option = ();

        fn new() -> Self {
            Self
        }

        fn is_available(&self) -> bool {
            true
        }

        fn compile(
            &mut self,
            _code: &Self::CodeRepr,
            _signature: crate::backend::KernelSignature,
        ) -> Self::Kernel {
            DummyKernel
        }

        fn create_buffer(&self, _shape: Vec<usize>, _element_size: usize) -> Self::Buffer {
            DummyBuffer
        }
    }

    #[test]
    fn test_generic_pipeline_creation() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let pipeline = GenericPipeline::new(renderer, compiler);

        assert_eq!(pipeline.cache_size(), 0);
    }

    #[test]
    fn test_compile_and_cache() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // シンプルなグラフを作成
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        // コンパイルしてキャッシュ
        let result = pipeline.compile_and_cache("test_key".to_string(), graph);
        assert!(result.is_ok());
        assert_eq!(pipeline.cache_size(), 1);

        // キャッシュから取得
        let cached = pipeline.get_cached_kernel("test_key");
        assert!(cached.is_some());
    }

    #[test]
    fn test_cache_operations() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // 複数のグラフをキャッシュ
        for i in 0..3 {
            let mut graph = Graph::new();
            let a = graph.input("a", DType::F32, vec![10]);
            graph.output("out", a);

            let key = format!("key_{}", i);
            pipeline.compile_and_cache(key, graph).unwrap();
        }

        assert_eq!(pipeline.cache_size(), 3);

        // 1つ削除
        let removed = pipeline.remove_cached_kernel("key_1");
        assert!(removed.is_some());
        assert_eq!(pipeline.cache_size(), 2);

        // クリア
        pipeline.clear_cache();
        assert_eq!(pipeline.cache_size(), 0);
    }

    #[test]
    fn test_cache_overwrite() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // 同じキーで2回キャッシュ
        for _ in 0..2 {
            let mut graph = Graph::new();
            let a = graph.input("a", DType::F32, vec![10]);
            graph.output("out", a);

            pipeline
                .compile_and_cache("same_key".to_string(), graph)
                .unwrap();
        }

        // 上書きされるのでサイズは1
        assert_eq!(pipeline.cache_size(), 1);
    }

    #[test]
    fn test_custom_optimization_config() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // フィールドに直接アクセスして設定をカスタマイズ
        pipeline.graph_config.beam_width = 20;
        pipeline.graph_config.max_steps = 50;
        pipeline.graph_config.show_progress = true;

        pipeline.ast_config.beam_width = 15;
        pipeline.ast_config.max_steps = 75;
        pipeline.ast_config.show_progress = true;

        // カスタム設定が適用されている
        assert_eq!(pipeline.graph_config.beam_width, 20);
        assert_eq!(pipeline.graph_config.max_steps, 50);
        assert_eq!(pipeline.ast_config.beam_width, 15);
    }
}
