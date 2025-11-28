use crate::ast::AstNode;
use crate::backend::{Compiler, Pipeline, Renderer};
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
    ContiguousInsertionSuggester, FusionSuggester, GraphCostEstimator, LoweringSuggester,
    OptimizationHistory as GraphOptimizationHistory, ParallelStrategyChanger, SimdSuggester,
    SimpleCostEstimator, TilingSuggester, ViewInsertionSuggester, ViewMergeSuggester,
};
use std::collections::HashMap;

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
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_steps: 10000,
            show_progress: false,
        }
    }
}

/// 最適化履歴を管理する構造体
#[derive(Debug, Clone, Default)]
pub struct OptimizationHistories {
    /// グラフ最適化履歴
    pub graph: Option<GraphOptimizationHistory>,
    /// AST最適化履歴
    pub ast: Option<AstOptimizationHistory>,
}

impl OptimizationHistories {
    /// 全ての履歴をクリア
    pub fn clear(&mut self) {
        self.graph = None;
        self.ast = None;
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
/// // AST最適化を有効化
/// pipeline.enable_ast_optimization = true;
///
/// // 設定のカスタマイズ
/// pipeline.graph_config.beam_width = 8;
///
/// // コンパイル
/// let kernel = pipeline.compile_graph(graph)?;
/// ```
///
/// # Note
/// グラフ最適化は常に有効です（LoweringSuggesterによるCustomノード変換が必須）。
pub struct GenericPipeline<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    renderer: R,
    compiler: C,
    /// コンパイル済みKernelのキャッシュ
    kernel_cache: HashMap<String, C::Kernel>,
    /// 最適化履歴
    pub histories: OptimizationHistories,
    /// グラフ最適化の設定
    pub graph_config: OptimizationConfig,
    /// AST最適化を有効にするか
    pub enable_ast_optimization: bool,
    /// AST最適化の設定
    pub ast_config: OptimizationConfig,
    /// 最適化履歴を収集するか（DEBUGビルドではデフォルトでtrue、RELEASEビルドではfalse）
    pub collect_histories: bool,
}

impl<R, C> GenericPipeline<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 新しいGenericPipelineを作成
    ///
    /// グラフ最適化は常に有効です（LoweringSuggesterによるCustomノード変換が必須）。
    /// AST最適化はデフォルトで無効です。
    ///
    /// 最適化履歴の収集は、DEBUGビルドではデフォルトで有効、RELEASEビルドでは無効です。
    pub fn new(renderer: R, compiler: C) -> Self {
        Self {
            renderer,
            compiler,
            kernel_cache: HashMap::new(),
            histories: OptimizationHistories::default(),
            graph_config: OptimizationConfig::default(),
            enable_ast_optimization: false,
            ast_config: OptimizationConfig::default(),
            collect_histories: cfg!(debug_assertions),
        }
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
    pub fn compile_graph_with_history(&mut self, graph: Graph) -> Result<C::Kernel, String> {
        // グラフ最適化
        let optimized_graph = self.optimize_graph_internal(graph);

        // Lowering
        let program = self.lower_to_program(optimized_graph);

        // AST最適化
        let optimized_program = if self.enable_ast_optimization {
            let (program, _history) = self.optimize_ast_internal(program);
            program
        } else {
            program
        };

        // レンダリングとコンパイル
        let code = self.renderer().render(&optimized_program);
        Ok(self.compiler().compile(&code))
    }

    /// 最適化のみを実行（コンパイルなし、AST履歴を返す）
    ///
    /// 最適化が有効な場合、最適化履歴を内部に保存します。
    /// プログラム全体のAST最適化履歴（キー: "program"）と最適化後のProgramを返します。
    /// コンパイルは行わないため、OpenMPなどのランタイムサポートが不要です。
    pub fn optimize_graph_with_all_histories(
        &mut self,
        graph: Graph,
    ) -> Result<(AstNode, HashMap<String, AstOptimizationHistory>), String> {
        // グラフ最適化（常に有効）
        let suggester = Self::create_graph_suggester();
        let estimator = SimpleCostEstimator::new();
        let optimizer = self.create_graph_optimizer(suggester, estimator);

        let (optimized_graph, history) = optimizer.optimize_with_history(graph);
        if self.collect_histories {
            self.histories.graph = Some(history);
        }

        // Lowering
        let program = self.lower_to_program(optimized_graph);

        // AST最適化（Program全体を最適化）
        let (program, all_histories) = if self.enable_ast_optimization {
            let (program, history) = self.optimize_ast_internal(program);

            let mut all_histories = HashMap::new();
            all_histories.insert("program".to_string(), history);

            (program, all_histories)
        } else {
            (program, HashMap::new())
        };

        Ok((program, all_histories))
    }

    /// 最適化履歴を記録しながらグラフをコンパイル（AST履歴を返す）
    ///
    /// 最適化が有効な場合、最適化履歴を内部に保存します。
    /// プログラム全体のAST最適化履歴（キー: "program"）と最適化後のProgramを返します。
    pub fn compile_graph_with_all_histories(
        &mut self,
        graph: Graph,
    ) -> CompileWithHistoriesResult<C::Kernel> {
        // グラフ最適化
        let optimized_graph = self.optimize_graph_internal(graph);

        // Lowering
        let program = self.lower_to_program(optimized_graph);

        // AST最適化（Program全体を最適化）
        let (optimized_program, all_histories) = if self.enable_ast_optimization {
            let (program, history) = self.optimize_ast_internal(program);

            let mut all_histories = HashMap::new();
            all_histories.insert("program".to_string(), history);

            (program, all_histories)
        } else {
            (program, HashMap::new())
        };

        // レンダリングとコンパイル
        let code = self.renderer().render(&optimized_program);
        let kernel = self.compiler().compile(&code);
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

    /// グラフ最適化用のSuggesterを作成
    fn create_graph_suggester() -> CompositeSuggester {
        CompositeSuggester::new(vec![
            Box::new(ViewInsertionSuggester::new()),
            Box::new(ViewMergeSuggester::new()),
            Box::new(ConstPropagationSuggester::new()),
            Box::new(TilingSuggester::with_default_tile_sizes()),
            Box::new(ContiguousInsertionSuggester::new()),
            Box::new(FusionSuggester::new()),
            Box::new(ParallelStrategyChanger::new()),
            Box::new(SimdSuggester::new()),
            // LoweringSuggesterは最後に追加（他の最適化後にlowering）
            Box::new(LoweringSuggester::new()),
        ])
    }

    /// グラフ最適化用のOptimizerを作成・設定
    fn create_graph_optimizer<E>(
        &self,
        suggester: CompositeSuggester,
        estimator: E,
    ) -> BeamSearchGraphOptimizer<CompositeSuggester, E>
    where
        E: GraphCostEstimator,
    {
        BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(self.graph_config.beam_width)
            .with_max_steps(self.graph_config.max_steps)
            .with_progress(self.graph_config.show_progress)
    }

    /// AST最適化用のSuggesterを作成
    fn create_ast_suggester() -> AstCompositeSuggester {
        AstCompositeSuggester::new(vec![
            Box::new(RuleBaseSuggester::new(all_rules_with_search())),
            Box::new(LoopTilingSuggester::with_default_sizes()),
            Box::new(LoopInliningSuggester::with_default_limit()),
            Box::new(LoopInterchangeSuggester::new()),
            Box::new(LoopFusionSuggester::new()),
            Box::new(FunctionInliningSuggester::with_default_limit()),
        ])
    }

    /// AST最適化用のOptimizerを作成・設定
    fn create_ast_optimizer<E>(
        &self,
        suggester: AstCompositeSuggester,
        estimator: E,
    ) -> AstBeamSearchOptimizer<AstCompositeSuggester, E>
    where
        E: crate::opt::ast::CostEstimator,
    {
        AstBeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(self.ast_config.beam_width)
            .with_max_steps(self.ast_config.max_steps)
            .with_progress(self.ast_config.show_progress)
    }

    /// グラフ最適化の内部処理（履歴付き）
    fn optimize_graph_internal(&mut self, graph: Graph) -> Graph {
        let suggester = Self::create_graph_suggester();
        let estimator = SimpleCostEstimator::new();
        let optimizer = self.create_graph_optimizer(suggester, estimator);

        let (optimized, history) = optimizer.optimize_with_history(graph);
        if self.collect_histories {
            self.histories.graph = Some(history);
        }
        optimized
    }

    /// AST最適化の内部処理（履歴付き）
    fn optimize_ast_internal(&mut self, program: AstNode) -> (AstNode, AstOptimizationHistory) {
        // 1. ルールベース最適化（代数的簡約など）を先に適用
        let rules = all_rules_with_search();
        let rule_optimizer = RuleBaseOptimizer::new(rules).with_max_iterations(100);
        let program = rule_optimizer.optimize(program);

        // 2. ビームサーチ最適化を適用
        let suggester = Self::create_ast_suggester();
        let estimator = AstSimpleCostEstimator::new();
        let optimizer = self.create_ast_optimizer(suggester, estimator);

        let (optimized, history) = optimizer.optimize_with_history(program);
        if self.collect_histories {
            self.histories.ast = Some(history.clone());
        }
        (optimized, history)
    }
}

impl<R, C> Pipeline for GenericPipeline<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
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
    /// 以下の最適化を適用（常に有効）：
    /// 1. ViewInsertionSuggester（Transpose含む）
    /// 2. FusionSuggester
    /// 3. ParallelStrategyChanger
    /// 4. SimdSuggester
    /// 5. LoweringSuggester（GraphOp → Custom変換）
    fn optimize_graph(&self, graph: Graph) -> Graph {
        let suggester = Self::create_graph_suggester();
        let estimator = SimpleCostEstimator::new();
        let optimizer = self.create_graph_optimizer(suggester, estimator);

        let (optimized_graph, history) = optimizer.optimize_with_history(graph);

        // 履歴を保存（mutabilityの問題があるため、内部可変性を使う必要がある）
        // ここでは一旦最適化だけを実行し、履歴の保存は外部で行う
        // より良い設計のために、後でCell/RefCellを使うことを検討
        drop(history); // 履歴は今は保存できない

        optimized_graph
    }

    /// プログラム（AST）最適化を実行
    ///
    /// 有効な場合、Program全体に対して以下の最適化を2段階で適用：
    /// 1. ルールベース最適化（代数的簡約）
    /// 2. ビームサーチ最適化（代数的ルール + ループタイル化 + ループインライン展開）
    fn optimize_program(&self, program: AstNode) -> AstNode {
        if !self.enable_ast_optimization {
            return program;
        }

        let suggester = Self::create_ast_suggester();
        let estimator = AstSimpleCostEstimator::new();
        let optimizer = self.create_ast_optimizer(suggester, estimator);

        let (program, _history) = optimizer.optimize_with_history(program);

        program
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Buffer, Kernel, KernelSignature};
    use crate::graph::DType;

    // テスト用のダミー実装
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
    }

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

        fn compile(&mut self, _code: &Self::CodeRepr) -> Self::Kernel {
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
    fn test_ast_optimization_disabled_by_default() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let pipeline = GenericPipeline::new(renderer, compiler);

        // AST最適化はデフォルトで無効（グラフ最適化は常に有効）
        assert!(!pipeline.enable_ast_optimization);
    }

    #[test]
    fn test_enable_ast_optimization() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // フィールドに直接アクセスしてAST最適化を有効化
        pipeline.enable_ast_optimization = true;

        // AST最適化が有効になっている
        assert!(pipeline.enable_ast_optimization);
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
