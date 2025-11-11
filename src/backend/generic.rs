use std::collections::HashMap;

use crate::ast::AstNode;
use crate::backend::{Compiler, Pipeline, Renderer};
use crate::graph::Graph;
use crate::opt::ast::rules::{all_algebraic_rules, all_rules_with_search};
use crate::opt::ast::{
    BeamSearchOptimizer as AstBeamSearchOptimizer, OptimizationHistory as AstOptimizationHistory,
    Optimizer, RuleBaseOptimizer, RuleBaseSuggester, SimpleCostEstimator as AstSimpleCostEstimator,
};
use crate::opt::graph::{
    BeamSearchGraphOptimizer, CompositeSuggester, FusionSuggester,
    OptimizationHistory as GraphOptimizationHistory, ParallelStrategyChanger,
    SimpleCostEstimator as GraphSimpleCostEstimator, ViewInsertionSuggester,
};

/// compile_graph_with_all_historiesの戻り値の型
type CompileWithHistoriesResult<K> =
    Result<(K, AstNode, HashMap<String, AstOptimizationHistory>), String>;

/// グラフ最適化の設定
pub struct GraphOptimizationConfig {
    /// ビーム幅
    pub beam_width: usize,
    /// 最大ステップ数
    pub max_steps: usize,
    /// プログレスバーを表示するか
    pub show_progress: bool,
}

impl Default for GraphOptimizationConfig {
    fn default() -> Self {
        Self {
            beam_width: 10,
            max_steps: 100,
            show_progress: false,
        }
    }
}

/// AST最適化の設定
pub struct AstOptimizationConfig {
    /// ルールベース最適化の最大反復回数
    pub rule_max_iterations: usize,
    /// ビームサーチのビーム幅
    pub beam_width: usize,
    /// ビームサーチの最大ステップ数
    pub max_steps: usize,
    /// プログレスバーを表示するか
    pub show_progress: bool,
}

impl Default for AstOptimizationConfig {
    fn default() -> Self {
        Self {
            rule_max_iterations: 100,
            beam_width: 10,
            max_steps: 100,
            show_progress: false,
        }
    }
}

/// 汎用的なPipeline実装
///
/// 任意のRendererとCompilerを組み合わせて使用でき、
/// コンパイル済みのKernelをキャッシュする機能を提供します。
///
/// 最適化履歴の記録機能を持ち、可視化ツールと統合できます。
pub struct GenericPipeline<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    renderer: R,
    compiler: C,
    /// コンパイル済みKernelのキャッシュ
    /// キーはユーザーが指定する識別文字列
    kernel_cache: HashMap<String, C::Kernel>,
    /// 最新のグラフ最適化履歴
    last_graph_optimization_history: Option<GraphOptimizationHistory>,
    /// 最新のAST最適化履歴
    last_ast_optimization_history: Option<AstOptimizationHistory>,
    /// グラフ最適化を有効にするか
    enable_graph_optimization: bool,
    /// グラフ最適化の設定
    graph_optimization_config: GraphOptimizationConfig,
    /// AST最適化を有効にするか
    enable_ast_optimization: bool,
    /// AST最適化の設定
    ast_optimization_config: AstOptimizationConfig,
}

impl<R, C> GenericPipeline<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 新しいGenericPipelineを作成
    ///
    /// デフォルトでは最適化が無効になっています。
    /// `with_graph_optimization()`や`with_ast_optimization()`で有効化してください。
    pub fn new(renderer: R, compiler: C) -> Self {
        Self {
            renderer,
            compiler,
            kernel_cache: HashMap::new(),
            last_graph_optimization_history: None,
            last_ast_optimization_history: None,
            enable_graph_optimization: false,
            graph_optimization_config: GraphOptimizationConfig::default(),
            enable_ast_optimization: false,
            ast_optimization_config: AstOptimizationConfig::default(),
        }
    }

    /// グラフ最適化を有効化
    pub fn with_graph_optimization(mut self, enabled: bool) -> Self {
        self.enable_graph_optimization = enabled;
        self
    }

    /// グラフ最適化の設定を指定
    pub fn with_graph_optimization_config(mut self, config: GraphOptimizationConfig) -> Self {
        self.graph_optimization_config = config;
        self.enable_graph_optimization = true;
        self
    }

    /// AST最適化を有効化
    pub fn with_ast_optimization(mut self, enabled: bool) -> Self {
        self.enable_ast_optimization = enabled;
        self
    }

    /// AST最適化の設定を指定
    pub fn with_ast_optimization_config(mut self, config: AstOptimizationConfig) -> Self {
        self.ast_optimization_config = config;
        self.enable_ast_optimization = true;
        self
    }

    /// グラフ最適化とAST最適化の両方を有効化
    pub fn with_all_optimizations(self) -> Self {
        self.with_graph_optimization(true)
            .with_ast_optimization(true)
    }

    /// 最新のグラフ最適化履歴を取得
    pub fn last_graph_optimization_history(&self) -> Option<&GraphOptimizationHistory> {
        self.last_graph_optimization_history.as_ref()
    }

    /// 最新のAST最適化履歴を取得
    pub fn last_ast_optimization_history(&self) -> Option<&AstOptimizationHistory> {
        self.last_ast_optimization_history.as_ref()
    }

    /// 最新のグラフ最適化履歴を所有権とともに取得
    pub fn take_graph_optimization_history(&mut self) -> Option<GraphOptimizationHistory> {
        self.last_graph_optimization_history.take()
    }

    /// 最新のAST最適化履歴を所有権とともに取得
    pub fn take_ast_optimization_history(&mut self) -> Option<AstOptimizationHistory> {
        self.last_ast_optimization_history.take()
    }

    /// 最適化履歴をクリア
    pub fn clear_histories(&mut self) {
        self.last_graph_optimization_history = None;
        self.last_ast_optimization_history = None;
    }

    /// グラフ最適化履歴を設定
    ///
    /// 外部で最適化を行った後、その履歴をGenericPipelineに保存します。
    pub fn set_graph_optimization_history(&mut self, history: GraphOptimizationHistory) {
        self.last_graph_optimization_history = Some(history);
    }

    /// AST最適化履歴を設定
    ///
    /// 外部で最適化を行った後、その履歴をGenericPipelineに保存します。
    pub fn set_ast_optimization_history(&mut self, history: AstOptimizationHistory) {
        self.last_ast_optimization_history = Some(history);
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
        let optimized_graph = if self.enable_graph_optimization {
            let suggester = CompositeSuggester::new(vec![
                Box::new(ViewInsertionSuggester::new().with_transpose(true)),
                Box::new(FusionSuggester::new()),
                Box::new(ParallelStrategyChanger::with_default_strategies()),
            ]);
            let estimator = GraphSimpleCostEstimator::new();

            let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
                .with_beam_width(self.graph_optimization_config.beam_width)
                .with_max_steps(self.graph_optimization_config.max_steps)
                .with_progress(self.graph_optimization_config.show_progress);

            let (optimized, history) = optimizer.optimize_with_history(graph);
            self.last_graph_optimization_history = Some(history);
            optimized
        } else {
            graph
        };

        // Lowering
        let program = self.lower_to_program(optimized_graph);

        // AST最適化
        let optimized_program = if self.enable_ast_optimization {
            let AstNode::Program {
                functions,
                entry_point,
            } = &program
            else {
                panic!("Expected AstNode::Program");
            };

            let mut optimized_functions = Vec::new();
            let mut all_histories = std::collections::HashMap::new();

            for func_node in functions {
                let AstNode::Function {
                    name,
                    params,
                    return_type,
                    body,
                    kind,
                } = func_node
                else {
                    continue;
                };

                // ステップ1: ルールベース最適化
                let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules())
                    .with_max_iterations(self.ast_optimization_config.rule_max_iterations);

                let rule_optimized = rule_optimizer.optimize(body.as_ref().clone());

                // ステップ2: ビームサーチ最適化
                let beam_suggester = RuleBaseSuggester::new(all_rules_with_search());
                let beam_estimator = AstSimpleCostEstimator::new();

                let beam_optimizer = AstBeamSearchOptimizer::new(beam_suggester, beam_estimator)
                    .with_beam_width(self.ast_optimization_config.beam_width)
                    .with_max_steps(self.ast_optimization_config.max_steps)
                    .with_progress(self.ast_optimization_config.show_progress);

                let (final_optimized, history) =
                    beam_optimizer.optimize_with_history(rule_optimized);

                if let Some(func_name) = name {
                    all_histories.insert(func_name.clone(), history);
                }

                // 最適化後の関数を作成
                let optimized_func = crate::ast::helper::function(
                    name.clone(),
                    kind.clone(),
                    params.clone(),
                    return_type.clone(),
                    final_optimized,
                );

                optimized_functions.push(optimized_func);
            }

            // 最初の関数の履歴を保存（後方互換性のため）
            if let Some((_, first_history)) = all_histories.iter().next() {
                self.last_ast_optimization_history = Some(first_history.clone());
            }

            crate::ast::helper::program(optimized_functions, entry_point.clone())
        } else {
            program
        };

        // レンダリングとコンパイル
        let code = self.renderer().render(&optimized_program);
        Ok(self.compiler().compile(&code))
    }

    /// 最適化のみを実行（コンパイルなし、すべてのAST履歴を返す）
    ///
    /// 最適化が有効な場合、最適化履歴を内部に保存します。
    /// すべてのFunction用のAST最適化履歴と最適化後のProgramを返します。
    /// コンパイルは行わないため、OpenMPなどのランタイムサポートが不要です。
    pub fn optimize_graph_with_all_histories(
        &mut self,
        graph: Graph,
    ) -> Result<(AstNode, HashMap<String, AstOptimizationHistory>), String> {
        // グラフ最適化
        let optimized_graph = if self.enable_graph_optimization {
            let suggester = CompositeSuggester::new(vec![
                Box::new(ViewInsertionSuggester::new().with_transpose(true)),
                Box::new(FusionSuggester::new()),
                Box::new(ParallelStrategyChanger::with_default_strategies()),
            ]);
            let estimator = GraphSimpleCostEstimator::new();

            let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
                .with_beam_width(self.graph_optimization_config.beam_width)
                .with_max_steps(self.graph_optimization_config.max_steps)
                .with_progress(self.graph_optimization_config.show_progress);

            let (optimized, history) = optimizer.optimize_with_history(graph);
            self.last_graph_optimization_history = Some(history);
            optimized
        } else {
            graph
        };

        // Lowering
        let program = self.lower_to_program(optimized_graph);

        // AST最適化
        let (optimized_program, all_histories) = if self.enable_ast_optimization {
            let AstNode::Program {
                functions,
                entry_point,
            } = &program
            else {
                panic!("Expected AstNode::Program");
            };

            let mut optimized_functions = Vec::new();
            let mut all_histories = std::collections::HashMap::new();

            for func_node in functions {
                let AstNode::Function {
                    name,
                    params,
                    return_type,
                    body,
                    kind,
                } = func_node
                else {
                    continue;
                };

                // ステップ1: ルールベース最適化
                let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules())
                    .with_max_iterations(self.ast_optimization_config.rule_max_iterations);

                let rule_optimized = rule_optimizer.optimize(body.as_ref().clone());

                // ステップ2: ビームサーチ最適化
                let beam_suggester = RuleBaseSuggester::new(all_rules_with_search());
                let beam_estimator = AstSimpleCostEstimator::new();

                let beam_optimizer = AstBeamSearchOptimizer::new(beam_suggester, beam_estimator)
                    .with_beam_width(self.ast_optimization_config.beam_width)
                    .with_max_steps(self.ast_optimization_config.max_steps)
                    .with_progress(self.ast_optimization_config.show_progress);

                let (final_optimized, history) =
                    beam_optimizer.optimize_with_history(rule_optimized);

                if let Some(func_name) = name {
                    all_histories.insert(func_name.clone(), history);
                }

                // 最適化後の関数を作成
                let optimized_func = crate::ast::helper::function(
                    name.clone(),
                    kind.clone(),
                    params.clone(),
                    return_type.clone(),
                    final_optimized,
                );

                optimized_functions.push(optimized_func);
            }

            // 最初の関数の履歴を保存（後方互換性のため）
            if let Some((_, first_history)) = all_histories.iter().next() {
                self.last_ast_optimization_history = Some(first_history.clone());
            }

            (
                crate::ast::helper::program(optimized_functions, entry_point.clone()),
                all_histories,
            )
        } else {
            (program, std::collections::HashMap::new())
        };

        Ok((optimized_program, all_histories))
    }

    /// 最適化履歴を記録しながらグラフをコンパイル（すべてのAST履歴を返す）
    ///
    /// 最適化が有効な場合、最適化履歴を内部に保存します。
    /// すべてのFunction用のAST最適化履歴と最適化後のProgramを返します。
    pub fn compile_graph_with_all_histories(
        &mut self,
        graph: Graph,
    ) -> CompileWithHistoriesResult<C::Kernel> {
        // グラフ最適化
        let optimized_graph = if self.enable_graph_optimization {
            let suggester = CompositeSuggester::new(vec![
                Box::new(ViewInsertionSuggester::new().with_transpose(true)),
                Box::new(FusionSuggester::new()),
                Box::new(ParallelStrategyChanger::with_default_strategies()),
            ]);
            let estimator = GraphSimpleCostEstimator::new();

            let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
                .with_beam_width(self.graph_optimization_config.beam_width)
                .with_max_steps(self.graph_optimization_config.max_steps)
                .with_progress(self.graph_optimization_config.show_progress);

            let (optimized, history) = optimizer.optimize_with_history(graph);
            self.last_graph_optimization_history = Some(history);
            optimized
        } else {
            graph
        };

        // Lowering
        let program = self.lower_to_program(optimized_graph);

        // AST最適化
        let (optimized_program, all_histories) = if self.enable_ast_optimization {
            let AstNode::Program {
                functions,
                entry_point,
            } = &program
            else {
                panic!("Expected AstNode::Program");
            };

            let mut optimized_functions = Vec::new();
            let mut all_histories = std::collections::HashMap::new();

            for func_node in functions {
                let AstNode::Function {
                    name,
                    params,
                    return_type,
                    body,
                    kind,
                } = func_node
                else {
                    continue;
                };

                // ステップ1: ルールベース最適化
                let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules())
                    .with_max_iterations(self.ast_optimization_config.rule_max_iterations);

                let rule_optimized = rule_optimizer.optimize(body.as_ref().clone());

                // ステップ2: ビームサーチ最適化
                let beam_suggester = RuleBaseSuggester::new(all_rules_with_search());
                let beam_estimator = AstSimpleCostEstimator::new();

                let beam_optimizer = AstBeamSearchOptimizer::new(beam_suggester, beam_estimator)
                    .with_beam_width(self.ast_optimization_config.beam_width)
                    .with_max_steps(self.ast_optimization_config.max_steps)
                    .with_progress(self.ast_optimization_config.show_progress);

                let (final_optimized, history) =
                    beam_optimizer.optimize_with_history(rule_optimized);

                if let Some(func_name) = name {
                    all_histories.insert(func_name.clone(), history);
                }

                // 最適化後の関数を作成
                let optimized_func = crate::ast::helper::function(
                    name.clone(),
                    kind.clone(),
                    params.clone(),
                    return_type.clone(),
                    final_optimized,
                );

                optimized_functions.push(optimized_func);
            }

            // 最初の関数の履歴を保存（後方互換性のため）
            if let Some((_, first_history)) = all_histories.iter().next() {
                self.last_ast_optimization_history = Some(first_history.clone());
            }

            (
                crate::ast::helper::program(optimized_functions, entry_point.clone()),
                all_histories,
            )
        } else {
            (program, std::collections::HashMap::new())
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
    /// 有効な場合、以下の最適化を適用：
    /// 1. ViewInsertionSuggester（Transpose含む）
    /// 2. FusionSuggester
    /// 3. ParallelStrategyChanger
    fn optimize_graph(&self, graph: Graph) -> Graph {
        if !self.enable_graph_optimization {
            return graph;
        }

        let suggester = CompositeSuggester::new(vec![
            Box::new(ViewInsertionSuggester::new().with_transpose(true)),
            Box::new(FusionSuggester::new()),
            Box::new(ParallelStrategyChanger::with_default_strategies()),
        ]);
        let estimator = GraphSimpleCostEstimator::new();

        let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(self.graph_optimization_config.beam_width)
            .with_max_steps(self.graph_optimization_config.max_steps)
            .with_progress(self.graph_optimization_config.show_progress);

        let (optimized_graph, history) = optimizer.optimize_with_history(graph);

        // 履歴を保存（mutabilityの問題があるため、内部可変性を使う必要がある）
        // ここでは一旦最適化だけを実行し、履歴の保存は外部で行う
        // より良い設計のために、後でCell/RefCellを使うことを検討
        drop(history); // 履歴は今は保存できない

        optimized_graph
    }

    /// プログラム（AST）最適化を実行
    ///
    /// 有効な場合、以下の最適化を2段階で適用：
    /// 1. ルールベース最適化（代数的簡約）
    /// 2. ビームサーチ最適化
    fn optimize_program(&self, program: AstNode) -> AstNode {
        if !self.enable_ast_optimization {
            return program;
        }

        let AstNode::Program {
            functions,
            entry_point,
        } = &program
        else {
            panic!("Expected AstNode::Program");
        };

        // 各関数を個別に最適化
        let mut optimized_functions = Vec::new();

        for func in functions {
            let AstNode::Function {
                name,
                params,
                return_type,
                body,
                kind,
            } = func
            else {
                continue;
            };
            // ステップ1: ルールベース最適化
            let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules())
                .with_max_iterations(self.ast_optimization_config.rule_max_iterations);

            let rule_optimized = rule_optimizer.optimize(body.as_ref().clone());

            // ステップ2: ビームサーチ最適化（探索用の完全なルール集を使用）
            let beam_suggester = RuleBaseSuggester::new(all_rules_with_search());
            let beam_estimator = AstSimpleCostEstimator::new();

            let beam_optimizer = AstBeamSearchOptimizer::new(beam_suggester, beam_estimator)
                .with_beam_width(self.ast_optimization_config.beam_width)
                .with_max_steps(self.ast_optimization_config.max_steps)
                .with_progress(self.ast_optimization_config.show_progress);

            let (final_optimized, _history) = beam_optimizer.optimize_with_history(rule_optimized);

            // 最適化後の関数を作成
            let optimized_func = crate::ast::helper::function(
                name.clone(),
                kind.clone(),
                params.clone(),
                return_type.clone(),
                final_optimized,
            );

            optimized_functions.push(optimized_func);
        }

        crate::ast::helper::program(optimized_functions, entry_point.clone())
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
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
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
            let a = graph
                .input("a")
                .with_dtype(DType::F32)
                .with_shape(vec![10])
                .build();
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
            let a = graph
                .input("a")
                .with_dtype(DType::F32)
                .with_shape(vec![10])
                .build();
            graph.output("out", a);

            pipeline
                .compile_and_cache("same_key".to_string(), graph)
                .unwrap();
        }

        // 上書きされるのでサイズは1
        assert_eq!(pipeline.cache_size(), 1);
    }

    #[test]
    fn test_optimization_disabled_by_default() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let pipeline = GenericPipeline::new(renderer, compiler);

        // デフォルトでは最適化が無効
        assert!(!pipeline.enable_graph_optimization);
        assert!(!pipeline.enable_ast_optimization);
    }

    #[test]
    fn test_enable_optimizations() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let pipeline = GenericPipeline::new(renderer, compiler)
            .with_graph_optimization(true)
            .with_ast_optimization(true);

        // 最適化が有効になっている
        assert!(pipeline.enable_graph_optimization);
        assert!(pipeline.enable_ast_optimization);
    }

    #[test]
    fn test_all_optimizations() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let pipeline = GenericPipeline::new(renderer, compiler).with_all_optimizations();

        // すべての最適化が有効
        assert!(pipeline.enable_graph_optimization);
        assert!(pipeline.enable_ast_optimization);
    }

    #[test]
    fn test_custom_optimization_config() {
        let renderer = DummyRenderer;
        let compiler = DummyCompiler;

        let graph_config = GraphOptimizationConfig {
            beam_width: 20,
            max_steps: 50,
            show_progress: true,
        };

        let ast_config = AstOptimizationConfig {
            rule_max_iterations: 200,
            beam_width: 15,
            max_steps: 75,
            show_progress: true,
        };

        let pipeline = GenericPipeline::new(renderer, compiler)
            .with_graph_optimization_config(graph_config)
            .with_ast_optimization_config(ast_config);

        // カスタム設定が適用されている
        assert_eq!(pipeline.graph_optimization_config.beam_width, 20);
        assert_eq!(pipeline.graph_optimization_config.max_steps, 50);
        assert_eq!(pipeline.ast_optimization_config.rule_max_iterations, 200);
        assert_eq!(pipeline.ast_optimization_config.beam_width, 15);
    }
}
