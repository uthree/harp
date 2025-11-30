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
    BeamSearchGraphOptimizer, BufferAbsorptionSuggester, CompositeSuggester,
    ContiguousInsertionSuggester, FusionSuggester, GraphCostEstimator, KernelMergeCostEstimator,
    KernelMergeSuggester, LoweringSuggester, OptimizationHistory as GraphOptimizationHistory,
    SimpleCostEstimator, SinkAbsorptionSuggester, TilingSuggester, ViewInsertionSuggester,
    ViewMergeSuggester,
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
    /// カーネルマージ（2段階最適化）を有効にするか
    /// 複数のCustom(Function)を1つのCustom(Program)にマージします
    pub enable_kernel_merge: bool,
}

/// 最適化済みグラフからAST Programを抽出する
///
/// 複数の出力がある場合、すべてのカーネルを1つのProgramに統合します。
/// SinkAbsorptionSuggesterが生成したSink(Program)を直接返すか、
/// KernelMergeSuggesterが生成したCustom(Program)を直接返すか、
/// Custom(Function)がある場合はLowererを使用してProgramを生成します。
fn extract_program_from_graph(graph: Graph) -> AstNode {
    use crate::graph::GraphOp;
    use std::collections::HashSet;

    // 1. SinkノードのProgramを最優先で確認
    // SinkAbsorptionSuggesterが生成した完全なProgramがあればそれを使用
    if let Some(sink) = graph.sink()
        && let GraphOp::Sink { ast, .. } = &sink.op
        && let AstNode::Program { functions, .. } = ast
        && !functions.is_empty()
    {
        return ast.clone();
    }

    // 2. Custom(Program/Function)を収集（フォールバック）
    let mut collected_programs: Vec<&AstNode> = Vec::new();
    let mut collected_functions: Vec<&AstNode> = Vec::new();
    let mut visited: HashSet<*const crate::graph::GraphNodeData> = HashSet::new();

    fn collect_customs<'a>(
        node: &'a crate::graph::GraphNode,
        visited: &mut HashSet<*const crate::graph::GraphNodeData>,
        programs: &mut Vec<&'a AstNode>,
        functions: &mut Vec<&'a AstNode>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        if let GraphOp::Custom { ast, .. } = &node.op {
            match ast {
                AstNode::Program { .. } => programs.push(ast),
                AstNode::Function { .. } => functions.push(ast),
                _ => {}
            }
        }

        for src in &node.src {
            collect_customs(src, visited, programs, functions);
        }
    }

    let outputs = graph.outputs();
    for output in outputs.values() {
        collect_customs(
            output,
            &mut visited,
            &mut collected_programs,
            &mut collected_functions,
        );
    }

    log::debug!(
        "extract_program_from_graph: found {} programs, {} functions",
        collected_programs.len(),
        collected_functions.len()
    );

    // デバッグ: 収集されたカーネルの詳細を表示
    for (i, program) in collected_programs.iter().enumerate() {
        if let AstNode::Program { functions, .. } = program {
            for func in functions {
                match func {
                    AstNode::Kernel { name, params, .. }
                    | AstNode::Function { name, params, .. } => {
                        log::debug!(
                            "  Program {}: kernel/function '{}' with {} params",
                            i,
                            name.as_deref().unwrap_or("unnamed"),
                            params.len()
                        );
                    }
                    _ => {}
                }
            }
        }
    }
    for (i, func) in collected_functions.iter().enumerate() {
        if let AstNode::Function { name, params, .. } = func {
            log::debug!(
                "  Function {}: '{}' with {} params",
                i,
                name.as_deref().unwrap_or("unnamed"),
                params.len()
            );
        }
    }

    // 単一のCustom(Program)のみの場合はそのまま返す
    if collected_programs.len() == 1 && collected_functions.is_empty() {
        log::debug!("Extracting single Custom(Program) directly from graph");
        return collected_programs[0].clone();
    }

    // Custom(Function) がある場合は Lowerer を使用
    // (LoweringSuggester生成のFunctionにはプレースホルダーのみでパラメータがないため、
    //  Lowerer で正しいパラメータを追加する必要がある)
    if !collected_functions.is_empty() {
        log::debug!(
            "Custom(Function) found ({} functions), using Lowerer to create Program",
            collected_functions.len()
        );
        return crate::lowerer::lower(graph);
    }

    // 複数のCustom(Program)がある場合、全てを1つのProgramに統合
    if collected_programs.len() > 1 {
        log::debug!(
            "Merging {} Custom(Program) nodes into single Program",
            collected_programs.len()
        );
        return merge_customs_into_program(&collected_programs, &[]);
    }

    // Custom(Program/Function)がない場合はLowererを使用
    log::debug!("No Custom nodes found, using Lowerer to create Program");
    crate::lowerer::lower(graph)
}

/// 複数のCustomノード（Program/Function）を1つのProgramに統合
///
/// 複数の出力がある場合、同じカーネルが異なるProgram内に含まれることがあります。
/// この関数はカーネルの重複を検出し、1つのProgramに統合します。
fn merge_customs_into_program(programs: &[&AstNode], functions: &[&AstNode]) -> AstNode {
    use crate::ast::helper::{assign, block, function, var};
    use crate::ast::{DType, Mutability, Scope, VarDecl};
    use std::collections::{HashMap, HashSet};

    let mut all_kernels: Vec<AstNode> = Vec::new();
    let mut kernel_names: HashSet<String> = HashSet::new();
    // カーネルのbody文字列をキーにして重複を検出
    let mut kernel_bodies: HashMap<String, String> = HashMap::new();
    let mut main_statements: Vec<AstNode> = Vec::new();
    let mut main_params: Vec<VarDecl> = Vec::new();
    let mut seen_param_names: HashSet<String> = HashSet::new();
    let mut allocated_buffers: HashSet<String> = HashSet::new();

    /// カーネルのbodyから重複検出用のキーを生成
    fn kernel_body_key(body: &AstNode) -> String {
        format!("{:?}", body)
    }

    /// カーネルが有効（パラメータを持つ）かどうかを確認
    fn is_valid_kernel(params: &[VarDecl]) -> bool {
        // パラメータがない場合は無効
        // 有効なカーネルには少なくとも1つのPtr型パラメータがあるはず
        !params.is_empty() && params.iter().any(|p| matches!(p.dtype, DType::Ptr(_)))
    }

    // Programsからカーネルを抽出
    for program in programs {
        if let AstNode::Program {
            functions: funcs, ..
        } = program
        {
            for func in funcs {
                match func {
                    AstNode::Kernel {
                        name, params, body, ..
                    }
                    | AstNode::Function {
                        name, params, body, ..
                    } => {
                        let func_name = name.clone().unwrap_or_else(|| "unknown".to_string());

                        // harp_mainはスキップ（後で新しく生成する）
                        if func_name == "harp_main" {
                            // main関数のパラメータを収集
                            for param in params {
                                if !seen_param_names.contains(&param.name) {
                                    main_params.push(param.clone());
                                    seen_param_names.insert(param.name.clone());
                                }
                            }
                            continue;
                        }

                        // パラメータを持たないカーネルはスキップ（プレースホルダーの可能性）
                        if !is_valid_kernel(params) {
                            log::debug!(
                                "Skipping kernel '{}' with invalid/no parameters",
                                func_name
                            );
                            continue;
                        }

                        // カーネルのbodyで重複を検出
                        let body_key = kernel_body_key(body);
                        if let Some(existing_name) = kernel_bodies.get(&body_key) {
                            log::debug!(
                                "Skipping duplicate kernel '{}' (same as '{}')",
                                func_name,
                                existing_name
                            );
                            continue;
                        }

                        // 元の名前が既に使われている場合のみユニークな名前を生成
                        let final_name = if kernel_names.contains(&func_name) {
                            let mut counter = 1;
                            loop {
                                let candidate = format!("{}__{}", func_name, counter);
                                if !kernel_names.contains(&candidate) {
                                    break candidate;
                                }
                                counter += 1;
                            }
                        } else {
                            func_name.clone()
                        };

                        kernel_names.insert(final_name.clone());
                        kernel_bodies.insert(body_key, final_name.clone());

                        // 名前を更新してカーネルを追加
                        let updated_kernel = match func {
                            AstNode::Kernel {
                                params,
                                return_type,
                                body,
                                thread_group_size,
                                ..
                            } => AstNode::Kernel {
                                name: Some(final_name),
                                params: params.clone(),
                                return_type: return_type.clone(),
                                body: body.clone(),
                                thread_group_size: *thread_group_size,
                            },
                            AstNode::Function {
                                params,
                                return_type,
                                body,
                                ..
                            } => AstNode::Kernel {
                                name: Some(final_name),
                                params: params.clone(),
                                return_type: return_type.clone(),
                                body: body.clone(),
                                thread_group_size: 64,
                            },
                            _ => continue,
                        };
                        all_kernels.push(updated_kernel);
                    }
                    _ => {}
                }
            }
        }
    }

    // Functionsをカーネルに変換
    for func in functions {
        if let AstNode::Function {
            name,
            params,
            return_type,
            body,
        } = func
        {
            // パラメータを持たないFunctionはスキップ
            if !is_valid_kernel(params) {
                log::debug!(
                    "Skipping function '{}' with invalid/no parameters",
                    name.as_deref().unwrap_or("unnamed")
                );
                continue;
            }

            // カーネルのbodyで重複を検出
            let body_key = kernel_body_key(body);
            if kernel_bodies.contains_key(&body_key) {
                log::debug!(
                    "Skipping duplicate function '{}'",
                    name.as_deref().unwrap_or("unnamed")
                );
                continue;
            }

            let func_name = name.clone().unwrap_or_else(|| "kernel".to_string());
            let final_name = if kernel_names.contains(&func_name) {
                let mut counter = 1;
                loop {
                    let candidate = format!("{}__{}", func_name, counter);
                    if !kernel_names.contains(&candidate) {
                        break candidate;
                    }
                    counter += 1;
                }
            } else {
                func_name
            };

            kernel_names.insert(final_name.clone());
            kernel_bodies.insert(body_key, final_name.clone());

            all_kernels.push(AstNode::Kernel {
                name: Some(final_name),
                params: params.clone(),
                return_type: return_type.clone(),
                body: body.clone(),
                thread_group_size: 64,
            });
        }
    }

    // main関数のパラメータを収集（全カーネルから）
    for kernel in &all_kernels {
        if let AstNode::Kernel { params, .. } = kernel {
            for param in params {
                // input/output/shape変数をmain関数のパラメータに追加
                if !seen_param_names.contains(&param.name) {
                    // 中間バッファー（tmp）は除外
                    if !param.name.starts_with("tmp") {
                        main_params.push(param.clone());
                        seen_param_names.insert(param.name.clone());
                    }
                }
            }
        }
    }

    // 各カーネルを呼び出すmain関数を生成
    let mut scope = Scope::new();

    // 中間バッファーの確保（tmpで始まるパラメータ）
    for kernel in &all_kernels {
        if let AstNode::Kernel { params, .. } = kernel {
            for param in params {
                if param.name.starts_with("tmp") && !allocated_buffers.contains(&param.name) {
                    // 中間バッファーを確保
                    if let DType::Ptr(inner) = &param.dtype {
                        let _ = scope.declare(
                            param.name.clone(),
                            param.dtype.clone(),
                            Mutability::Mutable,
                        );
                        // サイズは不明なため、大きめに確保（実際のサイズはカーネルのshapeから計算すべき）
                        let alloc_expr = AstNode::Allocate {
                            dtype: inner.clone(),
                            size: Box::new(crate::ast::helper::const_int(1024 * 1024)), // 仮のサイズ
                        };
                        main_statements.push(assign(&param.name, alloc_expr));
                        allocated_buffers.insert(param.name.clone());
                    }
                }
            }
        }
    }

    // 各カーネルを呼び出す
    for kernel in &all_kernels {
        if let AstNode::Kernel { name, params, .. } = kernel {
            let kernel_name = name.clone().unwrap_or_else(|| "unknown".to_string());
            let args: Vec<AstNode> = params.iter().map(|p| var(&p.name)).collect();
            main_statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });

            // カーネル呼び出し後にバリアを挿入
            main_statements.push(AstNode::Barrier);
        }
    }

    // 中間バッファーの解放
    for buffer_name in &allocated_buffers {
        main_statements.push(AstNode::Deallocate {
            ptr: Box::new(var(buffer_name)),
        });
    }

    // main関数を作成
    let main_body = block(main_statements, scope);
    let main_fn = function(
        Some("harp_main"),
        main_params,
        DType::Tuple(vec![]),
        main_body,
    );

    // すべてをProgramにまとめる
    let mut all_functions = all_kernels;
    all_functions.push(main_fn);

    AstNode::Program {
        functions: all_functions,
        entry_point: "harp_main".to_string(),
    }
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
            enable_kernel_merge: false, // デフォルトで無効（バグ修正後に有効化予定）
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
        // グラフ最適化（Phase 1 + Phase 2）
        let optimized_graph = self.optimize_graph_internal(graph);

        // グラフからAST Programを抽出（Custom(Program)があれば直接使用）
        let program = extract_program_from_graph(optimized_graph);

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
        // グラフ最適化（Phase 1 + Phase 2）
        let optimized_graph = self.optimize_graph_internal(graph);

        // グラフからAST Programを抽出（Custom(Program)があれば直接使用）
        let program = extract_program_from_graph(optimized_graph);

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
        // グラフ最適化（Phase 1 + Phase 2）
        let optimized_graph = self.optimize_graph_internal(graph);

        // グラフからAST Programを抽出（Custom(Program)があれば直接使用）
        let program = extract_program_from_graph(optimized_graph);

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
            // ViewMergeSuggesterはView(Const)パターンもマージする
            Box::new(ViewMergeSuggester::new()),
            Box::new(TilingSuggester::with_default_tile_sizes()),
            Box::new(ContiguousInsertionSuggester::new()),
            Box::new(FusionSuggester::new()),
            // LoweringSuggesterは他の最適化後にlowering
            Box::new(LoweringSuggester::new()),
            // BufferAbsorptionSuggesterはCustomノードにBufferを取り込む
            Box::new(BufferAbsorptionSuggester::new()),
            // SinkAbsorptionSuggesterはCustom(Function)をSinkに吸収
            Box::new(SinkAbsorptionSuggester::new()),
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
    ///
    /// 2段階最適化に対応:
    /// - Phase 1: 一般的なグラフ最適化（fusion, lowering等）
    /// - Phase 2: カーネルマージ（enable_kernel_mergeが有効な場合）
    fn optimize_graph_internal(&mut self, graph: Graph) -> Graph {
        // Phase 1: 一般的なグラフ最適化
        let suggester = Self::create_graph_suggester();
        let estimator = SimpleCostEstimator::new();
        let optimizer = self.create_graph_optimizer(suggester, estimator);

        let (phase1_graph, phase1_history) = optimizer.optimize_with_history(graph);
        if self.collect_histories {
            self.histories.graph = Some(phase1_history);
        }

        // Phase 2: カーネルマージ（有効な場合）
        if self.enable_kernel_merge {
            // Phase 1完了後のCustom(Function)の数を確認
            let custom_function_count = count_custom_functions(&phase1_graph);
            log::info!(
                "Phase 1 completed: {} Custom(Function) nodes found. Starting kernel merge.",
                custom_function_count
            );

            let merge_suggester =
                CompositeSuggester::new(vec![Box::new(KernelMergeSuggester::new())]);
            let merge_estimator = KernelMergeCostEstimator::new();
            let merge_optimizer = BeamSearchGraphOptimizer::new(merge_suggester, merge_estimator)
                .with_beam_width(self.graph_config.beam_width)
                .with_max_steps(self.graph_config.max_steps / 2) // カーネルマージは少ないステップで十分
                .with_progress(self.graph_config.show_progress);

            let (phase2_graph, phase2_history) =
                merge_optimizer.optimize_with_history(phase1_graph);
            if self.collect_histories {
                self.histories.graph_phase2 = Some(phase2_history);
            }

            phase2_graph
        } else {
            phase1_graph
        }
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

/// グラフ内のCustom(Function)ノードの数をカウント
fn count_custom_functions(graph: &Graph) -> usize {
    use std::collections::HashSet;

    fn visit(
        node: &crate::graph::GraphNode,
        visited: &mut HashSet<*const crate::graph::GraphNodeData>,
        count: &mut usize,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        if let crate::graph::GraphOp::Custom { ast, .. } = &node.op
            && matches!(ast, AstNode::Function { .. })
        {
            *count += 1;
        }

        for src in &node.src {
            visit(src, visited, count);
        }
    }

    let mut visited = HashSet::new();
    let mut count = 0;
    for output in graph.outputs().values() {
        visit(output, &mut visited, &mut count);
    }
    count
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
