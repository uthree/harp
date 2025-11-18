pub mod buffer;
pub mod compiler;
pub mod kernel;
pub mod renderer;

pub use buffer::OpenCLBuffer;
pub use compiler::OpenCLCompiler;
pub use kernel::OpenCLKernel;
pub use renderer::OpenCLRenderer;

/// libloading用のラッパー関数名
pub const LIBLOADING_WRAPPER_NAME: &str = "__harp_entry";

use crate::ast::AstNode;
use crate::backend::{
    pipeline::{optimize_ast_with_history, optimize_graph_with_history, SuggesterFlags},
    Compiler, OptimizationConfig, OptimizationHistories, Pipeline, Renderer,
};
use crate::graph::Graph;
use crate::opt::graph::SimpleCostEstimator;
use std::collections::HashMap;

/// OpenCLバックエンド専用のPipeline
///
/// 並列化・SIMD最適化を含むため、GPU/マルチコア実行に最適化されています。
pub struct OpenCLPipeline {
    renderer: OpenCLRenderer,
    compiler: OpenCLCompiler,
    kernel_cache: HashMap<String, OpenCLKernel>,
    pub histories: OptimizationHistories,
    pub enable_graph_optimization: bool,
    pub graph_config: OptimizationConfig,
    pub enable_ast_optimization: bool,
    pub ast_config: OptimizationConfig,
    pub collect_histories: bool,
}

impl OpenCLPipeline {
    /// 新しいOpenCLPipelineを作成
    pub fn new(renderer: OpenCLRenderer, compiler: OpenCLCompiler) -> Self {
        Self {
            renderer,
            compiler,
            kernel_cache: HashMap::new(),
            histories: OptimizationHistories::default(),
            enable_graph_optimization: false,
            graph_config: OptimizationConfig::default(),
            enable_ast_optimization: false,
            ast_config: OptimizationConfig::default(),
            collect_histories: cfg!(debug_assertions),
        }
    }

    /// キャッシュからKernelを取得
    pub fn get_cached_kernel(&self, key: &str) -> Option<&OpenCLKernel> {
        self.kernel_cache.get(key)
    }

    /// グラフをコンパイルし、結果をキャッシュに保存
    pub fn compile_and_cache(
        &mut self,
        key: String,
        graph: Graph,
    ) -> Result<&OpenCLKernel, String> {
        let kernel = self.compile_graph(graph)?;
        self.kernel_cache.insert(key.clone(), kernel);
        Ok(self.kernel_cache.get(&key).unwrap())
    }

    /// キャッシュをクリア
    pub fn clear_cache(&mut self) {
        self.kernel_cache.clear();
    }

    /// グラフ最適化の内部処理（並列化・SIMD有効）
    fn optimize_graph_internal(&mut self, graph: Graph) -> Graph {
        if !self.enable_graph_optimization {
            return graph;
        }

        let flags = SuggesterFlags::all(); // 並列化・SIMD有効
        let (optimized, history) = optimize_graph_with_history(
            graph,
            flags,
            SimpleCostEstimator::new(),
            self.graph_config.beam_width,
            self.graph_config.max_steps,
            self.graph_config.show_progress,
        );

        if self.collect_histories {
            self.histories.graph = Some(history);
        }
        optimized
    }

    /// AST最適化の内部処理
    fn optimize_ast_internal(&mut self, program: AstNode) -> AstNode {
        if !self.enable_ast_optimization {
            return program;
        }

        let (optimized, history) = optimize_ast_with_history(
            program,
            self.ast_config.beam_width,
            self.ast_config.max_steps,
            self.ast_config.show_progress,
        );

        if self.collect_histories {
            self.histories.ast = Some(history);
        }
        optimized
    }
}

impl Pipeline for OpenCLPipeline {
    type Compiler = OpenCLCompiler;
    type Renderer = OpenCLRenderer;
    type Error = String;

    fn renderer(&self) -> &Self::Renderer {
        &self.renderer
    }

    fn compiler(&mut self) -> &mut Self::Compiler {
        &mut self.compiler
    }

    fn optimize_graph(&self, graph: Graph) -> Graph {
        if !self.enable_graph_optimization {
            return graph;
        }

        // 並列化・SIMD有効のSuggesterを使用
        let flags = SuggesterFlags::all();
        let (optimized, _history) = optimize_graph_with_history(
            graph,
            flags,
            SimpleCostEstimator::new(),
            self.graph_config.beam_width,
            self.graph_config.max_steps,
            self.graph_config.show_progress,
        );
        optimized
    }

    fn optimize_program(&self, program: AstNode) -> AstNode {
        if !self.enable_ast_optimization {
            return program;
        }

        let (optimized, _history) = optimize_ast_with_history(
            program,
            self.ast_config.beam_width,
            self.ast_config.max_steps,
            self.ast_config.show_progress,
        );
        optimized
    }

    fn compile_graph(&mut self, graph: Graph) -> Result<OpenCLKernel, String> {
        // Signatureを作成（最適化前のGraphから）
        let signature = crate::lowerer::Lowerer::create_signature(&graph);

        let optimized_graph = self.optimize_graph_internal(graph);
        let program = self.lower_to_program(optimized_graph);
        let optimized_program = self.optimize_ast_internal(program);
        let mut code = self.renderer().render(&optimized_program);

        // Signatureを設定
        code = OpenCLCode::with_signature(code.into_inner(), signature);

        Ok(self.compiler().compile(&code))
    }
}

/// OpenCL Cコードを表す型
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCLCode {
    code: String,
    signature: crate::backend::KernelSignature,
}

impl OpenCLCode {
    /// 新しい OpenCLCode を作成
    pub fn new(code: String) -> Self {
        Self {
            code,
            signature: crate::backend::KernelSignature::empty(),
        }
    }

    /// シグネチャ付きで新しい OpenCLCode を作成
    pub fn with_signature(code: String, signature: crate::backend::KernelSignature) -> Self {
        Self { code, signature }
    }

    /// 内部の String への参照を取得
    pub fn as_str(&self) -> &str {
        &self.code
    }

    /// 内部の String を取得（所有権を移動）
    pub fn into_inner(self) -> String {
        self.code
    }

    /// シグネチャへの参照を取得
    pub fn signature(&self) -> &crate::backend::KernelSignature {
        &self.signature
    }

    /// コードのバイト数を取得
    pub fn len(&self) -> usize {
        self.code.len()
    }

    /// コードが空かどうか
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// 指定した文字列が含まれているかチェック
    pub fn contains(&self, pat: &str) -> bool {
        self.code.contains(pat)
    }
}

impl From<String> for OpenCLCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<OpenCLCode> for String {
    fn from(code: OpenCLCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for OpenCLCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for OpenCLCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code)
    }
}
