use std::collections::HashMap;

use crate::backend::{Compiler, Pipeline, Renderer};
use crate::graph::Graph;
use crate::opt::ast::OptimizationHistory as AstOptimizationHistory;
use crate::opt::graph::OptimizationHistory as GraphOptimizationHistory;

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
}

impl<R, C> GenericPipeline<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 新しいGenericPipelineを作成
    pub fn new(renderer: R, compiler: C) -> Self {
        Self {
            renderer,
            compiler,
            kernel_cache: HashMap::new(),
            last_graph_optimization_history: None,
            last_ast_optimization_history: None,
        }
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

        fn render(&self, _program: &crate::ast::Program) -> Self::CodeRepr {
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
}
