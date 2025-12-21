//! 実行コンテキスト
//!
//! バックエンドを抽象化し、計算グラフのコンパイルと実行を管理します。

use crate::cache::{GraphSignature, KernelCache};
use harp_core::ast::DType;
use harp_core::backend::pipeline::{CompiledKernel, KernelSourceRenderer};
use harp_core::backend::{Buffer, Compiler, Device, Kernel, Pipeline};
use harp_core::graph::{Graph, GraphNode};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use thiserror::Error;

// ============================================================================
// エラー型
// ============================================================================

/// 実行コンテキストのエラー
#[derive(Debug, Error)]
pub enum ContextError<CE: std::error::Error, BE: std::error::Error> {
    /// コンパイルエラー
    #[error("compilation error: {0}")]
    Compilation(CE),
    /// カーネル実行エラー
    #[error("kernel execution error: {0}")]
    KernelExecution(String),
    /// バッファエラー
    #[error("buffer error: {0}")]
    Buffer(BE),
    /// 形状不一致エラー
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// 入力バッファが見つからない
    #[error("input buffer not found: {0}")]
    InputNotFound(String),
}

// ============================================================================
// ExecutionConfig - 実行設定
// ============================================================================

/// 実行設定
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// キャッシュの有効化
    pub enable_cache: bool,
    /// グラフ最適化のビーム幅
    pub graph_beam_width: usize,
    /// AST最適化のビーム幅
    pub ast_beam_width: usize,
    /// 最大最適化ステップ数
    pub max_steps: usize,
    /// 高速数学演算を有効化
    pub fast_math: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            graph_beam_width: 4,
            ast_beam_width: 4,
            max_steps: 5000,
            fast_math: false,
        }
    }
}

// ============================================================================
// ExecutionContext - 実行コンテキスト
// ============================================================================

/// 実行コンテキスト
///
/// バックエンドを抽象化し、計算グラフのコンパイルと実行を管理します。
/// ジェネリクスでRenderer、Device、Compilerを受け取り、様々なバックエンドに対応できます。
///
/// # 型パラメータ
/// - `R`: レンダラー（KernelSourceRendererを実装）
/// - `Dev`: デバイス（Deviceを実装）
/// - `Comp`: コンパイラー（Compilerを実装）
/// - `Buf`: バッファ（Bufferを実装）
pub struct ExecutionContext<R, Dev, Comp, Buf>
where
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    /// パイプライン
    pipeline: RwLock<Pipeline<R, Dev, Comp>>,
    /// カーネルキャッシュ（シグネチャ → コンパイル済みカーネル情報）
    kernel_cache: KernelCache<CompiledKernel<Comp::Kernel, Buf>>,
    /// 実行設定
    config: ExecutionConfig,
    /// 型マーカー
    _marker: PhantomData<Buf>,
}

impl<R, Dev, Comp, Buf> ExecutionContext<R, Dev, Comp, Buf>
where
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    /// 新しい実行コンテキストを作成
    pub fn new(renderer: R, compiler: Comp, device: Dev) -> Arc<Self> {
        Self::with_config(renderer, compiler, device, ExecutionConfig::default())
    }

    /// 設定を指定して実行コンテキストを作成
    pub fn with_config(
        renderer: R,
        compiler: Comp,
        device: Dev,
        config: ExecutionConfig,
    ) -> Arc<Self> {
        let mut pipeline = Pipeline::new(renderer, compiler, device);

        // パイプライン設定を適用
        {
            let pipeline_config = pipeline.config_mut();
            pipeline_config.graph_beam_width = config.graph_beam_width;
            pipeline_config.ast_beam_width = config.ast_beam_width;
            pipeline_config.max_steps = config.max_steps;
            pipeline_config.fast_math = config.fast_math;
        }

        Arc::new(Self {
            pipeline: RwLock::new(pipeline),
            kernel_cache: KernelCache::new(),
            config,
            _marker: PhantomData,
        })
    }

    /// 設定を取得
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// パイプラインに対してクロージャを実行（読み取り）
    pub fn with_pipeline<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&Pipeline<R, Dev, Comp>) -> T,
    {
        let guard = self.pipeline.read().unwrap();
        f(&guard)
    }

    /// パイプラインに対してクロージャを実行（書き込み）
    pub fn with_pipeline_mut<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut Pipeline<R, Dev, Comp>) -> T,
    {
        let mut guard = self.pipeline.write().unwrap();
        f(&mut guard)
    }

    /// グラフをコンパイル（キャッシュあり）
    pub fn compile(
        &self,
        output_node: &GraphNode,
    ) -> Result<Arc<CompiledKernel<Comp::Kernel, Buf>>, ContextError<Comp::Error, Buf::Error>> {
        // シグネチャを計算
        let sig = GraphSignature::from_output(output_node);

        // キャッシュを確認
        if self.config.enable_cache
            && let Some(compiled) = self.kernel_cache.get(&sig)
        {
            return Ok(compiled);
        }

        // グラフを構築
        let mut graph = Graph::new();
        // 入力ノードを収集してメタデータを登録
        self.register_inputs(&mut graph, output_node);
        // 出力を登録
        graph.output("output", output_node.clone());

        // コンパイル
        let compiled = self
            .pipeline
            .write()
            .unwrap()
            .compile_graph(graph)
            .map_err(ContextError::Compilation)?;

        // キャッシュに挿入
        if self.config.enable_cache {
            Ok(self.kernel_cache.insert(sig, compiled))
        } else {
            Ok(Arc::new(compiled))
        }
    }

    /// コンパイル済みカーネルを実行
    ///
    /// 入力バッファマップを受け取り、出力バッファを返す
    ///
    /// # 引数
    /// - `compiled`: コンパイル済みカーネル
    /// - `inputs`: 入力バッファマップ（名前 → バッファ）
    /// - `output_shape`: 出力バッファの形状
    /// - `output_dtype`: 出力バッファのデータ型
    pub fn execute(
        &self,
        compiled: &CompiledKernel<Comp::Kernel, Buf>,
        inputs: HashMap<String, Buf>,
        output_shape: Vec<usize>,
        output_dtype: DType,
    ) -> Result<Buf, ContextError<Comp::Error, Buf::Error>> {
        // 出力バッファを割り当て
        let mut output_buffer = self
            .with_pipeline(|p| Buf::allocate(p.device(), output_shape, output_dtype))
            .map_err(ContextError::Buffer)?;

        // 入力バッファを順序付きで収集（参照のベクタ）
        let mut input_buffers = Vec::new();
        for input_sig in &compiled.signature.inputs {
            let buffer = inputs
                .get(&input_sig.name)
                .ok_or_else(|| ContextError::InputNotFound(input_sig.name.clone()))?;
            input_buffers.push(buffer);
        }

        // カーネルを実行
        compiled
            .kernel
            .execute(&input_buffers, &mut [&mut output_buffer])
            .map_err(|e| ContextError::KernelExecution(e.to_string()))?;

        Ok(output_buffer)
    }

    /// 入力ノードをグラフに登録
    fn register_inputs(&self, graph: &mut Graph, output_node: &GraphNode) {
        use harp_core::graph::GraphOp;
        use std::collections::HashSet;

        fn visit(node: &GraphNode, graph: &mut Graph, visited: &mut HashSet<usize>) {
            let ptr = node.as_ptr() as usize;
            if !visited.insert(ptr) {
                return;
            }

            // Bufferノードは入力として登録
            if let GraphOp::Buffer { name } = &node.op {
                graph.register_input_meta(
                    name.clone(),
                    node.dtype.clone(),
                    node.view.shape().to_vec(),
                );
            }

            for child in &node.src {
                visit(child, graph, visited);
            }
        }

        let mut visited = HashSet::new();
        visit(output_node, graph, &mut visited);
    }

    /// キャッシュをクリア
    pub fn clear_cache(&self) {
        self.kernel_cache.clear();
    }

    /// キャッシュのエントリ数を取得
    pub fn cache_size(&self) -> usize {
        self.kernel_cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        assert!(config.enable_cache);
        assert_eq!(config.graph_beam_width, 4);
        assert_eq!(config.ast_beam_width, 4);
        assert!(!config.fast_math);
    }
}
