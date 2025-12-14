//! グラフ用ランタイムコスト評価器
//!
//! グラフをLowering・コンパイル・実行して実行時間を計測し、それをコストとして使用します。
//! グラフ最適化のビームサーチにおいて、静的コスト推定よりも正確なコスト評価を提供します。
//!
//! # 設計思想
//!
//! - 静的コストで足切り後、実行時間を計測することで計算コストを削減
//! - 簡易Lowerer（create_simple_lowering_optimizer）を使用して高速にProgram化
//! - キャッシュなしのシンプルな設計（毎回再計測）
//! - バッファファクトリをユーザー定義にすることで柔軟なベンチマーク設定が可能
//!
//! # 注意
//!
//! グラフはLoweringが必要なため、AST版よりも計測コストが高くなります。
//! 足切り候補数を少なめに設定することを推奨します（デフォルト: 5件）。
//!
//! # Example
//!
//! ```ignore
//! use harp::backend::{Compiler, Renderer};
//! use harp::opt::graph::GraphRuntimeCostEstimator;
//!
//! let estimator = GraphRuntimeCostEstimator::new(
//!     renderer,
//!     compiler,
//!     |sig| create_buffers(sig),
//! );
//!
//! let cost_us = estimator.measure(&graph);
//! ```

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use crate::ast::AstNode;
use crate::backend::{Compiler, Kernel, KernelSignature, Renderer};
use crate::graph::{Graph, GraphOp};
use crate::lowerer::create_simple_lowering_optimizer;
use crate::opt::graph::{GraphCostEstimator, GraphOptimizer};

/// グラフ用ランタイムコスト評価器
///
/// グラフをLowering・コンパイル・実行して実行時間（マイクロ秒）を計測します。
/// ジェネリクスでRenderer/Compilerを保持し、様々なバックエンドに対応します。
///
/// # Type Parameters
///
/// * `R` - レンダラーの型（AST → ソースコード）
/// * `C` - コンパイラの型（ソースコード → カーネル）
///
/// # 内部可変性
///
/// `RefCell`を使用してRenderer/Compilerを管理しているため、`&self`で呼び出せます。
/// ただし、シングルスレッド環境での使用を前提としています。
pub struct GraphRuntimeCostEstimator<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// レンダラー
    renderer: RefCell<R>,
    /// コンパイラ
    compiler: RefCell<C>,
    /// ベンチマーク用バッファを生成するファクトリ関数
    buffer_factory: Arc<dyn Fn(&KernelSignature) -> Vec<C::Buffer> + Send + Sync>,
    /// 計測回数（デフォルト: 5）
    measurement_count: usize,
    /// Loweringの最大ステップ数（デフォルト: 1000）
    lowering_max_steps: usize,
}

impl<R, C> Clone for GraphRuntimeCostEstimator<R, C>
where
    R: Renderer + Clone,
    C: Compiler<CodeRepr = R::CodeRepr> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            renderer: RefCell::new(self.renderer.borrow().clone()),
            compiler: RefCell::new(self.compiler.borrow().clone()),
            buffer_factory: Arc::clone(&self.buffer_factory),
            measurement_count: self.measurement_count,
            lowering_max_steps: self.lowering_max_steps,
        }
    }
}

impl<R, C> GraphRuntimeCostEstimator<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 新しいGraphRuntimeCostEstimatorを作成
    ///
    /// # Arguments
    ///
    /// * `renderer` - ASTをソースコードに変換するレンダラー
    /// * `compiler` - ソースコードをカーネルにコンパイルするコンパイラ
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    ///
    /// # Example
    ///
    /// ```ignore
    /// let estimator = GraphRuntimeCostEstimator::new(
    ///     CRenderer::new(),
    ///     CCompiler::new(),
    ///     |sig| {
    ///         // シグネチャに基づいてバッファを生成
    ///         sig.inputs.iter()
    ///             .chain(sig.outputs.iter())
    ///             .map(|buf_sig| create_buffer(&buf_sig))
    ///             .collect()
    ///     },
    /// );
    /// ```
    pub fn new<F>(renderer: R, compiler: C, buffer_factory: F) -> Self
    where
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + Send + Sync + 'static,
    {
        Self {
            renderer: RefCell::new(renderer),
            compiler: RefCell::new(compiler),
            buffer_factory: Arc::new(buffer_factory),
            measurement_count: 5, // グラフ用はデフォルトを低めに
            lowering_max_steps: 1000,
        }
    }

    /// 計測回数を設定
    ///
    /// デフォルトは5回です。計測回数を増やすと精度が向上しますが、
    /// 最適化にかかる時間も増加します。
    pub fn with_measurement_count(mut self, count: usize) -> Self {
        self.measurement_count = count.max(1);
        self
    }

    /// Loweringの最大ステップ数を設定
    ///
    /// 簡易Loweringで使用する最大ステップ数を指定します。
    /// デフォルトは1000です。
    pub fn with_lowering_max_steps(mut self, steps: usize) -> Self {
        self.lowering_max_steps = steps;
        self
    }

    /// グラフの実行時間を計測（マイクロ秒）
    ///
    /// 1. Lowering（簡易Lowerer使用）
    /// 2. コンパイル
    /// 3. バッファ生成
    /// 4. 実行時間計測（measurement_count回の平均）
    ///
    /// # Returns
    ///
    /// 実行時間（マイクロ秒）。Lowering/コンパイル失敗時は`f32::INFINITY`を返します。
    pub fn measure(&self, graph: &Graph) -> f32 {
        // Lowering
        let program = match self.lower_graph(graph) {
            Some(p) => p,
            None => {
                log::debug!("Failed to lower graph for measurement");
                return f32::INFINITY;
            }
        };

        // シグネチャを作成
        let signature = crate::lowerer::create_signature(graph);

        // レンダリング
        let code = self.renderer.borrow().render(&program);

        // コンパイル
        let kernel = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.compiler.borrow_mut().compile(&code, signature.clone())
        })) {
            Ok(k) => k,
            Err(_) => {
                log::debug!("Failed to compile graph for measurement");
                return f32::INFINITY;
            }
        };

        // バッファ生成
        let mut buffers = (self.buffer_factory)(&signature);

        // 実行時間計測
        let mut total_time_us = 0u128;

        for _ in 0..self.measurement_count {
            let start = Instant::now();

            // カーネル実行
            // SAFETY: バッファは適切に初期化されている前提
            let mut buffer_refs: Vec<&mut C::Buffer> = buffers.iter_mut().collect();
            let result = unsafe { kernel.execute(&mut buffer_refs) };

            if result.is_err() {
                log::debug!("Kernel execution failed during measurement");
                return f32::INFINITY;
            }

            total_time_us += start.elapsed().as_micros();
        }

        (total_time_us as f32) / (self.measurement_count as f32)
    }

    /// グラフをLoweringしてProgramを取得
    fn lower_graph(&self, graph: &Graph) -> Option<AstNode> {
        // 簡易Loweringを実行
        let lowering_optimizer = create_simple_lowering_optimizer(self.lowering_max_steps);
        let (lowered_graph, _) = lowering_optimizer.optimize_with_history(graph.clone());

        // ProgramRootノードからProgramを取得
        if let Some(root) = lowered_graph.program_root()
            && let GraphOp::ProgramRoot { ast, .. } = &root.op
            && matches!(ast, AstNode::Program { .. })
        {
            return Some(ast.clone());
        }

        // Kernel(Program)ノードを探す
        for output in lowered_graph.outputs().values() {
            if let GraphOp::Kernel { ast, .. } = &output.op
                && matches!(ast, AstNode::Program { .. })
            {
                return Some(ast.clone());
            }
        }

        None
    }
}

impl<R, C> GraphCostEstimator for GraphRuntimeCostEstimator<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    fn estimate(&self, graph: &Graph) -> f32 {
        self.measure(graph)
    }
}

#[cfg(test)]
mod tests {
    // テストはCバックエンドが利用可能な環境でのみ実行
    // 統合テストで実施
}
