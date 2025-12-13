//! ランタイムコスト評価器
//!
//! ASTを実際にコンパイル・実行して実行時間を計測し、それをコストとして使用します。
//! ビームサーチ最適化において、静的コスト推定よりも正確なコスト評価を提供します。
//!
//! # 設計思想
//!
//! - 静的コストで足切り後、実行時間を計測することで計算コストを削減
//! - キャッシュなしのシンプルな設計（毎回再計測）
//! - バッファファクトリをユーザー定義にすることで柔軟なベンチマーク設定が可能
//!
//! # Example
//!
//! ```ignore
//! use harp::backend::{Compiler, Renderer};
//! use harp::opt::ast::RuntimeCostEstimator;
//!
//! let estimator = RuntimeCostEstimator::new(
//!     renderer,
//!     compiler,
//!     signature,
//!     |sig| create_buffers(sig),
//! );
//!
//! let cost_us = estimator.measure(&ast);
//! ```

use std::cell::RefCell;
use std::time::Instant;

use crate::ast::AstNode;
use crate::backend::{Compiler, Kernel, KernelSignature, Renderer};

/// ランタイムコスト評価器
///
/// ASTを実際にコンパイル・実行して実行時間（マイクロ秒）を計測します。
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
pub struct RuntimeCostEstimator<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// レンダラー
    renderer: RefCell<R>,
    /// コンパイラ
    compiler: RefCell<C>,
    /// カーネルシグネチャ（コンパイル時に必要）
    signature: KernelSignature,
    /// ベンチマーク用バッファを生成するファクトリ関数
    buffer_factory: Box<dyn Fn(&KernelSignature) -> Vec<C::Buffer>>,
    /// 計測回数（デフォルト: 10）
    measurement_count: usize,
}

impl<R, C> RuntimeCostEstimator<R, C>
where
    R: Renderer,
    C: Compiler<CodeRepr = R::CodeRepr>,
{
    /// 新しいRuntimeCostEstimatorを作成
    ///
    /// # Arguments
    ///
    /// * `renderer` - ASTをソースコードに変換するレンダラー
    /// * `compiler` - ソースコードをカーネルにコンパイルするコンパイラ
    /// * `signature` - カーネルシグネチャ（入出力バッファの形状情報）
    /// * `buffer_factory` - ベンチマーク用バッファを生成する関数
    ///
    /// # Example
    ///
    /// ```ignore
    /// let estimator = RuntimeCostEstimator::new(
    ///     CRenderer::new(),
    ///     CCompiler::new(),
    ///     signature,
    ///     |sig| {
    ///         // シグネチャに基づいてバッファを生成
    ///         sig.inputs.iter()
    ///             .chain(sig.outputs.iter())
    ///             .map(|buf_sig| create_buffer(&buf_sig))
    ///             .collect()
    ///     },
    /// );
    /// ```
    pub fn new<F>(renderer: R, compiler: C, signature: KernelSignature, buffer_factory: F) -> Self
    where
        F: Fn(&KernelSignature) -> Vec<C::Buffer> + 'static,
    {
        Self {
            renderer: RefCell::new(renderer),
            compiler: RefCell::new(compiler),
            signature,
            buffer_factory: Box::new(buffer_factory),
            measurement_count: 10,
        }
    }

    /// 計測回数を設定
    ///
    /// デフォルトは10回です。計測回数を増やすと精度が向上しますが、
    /// 最適化にかかる時間も増加します。
    pub fn with_measurement_count(mut self, count: usize) -> Self {
        self.measurement_count = count.max(1); // 最低1回
        self
    }

    /// ASTの実行時間を計測（マイクロ秒）
    ///
    /// 1. コンパイル
    /// 2. バッファ生成
    /// 3. 実行時間計測（measurement_count回の平均）
    ///
    /// # Returns
    ///
    /// 実行時間（マイクロ秒）。コンパイル失敗時は`f32::INFINITY`を返します。
    pub fn measure(&self, ast: &AstNode) -> f32 {
        // レンダリング
        let code = self.renderer.borrow().render(ast);

        // コンパイル
        let kernel = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.compiler
                .borrow_mut()
                .compile(&code, self.signature.clone())
        })) {
            Ok(k) => k,
            Err(_) => {
                log::warn!("Failed to compile AST for measurement");
                return f32::INFINITY;
            }
        };

        // バッファ生成
        let mut buffers = (self.buffer_factory)(&self.signature);

        // 実行時間計測
        let mut total_time_us = 0u128;

        for _ in 0..self.measurement_count {
            let start = Instant::now();

            // カーネル実行
            // SAFETY: バッファは適切に初期化されている前提
            let mut buffer_refs: Vec<&mut C::Buffer> = buffers.iter_mut().collect();
            let result = unsafe { kernel.execute(&mut buffer_refs) };

            if result.is_err() {
                log::warn!("Kernel execution failed during measurement");
                return f32::INFINITY;
            }

            total_time_us += start.elapsed().as_micros();
        }

        (total_time_us as f32) / (self.measurement_count as f32)
    }
}

#[cfg(test)]
mod tests {
    // テストはCバックエンドが利用可能な環境でのみ実行
    // 統合テストで実施
}
