//! バックエンド抽象化
//!
//! 各バックエンド（Metal, OpenCL等）が実装する `Backend` トレイトを提供します。
//! これにより、バックエンド固有の型パラメータをひとつにまとめ、
//! グローバルコンテキストの管理を可能にします。

use crate::context::ExecutionContext;
use harp_core::backend::pipeline::KernelSourceRenderer;
use harp_core::backend::{Buffer, Compiler, Device, Kernel};
use std::sync::Arc;

/// バックエンド抽象化トレイト
///
/// 各GPUバックエンド（Metal, OpenCL, CUDA等）はこのトレイトを実装し、
/// 関連する型とグローバルコンテキストの取得方法を定義します。
///
/// # 例（Metalバックエンド実装）
///
/// ```ignore
/// use std::sync::OnceLock;
///
/// pub struct MetalBackend;
///
/// impl Backend for MetalBackend {
///     type Renderer = MetalRenderer;
///     type Device = MetalDevice;
///     type Compiler = MetalCompiler;
///     type Buffer = MetalBuffer;
///
///     fn global_context() -> Arc<ExecutionContext<...>> {
///         static CTX: OnceLock<Arc<ExecutionContext<...>>> = OnceLock::new();
///         CTX.get_or_init(|| {
///             let device = MetalDevice::default();
///             let compiler = MetalCompiler::new(&device);
///             let renderer = MetalRenderer::new();
///             ExecutionContext::new(renderer, compiler, device)
///         }).clone()
///     }
/// }
/// ```
pub trait Backend: 'static + Sized {
    /// カーネルソースレンダラー
    type Renderer: KernelSourceRenderer + Clone + 'static;

    /// デバイス
    type Device: Device + 'static;

    /// コンパイラ（Kernel::Buffer = Self::Bufferを満たす必要がある）
    type Compiler: Compiler<Dev = Self::Device, Kernel = Self::Kernel> + 'static;

    /// バッファ
    type Buffer: Buffer<Dev = Self::Device> + 'static;

    /// カーネル型
    type Kernel: Kernel<Buffer = Self::Buffer> + Clone + 'static;

    /// グローバル共有コンテキストを取得
    ///
    /// 同一バックエンドでは常に同じコンテキストインスタンスを返します。
    /// 初回呼び出し時にコンテキストが初期化されます。
    fn global_context()
    -> Arc<ExecutionContext<Self::Renderer, Self::Device, Self::Compiler, Self::Buffer>>;
}

/// Backend制約を満たすCompilerのKernel型を取得するヘルパー
pub type BackendKernel<B> = <<B as Backend>::Compiler as Compiler>::Kernel;

/// コンテキストの型エイリアス
pub type Ctx<B> = ExecutionContext<
    <B as Backend>::Renderer,
    <B as Backend>::Device,
    <B as Backend>::Compiler,
    <B as Backend>::Buffer,
>;
