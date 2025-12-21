//! 実行コンテキスト
//!
//! 遅延評価されたArrayをGPU上で実行するためのコンテキストを提供します。

use std::cell::RefCell;

use crate::ArrayError;
use harp_core::ast::DType;
use harp_core::backend::Pipeline;
use harp_core::graph::Graph;

// ============================================================================
// OpenCL ExecutionContext
// ============================================================================

#[cfg(feature = "opencl")]
pub mod opencl {
    use super::*;
    use harp_backend_opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice, OpenCLRenderer};
    use harp_core::backend::sequence::CompiledProgram;
    use harp_core::backend::{Buffer, CompiledKernel, Compiler};

    /// OpenCL用のパイプライン型エイリアス
    pub type OpenCLPipeline = Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler>;

    /// OpenCL用のコンパイル済みカーネル型エイリアス
    pub type OpenCLCompiledKernel =
        CompiledKernel<<OpenCLCompiler as harp_core::backend::Compiler>::Kernel, OpenCLBuffer>;

    /// OpenCL用のコンパイル済みプログラム型エイリアス
    pub type OpenCLCompiledProgram =
        CompiledProgram<<OpenCLCompiler as harp_core::backend::Compiler>::Kernel, OpenCLBuffer>;

    /// OpenCL実行コンテキスト
    ///
    /// スレッドローカルに保持され、遅延初期化されます。
    pub struct OpenCLExecutionContext {
        device: OpenCLDevice,
        pipeline: OpenCLPipeline,
    }

    thread_local! {
        static OPENCL_CONTEXT: RefCell<Option<OpenCLExecutionContext>> = const { RefCell::new(None) };
    }

    impl OpenCLExecutionContext {
        /// 新しい実行コンテキストを作成
        fn new() -> Result<Self, ArrayError> {
            let device = OpenCLDevice::new()
                .map_err(|e| ArrayError::DeviceNotAvailable(format!("OpenCL: {}", e)))?;
            let renderer = OpenCLRenderer::new();
            let compiler = OpenCLCompiler::new();
            let pipeline = Pipeline::new(renderer, compiler, device.clone());

            Ok(Self { device, pipeline })
        }

        /// デバイスを取得
        pub fn device(&self) -> &OpenCLDevice {
            &self.device
        }

        /// グラフをコンパイル（単一カーネル）
        pub fn compile(&mut self, graph: Graph) -> Result<OpenCLCompiledKernel, ArrayError> {
            self.pipeline
                .compile_graph(graph)
                .map_err(|e| ArrayError::Compilation(e.to_string()))
        }

        /// グラフをコンパイル（複数カーネル対応）
        pub fn compile_program(
            &mut self,
            graph: Graph,
        ) -> Result<OpenCLCompiledProgram, ArrayError> {
            self.pipeline
                .compile_program(graph)
                .map_err(|e| ArrayError::Compilation(e.to_string()))
        }

        /// バッファを確保
        pub fn allocate_buffer(
            &self,
            shape: Vec<usize>,
            dtype: DType,
        ) -> Result<OpenCLBuffer, ArrayError> {
            OpenCLBuffer::allocate(&self.device, shape, dtype)
                .map_err(|e| ArrayError::Execution(e.to_string()))
        }
    }

    /// OpenCL実行コンテキストにアクセスして処理を実行
    ///
    /// コンテキストは遅延初期化され、スレッドローカルに保持されます。
    pub fn with_opencl_context<F, R>(f: F) -> Result<R, ArrayError>
    where
        F: FnOnce(&mut OpenCLExecutionContext) -> Result<R, ArrayError>,
    {
        OPENCL_CONTEXT.with(|cell| {
            let mut ctx_opt = cell.borrow_mut();

            // 遅延初期化
            if ctx_opt.is_none() {
                *ctx_opt = Some(OpenCLExecutionContext::new()?);
            }

            // コンテキストを使用
            let ctx = ctx_opt.as_mut().unwrap();
            f(ctx)
        })
    }

    /// OpenCL実行コンテキストがあれば、そのデバイスへの参照をコピーして返す
    ///
    /// バッファ確保などデバイスへの参照が必要な場合に使用します。
    pub fn get_opencl_device() -> Result<OpenCLDevice, ArrayError> {
        OPENCL_CONTEXT.with(|cell| {
            let mut ctx_opt = cell.borrow_mut();

            // 遅延初期化
            if ctx_opt.is_none() {
                *ctx_opt = Some(OpenCLExecutionContext::new()?);
            }

            Ok(ctx_opt.as_ref().unwrap().device.clone())
        })
    }
}

// ============================================================================
// Metal ExecutionContext
// ============================================================================

#[cfg(feature = "metal")]
pub mod metal {
    use super::*;
    use harp_backend_metal::{MetalBuffer, MetalCompiler, MetalDevice, MetalRenderer};
    use harp_core::backend::{Buffer, CompiledKernel};

    /// Metal用のパイプライン型エイリアス
    pub type MetalPipeline = Pipeline<MetalRenderer, MetalDevice, MetalCompiler>;

    /// Metal用のコンパイル済みカーネル型エイリアス
    pub type MetalCompiledKernel =
        CompiledKernel<<MetalCompiler as harp_core::backend::Compiler>::Kernel, MetalBuffer>;

    /// Metal実行コンテキスト
    pub struct MetalExecutionContext {
        device: MetalDevice,
        pipeline: MetalPipeline,
    }

    thread_local! {
        static METAL_CONTEXT: RefCell<Option<MetalExecutionContext>> = const { RefCell::new(None) };
    }

    impl MetalExecutionContext {
        /// 新しい実行コンテキストを作成
        fn new() -> Result<Self, ArrayError> {
            let device = MetalDevice::new()
                .map_err(|e| ArrayError::DeviceNotAvailable(format!("Metal: {}", e)))?;
            let renderer = MetalRenderer::new();
            let compiler = MetalCompiler::new();
            let pipeline = Pipeline::new(renderer, compiler, device.clone());

            Ok(Self { device, pipeline })
        }

        /// デバイスを取得
        pub fn device(&self) -> &MetalDevice {
            &self.device
        }

        /// グラフをコンパイル
        pub fn compile(&mut self, graph: Graph) -> Result<MetalCompiledKernel, ArrayError> {
            self.pipeline
                .compile_graph(graph)
                .map_err(|e| ArrayError::Compilation(e.to_string()))
        }

        /// バッファを確保
        pub fn allocate_buffer(
            &self,
            shape: Vec<usize>,
            dtype: DType,
        ) -> Result<MetalBuffer, ArrayError> {
            MetalBuffer::allocate(&self.device, shape, dtype)
                .map_err(|e| ArrayError::Execution(e.to_string()))
        }
    }

    /// Metal実行コンテキストにアクセスして処理を実行
    pub fn with_metal_context<F, R>(f: F) -> Result<R, ArrayError>
    where
        F: FnOnce(&mut MetalExecutionContext) -> Result<R, ArrayError>,
    {
        METAL_CONTEXT.with(|cell| {
            let mut ctx_opt = cell.borrow_mut();

            // 遅延初期化
            if ctx_opt.is_none() {
                *ctx_opt = Some(MetalExecutionContext::new()?);
            }

            // コンテキストを使用
            let ctx = ctx_opt.as_mut().unwrap();
            f(ctx)
        })
    }

    /// Metal実行コンテキストがあれば、そのデバイスへの参照をコピーして返す
    pub fn get_metal_device() -> Result<MetalDevice, ArrayError> {
        METAL_CONTEXT.with(|cell| {
            let mut ctx_opt = cell.borrow_mut();

            // 遅延初期化
            if ctx_opt.is_none() {
                *ctx_opt = Some(MetalExecutionContext::new()?);
            }

            Ok(ctx_opt.as_ref().unwrap().device.clone())
        })
    }
}
