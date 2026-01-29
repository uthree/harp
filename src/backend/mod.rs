//! Backend module
//!
//! This module provides GPU execution capabilities.
//!
//! ## Architecture
//!
//! The backend is organized into:
//! - **Traits**: Common interfaces for GPU execution (Device, Buffer, etc.)
//! - **Execution**: Pipeline for end-to-end compilation from Graph to executable kernel
//! - **Renderer**: Code generation for C-like languages
//!
//! Backend-specific implementations (C, Metal, OpenCL) are provided by separate crates:
//! - `eclat-backend-c`: Pure C backend for CPU
//! - `eclat-backend-metal`: Metal backend for macOS
//! - `eclat-backend-opencl`: OpenCL backend for cross-platform GPU
//!
//! ## Usage
//!
//! ```ignore
//! use eclat::backend::{Pipeline, EclatDevice, set_device};
//!
//! // Set up a device
//! let device = EclatDevice::auto()?;
//! set_device(device);
//!
//! // Compile and run a program
//! let kernel = Pipeline::compile(&program)?;
//! ```

pub mod cache;
pub mod compile;
pub mod device;
pub mod executor;
pub mod global;
pub mod pipeline;
pub mod renderer;
pub mod sequence;
pub mod traits;

// Re-export core traits
pub use traits::{
    Buffer, Compiler, Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, Kernel,
    KernelConfig, MatrixCapability, OpKind, SimdCapability, TypedBuffer,
};

// Re-export pipeline types
pub use pipeline::{
    DispatchSizeConfig, DispatchSizeExpr, KernelSourceRenderer, OptimizationHistories, Pipeline,
    PipelineConfig,
};

// Re-export unified cache types
pub use cache::{
    CacheEntry, CacheStats, KernelCacheKey, get_cache_stats, get_cached_kernel,
    insert_cached_kernel,
};

// Re-export sequence types
pub use sequence::{ExecutionQuery, ProgramExecutionError};

// Re-export executor types
pub use executor::{ExecutionError, ExecutionResult, execute_graph};

// Re-export Renderer trait and types
pub use renderer::{
    CLikeRenderer, GenericRenderer, OptimizationLevel, Renderer, extract_buffer_placeholders,
};

// Re-export global device management
pub use global::{
    DeviceKind, allocate_buffer_on_default_device, clear_default_device,
    compile_ast_on_default_device, get_default_device, get_default_device_kind, has_default_device,
    set_default_device, set_device, set_device_str, with_device,
};

// Re-export device types
pub use device::{
    BackendRegistry, DeviceError, EclatDevice, allocate_buffer_on_device, compile_ast_on_device,
    register_backend,
};

// Re-export compilation pipeline
pub use compile::{CompilationPipeline, OptimizationConfig, mark_parallel_for_openmp};

/// 最適化のターゲットバックエンド
///
/// 最適化がどのバックエンドを対象として行われたかを示す。
/// 可視化ツールでレンダラーを自動選択するために使用される。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TargetBackend {
    /// 汎用Cバックエンド（CPU）
    #[default]
    Generic,
    /// Metal (macOS GPU)
    Metal,
    /// OpenCL (クロスプラットフォームGPU)
    OpenCL,
    /// CUDA (NVIDIA GPU)
    Cuda,
    /// OpenMP (CPU並列)
    OpenMP,
}

impl std::fmt::Display for TargetBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetBackend::Generic => write!(f, "Generic"),
            TargetBackend::Metal => write!(f, "Metal"),
            TargetBackend::OpenCL => write!(f, "OpenCL"),
            TargetBackend::Cuda => write!(f, "CUDA"),
            TargetBackend::OpenMP => write!(f, "OpenMP"),
        }
    }
}

/// カーネルのシグネチャ（入出力バッファの形状情報）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSignature {
    pub inputs: Vec<BufferSignature>,
    pub outputs: Vec<BufferSignature>,
}

impl KernelSignature {
    /// 新しいKernelSignatureを作成
    pub fn new(inputs: Vec<BufferSignature>, outputs: Vec<BufferSignature>) -> Self {
        Self { inputs, outputs }
    }

    /// 空のKernelSignatureを作成
    pub fn empty() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

/// バッファのシグネチャ（名前と形状）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSignature {
    pub name: String,
    pub shape: Vec<crate::shape::Expr>,
}

impl BufferSignature {
    /// 新しいBufferSignatureを作成
    pub fn new(name: String, shape: Vec<crate::shape::Expr>) -> Self {
        Self { name, shape }
    }
}
