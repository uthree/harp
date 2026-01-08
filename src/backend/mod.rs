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
//! - `harp-backend-c`: Pure C backend for CPU
//! - `harp-backend-metal`: Metal backend for macOS
//! - `harp-backend-opencl`: OpenCL backend for cross-platform GPU
//!
//! ## Usage
//!
//! ```ignore
//! use harp::backend::{Pipeline, HarpDevice, set_device};
//!
//! // Set up a device
//! let device = HarpDevice::auto()?;
//! set_device(device);
//!
//! // Compile and run a program
//! let kernel = Pipeline::compile(&program)?;
//! ```

pub mod cache;
pub mod device;
pub mod global;
pub mod pipeline;
pub mod renderer;
pub mod sequence;
pub mod traits;

// Re-export core traits
pub use traits::{
    Buffer, Compiler, Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, Kernel,
    KernelConfig, OpKind, SimdCapability, TypedBuffer,
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

// Re-export Renderer trait and types
pub use renderer::{
    CLikeRenderer, GenericRenderer, OptimizationLevel, Renderer, extract_buffer_placeholders,
};

// Re-export global device management
pub use global::{
    DeviceKind, clear_default_device, get_default_device, get_default_device_kind,
    has_default_device, set_default_device, set_device, set_device_str, with_device,
};

// Re-export device types
pub use device::{BackendRegistry, DeviceError, HarpDevice, register_backend};

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
