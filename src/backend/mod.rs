//! Backend module
//!
//! This module provides GPU execution capabilities.
//!
//! ## Architecture
//!
//! The backend is organized into:
//! - **Traits**: Common interfaces for GPU execution (Device, Buffer, etc.)
//! - **Execution**: Pipeline for end-to-end compilation from Graph to executable kernel
//!
//! ## Renderers
//!
//! Renderers are now in `crate::renderer` module and are always available
//! without feature flags. Use `harp::renderer::{OpenCLRenderer, MetalRenderer}`.
//!
//! ## Usage
//!
//! ```ignore
//! use harp::backend::{Pipeline, Device, Compiler};
//!
//! // Enable backends via feature flags:
//! // - opencl: OpenCL backend
//! // - metal: Metal backend (macOS only)
//! ```

pub mod global;
pub mod pipeline;
pub mod sequence;
pub mod traits;

/// OpenCL バックエンド（`opencl` feature required）
#[cfg(feature = "opencl")]
pub mod opencl;

/// Metal バックエンド（`metal` feature required、macOSのみ）
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

// Re-export core traits
pub use traits::{
    Buffer, Compiler, Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType,
    DynBuffer, Kernel, KernelConfig, OpKind, SimdCapability,
};

// Re-export pipeline types (Pipeline, CompiledKernel, etc.)
pub use pipeline::{
    BoundExecutionQuery, CompiledKernel, DispatchSizeConfig, DispatchSizeExpr,
    KernelExecutionError, KernelSourceRenderer, OptimizationHistories, Pipeline, PipelineConfig,
};

// Re-export sequence types
pub use sequence::{ExecutionQuery, ProgramExecutionError};

// Re-export Renderer trait from renderer module for backwards compatibility
pub use crate::renderer::Renderer;

// Re-export global device management
pub use global::{
    DeviceKind, clear_default_device, get_default_device, get_default_device_kind,
    has_default_device, set_default_device, with_device,
};

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
    pub shape: Vec<crate::tensor::shape::Expr>,
}

impl BufferSignature {
    /// 新しいBufferSignatureを作成
    pub fn new(name: String, shape: Vec<crate::tensor::shape::Expr>) -> Self {
        Self { name, shape }
    }
}
