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

use std::collections::HashMap;

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
    Buffer, Compiler, Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, Kernel,
    KernelConfig, OpKind, SimdCapability,
};

// Re-export pipeline types (Pipeline, CompiledKernel, etc.)
pub use pipeline::{
    BoundExecutionQuery, CompiledKernel, DispatchSizeConfig, DispatchSizeExpr,
    KernelExecutionError, KernelSourceRenderer, OptimizationHistories, Pipeline, PipelineConfig,
};

// Re-export sequence types
// Note: CompiledProgram, IntermediateBufferSpec, KernelCallInfo are deprecated
// but still exported for backwards compatibility
#[allow(deprecated)]
pub use sequence::{
    CompiledProgram, ExecutionQuery, IntermediateBufferSpec, KernelCallInfo, ProgramExecutionError,
};

// Re-export graph optimizer factory functions from opt::graph
pub use crate::opt::graph::{
    IdentityOptimizer, MultiPhaseConfig, SubgraphMode, create_greedy_optimizer,
    create_multi_phase_optimizer, create_multi_phase_optimizer_with_selector,
    optimize_graph_greedy, optimize_graph_multi_phase,
};

// Re-export Renderer trait from renderer module for backwards compatibility
pub use crate::renderer::Renderer;

/// カーネルのシグネチャ（入出力バッファの形状情報）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSignature {
    pub inputs: Vec<BufferSignature>,
    pub outputs: Vec<BufferSignature>,
    pub shape_vars: HashMap<String, i64>,
}

impl KernelSignature {
    /// 新しいKernelSignatureを作成
    pub fn new(
        inputs: Vec<BufferSignature>,
        outputs: Vec<BufferSignature>,
        shape_vars: HashMap<String, i64>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            shape_vars,
        }
    }

    /// 空のKernelSignatureを作成
    pub fn empty() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            shape_vars: HashMap::new(),
        }
    }
}

/// バッファのシグネチャ（名前と形状）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSignature {
    pub name: String,
    pub shape: Vec<crate::graph::shape::Expr>,
}

impl BufferSignature {
    /// 新しいBufferSignatureを作成
    pub fn new(name: String, shape: Vec<crate::graph::shape::Expr>) -> Self {
        Self { name, shape }
    }
}
