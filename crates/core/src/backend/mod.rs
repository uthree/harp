//! Backend module
//!
//! This module provides GPU kernel rendering and execution capabilities.
//!
//! ## Architecture
//!
//! The backend is organized into:
//! - **Renderers**: Convert AST to kernel source code (provided by backend crates)
//! - **Traits**: Common interfaces for GPU execution (Device, Buffer, etc.)
//! - **Execution**: Pipeline for end-to-end compilation from Graph to executable kernel
//!
//! ## Usage
//!
//! ```ignore
//! use harp_core::backend::{Pipeline, Device, Compiler};
//! // Backend implementations are provided by separate crates:
//! // - harp-backend-opencl
//! // - harp-backend-metal
//! ```

use std::collections::HashMap;

pub mod c_like;
pub mod pipeline;
pub mod sequence;
pub mod traits;

// Re-export core traits
pub use traits::{
    Buffer, Compiler, Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, Kernel,
    KernelConfig,
};

// Re-export pipeline types (Pipeline, CompiledKernel, etc.)
pub use pipeline::{
    BoundExecutionQuery, CompiledKernel, DispatchSizeConfig, DispatchSizeExpr,
    KernelExecutionError, KernelSourceRenderer, OptimizationHistories, Pipeline, PipelineConfig,
};

// Re-export sequence types
pub use sequence::{
    CompiledProgram, ExecutionQuery, IntermediateBufferSpec, KernelCallInfo, ProgramExecutionError,
};

// Re-export graph optimizer factory functions from opt::graph
pub use crate::opt::graph::{
    IdentityOptimizer, MultiPhaseConfig, SubgraphMode, create_greedy_optimizer,
    create_multi_phase_optimizer, create_multi_phase_optimizer_with_selector,
    optimize_graph_greedy, optimize_graph_multi_phase,
};

/// Renderer trait for converting AST to source code
pub trait Renderer {
    type CodeRepr: Into<String>;
    type Option;
    fn render(&self, program: &crate::ast::AstNode) -> Self::CodeRepr;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, _option: Self::Option) {}
}

/// カーネルのシグネチャ（入出力バッファの形状情報）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSignature {
    pub inputs: Vec<BufferSignature>,
    pub outputs: Vec<BufferSignature>,
    pub shape_vars: HashMap<String, isize>,
}

impl KernelSignature {
    /// 新しいKernelSignatureを作成
    pub fn new(
        inputs: Vec<BufferSignature>,
        outputs: Vec<BufferSignature>,
        shape_vars: HashMap<String, isize>,
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
