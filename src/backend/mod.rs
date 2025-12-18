//! Backend module
//!
//! This module provides GPU kernel rendering and execution capabilities.
//!
//! ## Architecture
//!
//! The backend is organized into:
//! - **Renderers**: Convert AST to kernel source code (OpenCL, Metal)
//! - **Traits**: Common interfaces for GPU execution (Device, Buffer, etc.)
//! - **Execution**: Pipeline for end-to-end compilation from Graph to executable kernel
//!
//! ## Usage
//!
//! ```ignore
//! use harp::backend::{Pipeline, Device, Compiler};
//! use harp::backend::opencl::{OpenCLDevice, OpenCLCompiler, OpenCLRenderer};
//!
//! let device = OpenCLDevice::new()?;
//! let renderer = OpenCLRenderer::new();
//! let compiler = OpenCLCompiler::new();
//! let mut pipeline = Pipeline::new(renderer, compiler, device);
//!
//! let kernel = pipeline.compile_graph(graph)?;
//! ```

use std::collections::HashMap;

pub mod c_like;
pub mod metal;
pub mod opencl;
pub mod pipeline;

// Core traits and execution modules
pub mod execution;
pub mod sequence;
pub mod traits;

// Re-export commonly used types
pub use metal::{MetalCode, MetalRenderer};
pub use opencl::{OpenCLCode, OpenCLRenderer};

// Re-export core traits
pub use traits::{Buffer, Compiler, Device, Kernel, KernelConfig};

// Re-export execution types
pub use execution::{
    CompiledKernel, KernelSourceRenderer, OptimizationHistories, Pipeline, PipelineConfig,
};

// Re-export sequence types
pub use sequence::{
    CompiledProgram, IntermediateBufferSpec, KernelCallInfo, ProgramExecutionError,
};

// Re-export pipeline utilities
pub use pipeline::{
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
