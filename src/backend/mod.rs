//! Backend module
//!
//! This module provides GPU kernel rendering and execution capabilities.
//!
//! ## Architecture
//!
//! The backend is organized into:
//! - **Renderers**: Convert AST to kernel source code (OpenCL, Metal)
//! - **Native backends**: Execute kernels using GPU APIs (via `ocl` and `metal` crates)
//! - **Pipeline**: End-to-end compilation from Graph to executable kernel
//!
//! ## Usage
//!
//! ```ignore
//! use harp::backend::native::{NativePipeline, NativeContext, NativeCompiler};
//! use harp::backend::native::opencl::{OpenCLNativeContext, OpenCLNativeCompiler};
//! use harp::backend::opencl::OpenCLRenderer;
//!
//! let context = OpenCLNativeContext::new()?;
//! let renderer = OpenCLRenderer::new();
//! let compiler = OpenCLNativeCompiler::new();
//! let mut pipeline = NativePipeline::new(renderer, compiler, context);
//!
//! let kernel = pipeline.compile_graph(graph)?;
//! ```

use std::collections::HashMap;

pub mod c_like;
pub mod metal;
pub mod native;
pub mod opencl;
pub mod pipeline;

// Re-export commonly used types
pub use metal::{MetalCode, MetalRenderer};
pub use opencl::{OpenCLCode, OpenCLRenderer};

// Re-export native backend types
pub use native::{
    CompiledNativeKernel, KernelConfig, KernelSourceRenderer, NativeBuffer, NativeCompiler,
    NativeContext, NativeKernel, NativeOptimizationHistories, NativePipeline, NativePipelineConfig,
};

// Re-export pipeline utilities
pub use pipeline::{MultiPhaseConfig, create_multi_phase_optimizer};

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
