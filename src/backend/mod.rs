//! Defines the core traits and data structures for the backend infrastructure.
//!
//! This module provides abstractions for:
//! - `Buffer`: A generic container for data on a device.
//! - `Kernel`: A compiled, executable function.
//! - `Compiler`: A component that compiles code into a `Kernel`.
//! - `Renderer`: A component that translates an AST into source code.
//! - `Device`: A component that manages memory allocation.

use crate::ast::{AstNode, DType};
use crate::tensor::shape::expr::Expr;
use std::ffi::c_void;

// --- Core Data Structures ---

/// Information about a single buffer used in a kernel.
#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub dtype: DType,
    pub shape: Vec<Expr>,
}

/// Detailed information about a compiled kernel's inputs and outputs.
#[derive(Debug, Clone, Default)]
pub struct KernelDetails {
    /// Information about each buffer (input and output).
    pub buffers: Vec<BufferInfo>,
    /// The names of variables used to define dynamic shapes.
    pub shape_variables: Vec<String>,
}

// --- Core Traits ---

/// A trait for a generic buffer that can be passed to a kernel.
pub trait Buffer {
    /// Returns a mutable pointer to the buffer's raw data.
    fn as_mut_ptr(&mut self) -> *mut c_void;
    /// Returns the data type of the elements in the buffer.
    fn dtype(&self) -> DType;
    /// Returns the shape of the buffer.
    fn shape(&self) -> &[Expr];
}

/// A trait for a compiled, executable kernel.
pub trait Kernel<Var: Buffer> {
    /// Executes the kernel with the given buffers and shape variables.
    fn call(&self, buffers: Vec<Var>, shape_variables: Vec<usize>) -> Vec<Var>;
}

/// A trait for a compiler that turns a code representation into a `Kernel`.
pub trait Compiler<Var: Buffer, CodeRepr = String, CompilerOption = ()> {
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: CompilerOption);
    fn compile(&mut self, code: CodeRepr) -> impl Kernel<Var>;
}

/// A trait for a component that translates an `AstNode` into a code representation.
pub trait Renderer<CodeRepr = String> {
    fn new() -> Self;
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}

/// A trait for a device that can allocate and manage memory.
pub trait Device<Var: Buffer> {
    fn allocate(&mut self, dtype: DType, size: usize) -> Var;
    fn free(&mut self, var: Var);
    fn is_available(&self) -> bool;
}

// --- Submodules ---
pub mod c;
