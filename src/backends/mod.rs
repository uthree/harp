//! Execution backends and compilation infrastructure.
//!
//! This module provides the traits and structs responsible for taking a computation
//! graph (`UOp`), compiling it, and executing it.
//!
//! The central trait is `Backend`, which orchestrates the process. Different
//! implementations of `Backend` can target different hardware or compilation toolchains.
//! Currently, a C-based backend using Clang (`ClangBackend`) is provided.

use crate::autotuner::BackendOptions;
use crate::uop::UOp;
use std::error::Error;
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;

// Re-export submodules
pub mod c;

// --- Top-level Backend Controller ---
mod clang;
pub use clang::ClangBackend;

// --- Backend Error ---
/// Errors that can occur during backend initialization or operation.
#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    /// Indicates that the required compiler for a backend could not be found.
    #[error("Compiler '{0}' not found. Please ensure it is installed and in your PATH.")]
    CompilerNotFound(String),
}

// --- Common Backend Traits ---

/// A trait representing a computation backend.
///
/// Backends are responsible for memory management (`alloc`, `free`) and for
/// orchestrating the compilation and execution of computation graphs.
pub trait Backend: Debug {
    /// Compiles and executes a `UOp` graph.
    ///
    /// # Arguments
    /// * `uop` - The root of the `UOp` abstract syntax tree to execute.
    /// * `bufs` - A slice of `Buffer` handles that are passed as arguments to the kernel.
    /// * `shape_args` - A slice of `usize` values representing shape-related arguments.
    /// * `options` - Backend-agnostic options for this execution.
    fn compile_and_exec(
        &self,
        uops: &[UOp],
        bufs: &[&Buffer],
        shape_args: &[usize],
        options: &BackendOptions,
    );

    /// Allocates a memory buffer on the device managed by this backend.
    ///
    /// # Arguments
    /// * `size` - The size of the buffer to allocate in bytes.
    /// * `backend` - A reference-counted handle to the backend itself, which will be
    ///   stored in the `Buffer` to manage its lifetime.
    fn alloc(&self, size: usize, backend: Rc<dyn Backend>) -> Buffer;

    /// Frees a memory buffer associated with the given ID.
    fn free(&self, id: usize);

    /// Gets a raw mutable pointer to the memory buffer for a given ID.
    ///
    /// # Safety
    /// This is highly unsafe as it bypasses Rust's memory safety guarantees.
    /// It should only be used by kernel execution code that knows how to handle raw pointers.
    fn get_buffer_ptr(&self, id: usize) -> *mut u8;
}

/// A trait for compiling source code into an executable `Kernel`.
pub trait Compiler {
    /// The type for compiler-specific options.
    type Options: Default + Clone + Debug;

    /// Checks if the compiler is available on the system.
    fn is_available(&self) -> bool;

    /// Compiles the given source code.
    ///
    /// # Arguments
    /// * `source_code` - The source code string to compile.
    /// * `options` - Compiler-specific options.
    fn compile(
        &self,
        source_code: &str,
        options: &Self::Options,
    ) -> Result<Rc<dyn Kernel>, Box<dyn Error>>;
}

/// A trait for rendering a `UOp` tree into source code.
pub trait Renderer {
    /// Renders a `UOp` tree into a source code string.
    fn render(&self, uops: &[UOp]) -> String;
}

/// A trait representing a compiled, executable kernel.
pub trait Kernel {
    /// Executes the kernel with the given buffers and shape arguments.
    fn exec(&self, bufs: &[&Buffer], shape_args: &[usize]);
    /// Returns metadata about the kernel, such as argument info.
    fn metadata(&self) -> &KernelMetadata;
}

// --- Common Kernel Structs ---

/// Information about a single kernel argument.
#[derive(Debug)]
pub struct ArgInfo {
    pub dtype: crate::dtype::DType,
    pub size: usize,
}

/// Metadata about a compiled kernel.
#[derive(Debug)]
pub struct KernelMetadata {
    pub args_info: Vec<ArgInfo>,
    pub global_work_size: usize,
    pub local_work_size: usize,
}

// --- Buffer ---

/// The internal, reference-counted implementation of a `Buffer`.
///
/// This struct holds the metadata for a device memory buffer. Its `Drop`
/// implementation ensures that the memory is freed on the backend when the
/// last reference is dropped.
pub struct Buffer_ {
    pub id: usize,
    pub size: usize,
    pub backend: Rc<dyn Backend>,
}

impl Drop for Buffer_ {
    fn drop(&mut self) {
        self.backend.free(self.id);
    }
}

/// A handle to a memory buffer on a compute device.
///
/// `Buffer` is a lightweight, reference-counted handle to a `Buffer_`.
/// It automatically manages the lifetime of the underlying device memory
/// via its `Drop` implementation.
#[derive(Clone)]
pub struct Buffer(pub Rc<Buffer_>);

impl Deref for Buffer {
    type Target = Buffer_;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
