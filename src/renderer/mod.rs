//! Renderer module
//!
//! This module provides source code rendering for various backends.
//! Renderers are always available without feature flags, allowing
//! source code generation without GPU execution capabilities.
//!
//! ## Available Renderers
//!
//! - `GenericRenderer`: C-like generic renderer (always available)
//! - `OpenCLRenderer`: OpenCL C renderer (always available)
//! - `MetalRenderer`: Metal Shading Language renderer (always available, macOS targets)
//!
//! ## Usage
//!
//! ```ignore
//! use harp::renderer::{OpenCLRenderer, Renderer};
//! use harp::ast::AstNode;
//!
//! let renderer = OpenCLRenderer::new();
//! let code = renderer.render(&program);
//! println!("{}", code);
//! ```

pub mod c_like;
pub mod metal;
pub mod opencl;

// Re-export common types
pub use c_like::{CLikeRenderer, GenericRenderer, OptimizationLevel};
pub use metal::{MetalCode, MetalRenderer};
pub use opencl::{OpenCLCode, OpenCLRenderer};

/// Renderer trait for converting AST to source code
///
/// This trait is implemented by all renderers and provides a common
/// interface for source code generation.
pub trait Renderer {
    /// The type representing rendered code
    type CodeRepr: Into<String> + AsRef<str>;

    /// Renderer-specific options
    type Option;

    /// Render an AST program to source code
    fn render(&self, program: &crate::ast::AstNode) -> Self::CodeRepr;

    /// Check if this renderer is available on the current platform
    fn is_available(&self) -> bool;

    /// Configure renderer with an option
    fn with_option(&mut self, _option: Self::Option) {}
}
