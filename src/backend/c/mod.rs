//! C backend implementation
//!
//! This module provides a C language backend for code generation and execution.
//! The backend includes:
//!
//! - `CRenderer`: Renders AST to C code
//! - `CCompiler`: Compiles C code to shared libraries
//! - `CBuffer`: Memory buffer implementation for C backend
//! - `CBackend`: Combined backend using the above components
//!
//! # OpenMP Support
//!
//! The C backend supports optional OpenMP parallelization for kernel execution.
//! By default, OpenMP is enabled. You can disable it using the option types:
//!
//! ```no_run
//! use harp::backend::c::{CBackend, CRendererOption, CCompilerOption};
//! use harp::backend::generic::GenericBackendOption;
//! use harp::backend::Backend;
//!
//! let mut backend = CBackend::new();
//!
//! // Disable OpenMP for both rendering and compilation
//! backend.with_option(GenericBackendOption::Both {
//!     renderer: CRendererOption { use_openmp: false },
//!     compiler: CCompilerOption { use_openmp: false },
//! });
//! ```
//!
//! When OpenMP is disabled:
//! - Kernel calls are rendered as simple for loops instead of `#pragma omp parallel for`
//! - No OpenMP headers are included in generated code
//! - No OpenMP libraries (`-fopenmp`, `-lomp`, `-lgomp`) are linked during compilation

pub mod buffer;
pub mod compiler;
pub mod kernel;
pub mod renderer;

pub use buffer::CBuffer;
pub use compiler::{CCompiler, CCompilerOption};
pub use kernel::CKernel;
pub use renderer::{CRenderer, CRendererOption};

use crate::backend::generic::GenericBackend;

pub type CBackend = GenericBackend<CRenderer, CCompiler, CBuffer>;
