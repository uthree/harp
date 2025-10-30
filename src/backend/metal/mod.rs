pub mod renderer;

#[cfg(target_os = "macos")]
pub mod compiler;

#[cfg(target_os = "macos")]
pub use compiler::{MetalBuffer, MetalCompiler, MetalKernel};

pub use renderer::MetalRenderer;
