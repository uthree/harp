// pub use backend::CBackend;
pub use buffer::CBuffer;
pub use compiler::CCompiler;
pub use kernel::CKernel;
pub use renderer::CRenderer;

use crate::backend::generic::GenericBackend;

// mod backend;
pub mod buffer;
mod compiler;
mod kernel;
pub mod renderer;

pub type CBackend = GenericBackend<CCompiler, CRenderer, CBuffer, String, ()>;
