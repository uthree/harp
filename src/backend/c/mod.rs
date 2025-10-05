pub mod buffer;
pub mod compiler;
pub mod kernel;
pub mod renderer;

pub use buffer::CBuffer;
pub use compiler::CCompiler;
pub use kernel::CKernel;
pub use renderer::CRenderer;

use crate::backend::generic::GenericBackend;

pub type CBackend = GenericBackend<CRenderer, CCompiler, CBuffer>;
