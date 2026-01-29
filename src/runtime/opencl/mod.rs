//! OpenCL backend implementation.

pub mod buffer;
pub mod codegen;
pub mod device;
pub mod interpreter;
pub mod kernel;
pub mod ops;

pub use buffer::OpenCLBuffer;
pub use codegen::FusedKernelCodeGen;
pub use device::OpenCLDevice;
