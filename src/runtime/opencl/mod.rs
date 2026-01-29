//! OpenCL backend implementation.

pub mod buffer;
pub mod device;
pub mod interpreter;
pub mod kernel;
pub mod ops;

pub use buffer::OpenCLBuffer;
pub use device::OpenCLDevice;
