//! Runtime implementations for different backends.

pub mod cpu;

#[cfg(feature = "opencl")]
pub mod opencl;

pub use cpu::CpuDevice;

#[cfg(feature = "opencl")]
pub use opencl::OpenCLDevice;
