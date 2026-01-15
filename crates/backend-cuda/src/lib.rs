//! CUDA backend for Eclat with GPU runtime execution
//!
//! This crate provides CUDA code generation and GPU runtime execution for Eclat.
//! It uses nvcc to compile CUDA kernels and cudarc for runtime execution.
//!
//! # Features
//!
//! - `cuda-runtime`: Enable full CUDA runtime support (requires CUDA SDK)
//!
//! # Usage
//!
//! ```ignore
//! use eclat_backend_cuda::init;
//!
//! // Initialize the backend
//! init();
//!
//! // Now CUDA backend is available for runtime execution
//! use eclat::backend::set_device_str;
//! set_device_str("cuda:0").unwrap();
//! ```

mod buffer;
mod compiler;
mod device;
pub mod kernel;
pub mod renderer;

pub use buffer::CudaBuffer;
pub use compiler::CudaCompiler;
pub use device::CudaDevice;
pub use kernel::CudaKernel;
pub use renderer::{CudaCode, CudaRenderer};

use eclat::backend::device::{BackendRegistry, DeviceError};
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::Compiler;
use eclat::backend::Pipeline;
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// CUDA backend registry implementation
struct CudaBackendRegistry;

impl BackendRegistry for CudaBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn name(&self) -> &str {
        "CUDA (GPU)"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "cuda-runtime")]
        {
            use eclat::backend::traits::Device;
            CudaDevice::is_available()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            false
        }
    }

    fn create_device(&self, index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        let device = CudaDevice::new(index).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to create CUDA device: {}", e))
        })?;
        Ok(Arc::new(device))
    }

    fn list_devices(&self) -> Vec<String> {
        CudaDevice::list_devices()
    }

    fn supports_runtime(&self) -> bool {
        cfg!(feature = "cuda-runtime")
    }

    fn allocate_buffer(
        &self,
        device: &dyn Any,
        shape: Vec<usize>,
        dtype: eclat::ast::DType,
    ) -> Result<Box<dyn eclat::backend::Buffer>, DeviceError> {
        let cuda_device = device.downcast_ref::<CudaDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected CudaDevice".to_string(),
            )
        })?;

        let buffer = CudaBuffer::allocate(cuda_device, shape, dtype).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to allocate CUDA buffer: {}", e))
        })?;

        Ok(Box::new(buffer))
    }

    fn compile_ast(
        &self,
        device: &dyn Any,
        program: eclat::ast::AstNode,
        signature: eclat::backend::KernelSignature,
    ) -> Result<Box<dyn eclat::backend::Kernel>, DeviceError> {
        let cuda_device = device.downcast_ref::<CudaDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected CudaDevice".to_string(),
            )
        })?;

        let renderer = CudaRenderer::new();
        let compiler = CudaCompiler::new();
        let mut pipeline = Pipeline::new(renderer, compiler, cuda_device.clone());

        let cache_entry = pipeline.compile_ast(program, signature).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to compile AST: {}", e))
        })?;

        Ok(cache_entry.kernel)
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the CUDA backend
///
/// This function registers the CUDA backend with eclat-core, making it
/// available for device selection via `EclatDevice::new("cuda:0")`.
///
/// The CUDA backend requires:
/// - NVIDIA GPU with CUDA support
/// - CUDA Toolkit installed (nvcc compiler)
/// - CUDA drivers installed
///
/// Note: If built without the `cuda-runtime` feature, the backend will
/// register but report as unavailable.
///
/// This should be called once at program startup.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(CudaBackendRegistry));
    log::info!("CUDA backend initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        // Just ensure init doesn't panic
        init();
    }

    #[test]
    fn test_cuda_availability_check() {
        // This test just checks that availability check doesn't panic
        #[cfg(feature = "cuda-runtime")]
        {
            use eclat::backend::traits::Device;
            let available = CudaDevice::is_available();
            println!("CUDA backend available: {}", available);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            println!("CUDA runtime feature not enabled");
        }
    }

    #[test]
    fn test_nvcc_check() {
        let nvcc_available = CudaDevice::is_nvcc_available();
        println!("nvcc available: {}", nvcc_available);
    }
}
