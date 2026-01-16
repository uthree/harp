//! CUDA kernel implementation
//!
//! Loads and executes CUDA kernels from PTX.

#[cfg(feature = "cuda-runtime")]
use crate::buffer::CudaBuffer;
use crate::device::CudaDevice;
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::{Buffer, Kernel};
use eclat::backend::KernelConfig;
use std::any::Any;
use std::error::Error;
use tempfile::TempDir;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig};
#[cfg(feature = "cuda-runtime")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

/// CUDA kernel for GPU execution
#[cfg(feature = "cuda-runtime")]
pub struct CudaKernel {
    /// CUDA context for stream access
    context: Arc<CudaContext>,
    /// CUDA module containing the kernel
    module: Arc<CudaModule>,
    /// The kernel function
    function: CudaFunction,
    /// Kernel configuration
    config: KernelConfig,
    /// Keep temp directory alive (contains PTX)
    #[allow(dead_code)]
    temp_dir: Option<TempDir>,
}

/// Stub CUDA kernel (when cuda-runtime feature is disabled)
#[cfg(not(feature = "cuda-runtime"))]
pub struct CudaKernel {
    /// Kernel configuration
    config: KernelConfig,
    /// Keep temp directory alive
    #[allow(dead_code)]
    temp_dir: Option<TempDir>,
}

#[cfg(feature = "cuda-runtime")]
impl CudaKernel {
    /// Create a new CUDA kernel from PTX
    pub fn from_ptx(
        device: &CudaDevice,
        temp_dir: TempDir,
        ptx_content: Vec<u8>,
        config: KernelConfig,
    ) -> Result<Self, CudaKernelError> {
        // Load PTX into CUDA module
        let ptx = Ptx::from_src(ptx_content);
        let context = Arc::clone(device.context());
        let module = context
            .load_module(ptx)
            .map_err(|e| CudaKernelError::ModuleLoadError(format!("Failed to load PTX: {}", e)))?;

        // Get the kernel function
        let function = module
            .load_function(&config.entry_point)
            .map_err(|e| CudaKernelError::FunctionNotFound(format!(
                "Function '{}' not found: {}",
                config.entry_point, e
            )))?;

        Ok(Self {
            context,
            module: Arc::new(module),
            function,
            config,
            temp_dir: Some(temp_dir),
        })
    }
}

#[cfg(not(feature = "cuda-runtime"))]
impl CudaKernel {
    /// Create a new CUDA kernel from PTX (stub - always fails)
    pub fn from_ptx(
        _device: &CudaDevice,
        _temp_dir: TempDir,
        _ptx_content: Vec<u8>,
        _config: KernelConfig,
    ) -> Result<Self, CudaKernelError> {
        Err(CudaKernelError::NotImplemented(
            "CUDA runtime not available. Rebuild with --features cuda-runtime".to_string()
        ))
    }
}

#[cfg(feature = "cuda-runtime")]
impl Clone for CudaKernel {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            module: Arc::clone(&self.module),
            function: self.function.clone(),
            config: self.config.clone(),
            temp_dir: None,
        }
    }
}

#[cfg(not(feature = "cuda-runtime"))]
impl Clone for CudaKernel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            temp_dir: None,
        }
    }
}

// Safety: CudaKernel can be sent between threads as the module is reference counted
unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

impl Kernel for CudaKernel {
    fn clone_kernel(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn execute(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let grid_size = self.config.global_work_size;
        let local_size = self.config.local_work_size.unwrap_or([256, 1, 1]);
        self.execute_with_sizes(inputs, outputs, grid_size, local_size)
    }

    #[cfg(feature = "cuda-runtime")]
    fn execute_with_sizes(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
        global_size: [usize; 3],
        local_size: [usize; 3],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        use std::ffi::c_void;

        // CUDA launch configuration
        // Ensure at least 1 block in each dimension
        let grid_dim = (
            (global_size[0] / local_size[0].max(1)).max(1) as u32,
            (global_size[1] / local_size[1].max(1)).max(1) as u32,
            (global_size[2] / local_size[2].max(1)).max(1) as u32,
        );
        let block_dim = (
            local_size[0] as u32,
            local_size[1] as u32,
            local_size[2] as u32,
        );

        let launch_config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        // Collect buffer device pointers
        // Each kernel argument is a pointer to the device memory
        let mut kernel_args: Vec<u64> = Vec::with_capacity(inputs.len() + outputs.len());

        for input in inputs {
            let cuda_buf = input
                .as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| CudaKernelError::InvalidBuffer("Input buffer must be CudaBuffer".to_string()))?;
            kernel_args.push(cuda_buf.as_device_ptr_u64());
        }

        for output in outputs.iter_mut() {
            let cuda_buf = output
                .as_any_mut()
                .downcast_mut::<CudaBuffer>()
                .ok_or_else(|| CudaKernelError::InvalidBuffer("Output buffer must be CudaBuffer".to_string()))?;
            kernel_args.push(cuda_buf.as_device_ptr_u64());
        }

        log::debug!(
            "Launching kernel '{}' with grid={:?}, block={:?}, {} args",
            self.config.entry_point,
            grid_dim,
            block_dim,
            kernel_args.len()
        );

        // Create array of pointers to kernel arguments
        // CUDA expects void** where each element points to the actual argument value
        let mut arg_ptrs: Vec<*mut c_void> = kernel_args
            .iter()
            .map(|arg| arg as *const u64 as *mut c_void)
            .collect();

        // Launch kernel
        unsafe {
            self.function
                .launch_raw(launch_config, &mut arg_ptrs)
                .map_err(|e| CudaKernelError::LaunchError(format!("Kernel launch failed: {}", e)))?;
        }

        // Synchronize to ensure kernel completion
        let stream = self.context.default_stream();
        stream
            .synchronize()
            .map_err(|e| CudaKernelError::LaunchError(format!("Stream synchronization failed: {}", e)))?;

        log::debug!("Kernel '{}' completed successfully", self.config.entry_point);

        Ok(())
    }

    #[cfg(not(feature = "cuda-runtime"))]
    fn execute_with_sizes(
        &self,
        _inputs: &[&dyn Buffer],
        _outputs: &mut [&mut dyn Buffer],
        _global_size: [usize; 3],
        _local_size: [usize; 3],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        Err(Box::new(CudaKernelError::NotImplemented(
            "CUDA runtime not available".to_string()
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl std::fmt::Debug for CudaKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKernel")
            .field("entry_point", &self.config.entry_point)
            .finish()
    }
}

/// Error types for CUDA kernel operations
#[derive(Debug)]
pub enum CudaKernelError {
    /// Module loading failed
    ModuleLoadError(String),
    /// Function not found in module
    FunctionNotFound(String),
    /// Invalid buffer type
    InvalidBuffer(String),
    /// Kernel launch failed
    LaunchError(String),
    /// Not yet implemented
    NotImplemented(String),
}

impl std::fmt::Display for CudaKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaKernelError::ModuleLoadError(msg) => write!(f, "Module load error: {}", msg),
            CudaKernelError::FunctionNotFound(msg) => write!(f, "Function not found: {}", msg),
            CudaKernelError::InvalidBuffer(msg) => write!(f, "Invalid buffer: {}", msg),
            CudaKernelError::LaunchError(msg) => write!(f, "Kernel launch error: {}", msg),
            CudaKernelError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
        }
    }
}

impl std::error::Error for CudaKernelError {}
