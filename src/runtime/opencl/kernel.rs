//! OpenCL kernel compilation and caching.

use std::collections::HashMap;

use opencl3::context::Context;
use opencl3::device::Device as ClDevice;
use opencl3::kernel::Kernel;
use opencl3::program::Program;

use crate::device::{DeviceError, Result};
use crate::dtype::DType;

/// Cached compiled kernel.
pub struct CompiledKernel {
    #[allow(dead_code)]
    program: Program,
    kernel: Kernel,
}

impl CompiledKernel {
    /// Returns the OpenCL kernel.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }
}

/// Kernel cache for storing compiled kernels.
pub struct KernelCache {
    cache: HashMap<String, CompiledKernel>,
}

impl KernelCache {
    /// Creates a new kernel cache.
    pub fn new() -> Self {
        KernelCache {
            cache: HashMap::new(),
        }
    }

    /// Gets or compiles a kernel.
    pub fn get_or_compile(
        &mut self,
        context: &Context,
        device: &ClDevice,
        source: &str,
        kernel_name: &str,
    ) -> Result<&Kernel> {
        let cache_key = format!("{}:{}", kernel_name, source);

        if !self.cache.contains_key(&cache_key) {
            let compiled = compile_kernel(context, device, source, kernel_name)?;
            self.cache.insert(cache_key.clone(), compiled);
        }

        Ok(self.cache.get(&cache_key).unwrap().kernel())
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiles an OpenCL kernel from source.
fn compile_kernel(
    context: &Context,
    _device: &ClDevice,
    source: &str,
    kernel_name: &str,
) -> Result<CompiledKernel> {
    let program = Program::create_and_build_from_source(context, source, "").map_err(|e| {
        // Try to get build log for better error messages
        let error_msg = format!(
            "Failed to compile kernel '{}': {:?}\nSource:\n{}",
            kernel_name, e, source
        );
        DeviceError::CompilationFailed(error_msg)
    })?;

    let kernel = Kernel::create(&program, kernel_name).map_err(|e| {
        DeviceError::CompilationFailed(format!(
            "Failed to create kernel '{}': {:?}",
            kernel_name, e
        ))
    })?;

    Ok(CompiledKernel { program, kernel })
}

/// Returns the OpenCL type name for a DType.
pub fn dtype_to_cl_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "uchar",
        DType::Int32 => "int",
        DType::Int64 => "long",
        DType::Float32 => "float",
        DType::Float64 => "double",
    }
}

/// Returns the size in bytes for a DType.
pub fn dtype_size(dtype: DType) -> usize {
    dtype.size_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_to_cl_type() {
        assert_eq!(dtype_to_cl_type(DType::Float32), "float");
        assert_eq!(dtype_to_cl_type(DType::Float64), "double");
        assert_eq!(dtype_to_cl_type(DType::Int32), "int");
        assert_eq!(dtype_to_cl_type(DType::Int64), "long");
        assert_eq!(dtype_to_cl_type(DType::Bool), "uchar");
    }
}
