//! Metal native compiler

use super::device::{MetalDevice, MetalError};
use super::kernel::MetalKernel;
use harp_core::backend::traits::{Compiler, KernelConfig};
use metal::CompileOptions;
use std::sync::Arc;

/// Metal native compiler
///
/// Compiles Metal Shading Language source code into executable compute pipelines.
pub struct MetalCompiler;

impl Compiler for MetalCompiler {
    type Dev = MetalDevice;
    type Kernel = MetalKernel;
    type Error = MetalError;

    fn new() -> Self {
        Self
    }

    fn compile(
        &self,
        device: &Self::Dev,
        source: &str,
        config: KernelConfig,
    ) -> Result<Self::Kernel, Self::Error> {
        // Compile the shader source
        let options = CompileOptions::new();
        let library = device
            .device()
            .new_library_with_source(source, &options)
            .map_err(|e| MetalError::from(format!("Failed to compile Metal shader: {}", e)))?;

        // Get the kernel function
        let function = library
            .get_function(&config.entry_point, None)
            .map_err(|e| {
                MetalError::from(format!(
                    "Failed to get function '{}': {}",
                    config.entry_point, e
                ))
            })?;

        // Create compute pipeline state
        let pipeline = device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::from(format!("Failed to create compute pipeline: {}", e)))?;

        Ok(MetalKernel::new(
            pipeline,
            Arc::clone(&device.command_queue),
            config,
        ))
    }
}

impl MetalCompiler {
    /// Compile with custom compile options
    pub fn compile_with_options(
        &self,
        device: &MetalDevice,
        source: &str,
        config: KernelConfig,
        options: &CompileOptions,
    ) -> Result<MetalKernel, MetalError> {
        // Compile the shader source with custom options
        let library = device
            .device()
            .new_library_with_source(source, options)
            .map_err(|e| MetalError::from(format!("Failed to compile Metal shader: {}", e)))?;

        // Get the kernel function
        let function = library
            .get_function(&config.entry_point, None)
            .map_err(|e| {
                MetalError::from(format!(
                    "Failed to get function '{}': {}",
                    config.entry_point, e
                ))
            })?;

        // Create compute pipeline state
        let pipeline = device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::from(format!("Failed to create compute pipeline: {}", e)))?;

        Ok(MetalKernel::new(
            pipeline,
            Arc::clone(&device.command_queue),
            config,
        ))
    }

    /// Compile with fast math optimizations enabled
    ///
    /// This enables Metal's fast math mode which can significantly improve
    /// performance but may affect numerical precision.
    pub fn compile_with_fast_math(
        &self,
        device: &MetalDevice,
        source: &str,
        config: KernelConfig,
    ) -> Result<MetalKernel, MetalError> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);
        self.compile_with_options(device, source, config, &options)
    }

    /// Compile with optional fast math based on configuration
    pub fn compile_configurable(
        &self,
        device: &MetalDevice,
        source: &str,
        config: KernelConfig,
        fast_math: bool,
    ) -> Result<MetalKernel, MetalError> {
        if fast_math {
            self.compile_with_fast_math(device, source, config)
        } else {
            self.compile(device, source, config)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalBuffer;
    use harp_core::ast::DType;
    use harp_core::backend::traits::{Buffer, Device};

    #[test]
    fn test_metal_simple_kernel() {
        if !MetalDevice::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalDevice::new().unwrap();
        let compiler = MetalCompiler::new();

        // Simple kernel that adds two arrays
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void add(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                uint i [[thread_position_in_grid]]
            ) {
                c[i] = a[i] + b[i];
            }
        "#;

        let config = KernelConfig::new("add").with_global_work_size([4, 1, 1]);

        let kernel = compiler.compile(&context, source, config);
        assert!(
            kernel.is_ok(),
            "Failed to compile kernel: {:?}",
            kernel.err()
        );

        // Create input/output buffers
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let a_buffer = MetalBuffer::from_vec(&context, vec![4], DType::F32, &a_data).unwrap();
        let b_buffer = MetalBuffer::from_vec(&context, vec![4], DType::F32, &b_data).unwrap();
        let c_buffer = MetalBuffer::allocate(&context, vec![4], DType::F32).unwrap();

        // Execute kernel
        let kernel = kernel.unwrap();
        let result = kernel.execute_with_buffers(&[&a_buffer, &b_buffer, &c_buffer]);
        assert!(
            result.is_ok(),
            "Failed to execute kernel: {:?}",
            result.err()
        );

        // Read result
        let c_result: Vec<f32> = c_buffer.read_vec().unwrap();
        assert_eq!(c_result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_metal_matrix_multiply() {
        if !MetalDevice::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalDevice::new().unwrap();
        let compiler = MetalCompiler::new();

        // Simple 2x2 matrix multiply kernel
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void matmul(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                // Simple 2x2 matrix multiply
                uint row = gid.y;
                uint col = gid.x;
                uint N = 2;

                float sum = 0.0;
                for (uint k = 0; k < N; k++) {
                    sum += a[row * N + k] * b[k * N + col];
                }
                c[row * N + col] = sum;
            }
        "#;

        let config = KernelConfig::new("matmul").with_global_work_size([2, 2, 1]);

        let kernel = compiler.compile(&context, source, config);
        assert!(
            kernel.is_ok(),
            "Failed to compile kernel: {:?}",
            kernel.err()
        );

        // 2x2 matrices:
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A * B = [[19, 22], [43, 50]]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let a_buffer = MetalBuffer::from_vec(&context, vec![2, 2], DType::F32, &a_data).unwrap();
        let b_buffer = MetalBuffer::from_vec(&context, vec![2, 2], DType::F32, &b_data).unwrap();
        let c_buffer = MetalBuffer::allocate(&context, vec![2, 2], DType::F32).unwrap();

        // Execute kernel
        let kernel = kernel.unwrap();
        let result = kernel.execute_with_buffers(&[&a_buffer, &b_buffer, &c_buffer]);
        assert!(
            result.is_ok(),
            "Failed to execute kernel: {:?}",
            result.err()
        );

        // Read result
        let c_result: Vec<f32> = c_buffer.read_vec().unwrap();
        assert_eq!(c_result, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
