//! OpenCL native kernel

use super::buffer::OpenCLBuffer;
use super::device::OpenCLError;
use harp_core::backend::traits::{Kernel, KernelConfig};
use ocl::{Kernel as OclKernel, Program, Queue};
use std::sync::Arc;

/// OpenCL native kernel
///
/// Wraps a compiled OpenCL kernel that can be executed on the GPU.
#[derive(Clone)]
pub struct OpenCLKernel {
    program: Arc<Program>,
    queue: Arc<Queue>,
    config: KernelConfig,
}

impl Kernel for OpenCLKernel {
    type Buffer = OpenCLBuffer;
    type Error = OpenCLError;

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn execute(
        &self,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
    ) -> Result<(), Self::Error> {
        self.execute_kernel_internal(
            inputs,
            outputs,
            self.config.global_work_size,
            self.config.local_work_size,
        )
    }

    fn execute_with_sizes(
        &self,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
        grid_size: [usize; 3],
        local_size: [usize; 3],
    ) -> Result<(), Self::Error> {
        self.execute_kernel_internal(inputs, outputs, grid_size, Some(local_size))
    }
}

impl OpenCLKernel {
    /// Create a new OpenCL native kernel
    pub(crate) fn new(program: Program, queue: Arc<Queue>, config: KernelConfig) -> Self {
        Self {
            program: Arc::new(program),
            queue,
            config,
        }
    }

    /// Internal kernel execution with explicit sizes
    fn execute_kernel_internal(
        &self,
        inputs: &[&OpenCLBuffer],
        outputs: &mut [&mut OpenCLBuffer],
        grid_size: [usize; 3],
        local_size: Option<[usize; 3]>,
    ) -> Result<(), OpenCLError> {
        // Build kernel with arguments
        let mut kernel_builder = OclKernel::builder();
        kernel_builder
            .program(&self.program)
            .name(&self.config.entry_point)
            .queue((*self.queue).clone());

        // Set global work size
        kernel_builder.global_work_size([grid_size[0], grid_size[1], grid_size[2]]);

        // Set local work size if specified
        if let Some(lws) = local_size {
            kernel_builder.local_work_size([lws[0], lws[1], lws[2]]);
        }

        // Add input buffer arguments (positional)
        for input in inputs.iter() {
            kernel_builder.arg(input.ocl_buffer());
        }

        // Add output buffer arguments (positional)
        for output in outputs.iter() {
            kernel_builder.arg(output.ocl_buffer());
        }

        // Build and execute the kernel
        let kernel = kernel_builder.build()?;
        unsafe {
            kernel.enq()?;
        }
        self.queue.finish()?;

        Ok(())
    }

    /// Execute with explicit buffer references (alternative API)
    pub fn execute_with_buffers(&self, buffers: &[&OpenCLBuffer]) -> Result<(), OpenCLError> {
        // Build kernel with arguments
        let mut kernel_builder = OclKernel::builder();
        kernel_builder
            .program(&self.program)
            .name(&self.config.entry_point)
            .queue((*self.queue).clone());

        // Set global work size
        let gws = self.config.global_work_size;
        kernel_builder.global_work_size([gws[0], gws[1], gws[2]]);

        // Set local work size if specified
        if let Some(lws) = self.config.local_work_size {
            kernel_builder.local_work_size([lws[0], lws[1], lws[2]]);
        }

        // Add all buffer arguments (positional)
        for buffer in buffers.iter() {
            kernel_builder.arg(buffer.ocl_buffer());
        }

        // Build and execute the kernel
        let kernel = kernel_builder.build()?;
        unsafe {
            kernel.enq()?;
        }
        self.queue.finish()?;

        Ok(())
    }
}

// Safety: OpenCL kernel is thread-safe
unsafe impl Send for OpenCLKernel {}
unsafe impl Sync for OpenCLKernel {}
