//! Metal native kernel

use super::buffer::MetalBuffer;
use super::device::MetalError;
use crate::backend::traits::{Kernel, KernelConfig};
use metal::{CommandQueue, ComputePipelineState, MTLSize};
use std::sync::Arc;

/// Metal native kernel
///
/// Wraps a compiled Metal compute pipeline that can be executed on the GPU.
#[derive(Clone)]
pub struct MetalKernel {
    pipeline: Arc<ComputePipelineState>,
    command_queue: Arc<CommandQueue>,
    config: KernelConfig,
}

impl Kernel for MetalKernel {
    type Buffer = MetalBuffer;
    type Error = MetalError;

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn execute(
        &self,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
    ) -> Result<(), Self::Error> {
        // Combine all buffers for execution
        let all_buffers: Vec<&MetalBuffer> = inputs
            .iter()
            .copied()
            .chain(outputs.iter().map(|b| &**b))
            .collect();

        self.execute_with_buffers(&all_buffers)
    }

    fn execute_with_sizes(
        &self,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
        grid_size: [usize; 3],
        local_size: [usize; 3],
    ) -> Result<(), Self::Error> {
        // Combine all buffers for execution
        let all_buffers: Vec<&MetalBuffer> = inputs
            .iter()
            .copied()
            .chain(outputs.iter().map(|b| &**b))
            .collect();

        self.execute_with_buffers_and_sizes(&all_buffers, grid_size, local_size)
    }
}

impl MetalKernel {
    /// Create a new Metal native kernel
    pub(crate) fn new(
        pipeline: ComputePipelineState,
        command_queue: Arc<CommandQueue>,
        config: KernelConfig,
    ) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
            command_queue,
            config,
        }
    }

    /// Execute with explicit buffer references
    pub fn execute_with_buffers(&self, buffers: &[&MetalBuffer]) -> Result<(), MetalError> {
        let gws = self.config.global_work_size;
        let lws = self.config.local_work_size;
        self.execute_with_buffers_internal(buffers, gws, lws)
    }

    /// Execute with explicit buffer references and dispatch sizes
    pub fn execute_with_buffers_and_sizes(
        &self,
        buffers: &[&MetalBuffer],
        grid_size: [usize; 3],
        local_size: [usize; 3],
    ) -> Result<(), MetalError> {
        self.execute_with_buffers_internal(buffers, grid_size, Some(local_size))
    }

    /// Internal execution implementation
    fn execute_with_buffers_internal(
        &self,
        buffers: &[&MetalBuffer],
        grid_size: [usize; 3],
        local_size: Option<[usize; 3]>,
    ) -> Result<(), MetalError> {
        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // Create compute command encoder
        let encoder = command_buffer.new_compute_command_encoder();

        // Set the pipeline state
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffer arguments
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer.mtl_buffer()), 0);
        }

        // Calculate threadgroup size
        let threadgroup_size = if let Some(lws) = local_size {
            MTLSize::new(lws[0] as u64, lws[1] as u64, lws[2] as u64)
        } else {
            // Use default threadgroup size based on pipeline
            let max_threads = self.pipeline.max_total_threads_per_threadgroup();
            // Use a reasonable default that works for most cases
            let size = (max_threads as f64).sqrt() as u64;
            MTLSize::new(size.max(1), size.max(1), 1)
        };

        // Grid size is the total number of threads we want to dispatch
        let mtl_grid_size = MTLSize::new(
            grid_size[0] as u64,
            grid_size[1] as u64,
            grid_size[2] as u64,
        );

        // Dispatch the compute kernel
        // dispatch_threads handles non-uniform threadgroups automatically
        encoder.dispatch_threads(mtl_grid_size, threadgroup_size);

        // End encoding and commit
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check for errors
        if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
            return Err(MetalError::from("Metal command buffer execution failed"));
        }

        Ok(())
    }
}

// Safety: Metal kernel is thread-safe
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}
