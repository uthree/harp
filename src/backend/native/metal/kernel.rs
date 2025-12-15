//! Metal native kernel

use super::buffer::MetalNativeBuffer;
use super::context::MetalNativeError;
use crate::backend::native::{KernelConfig, NativeKernel};
use metal::{CommandQueue, ComputePipelineState, MTLSize};
use std::sync::Arc;

/// Metal native kernel
///
/// Wraps a compiled Metal compute pipeline that can be executed on the GPU.
#[derive(Clone)]
pub struct MetalNativeKernel {
    pipeline: Arc<ComputePipelineState>,
    command_queue: Arc<CommandQueue>,
    config: KernelConfig,
}

impl NativeKernel for MetalNativeKernel {
    type Buffer = MetalNativeBuffer;
    type Error = MetalNativeError;

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn execute(
        &self,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
    ) -> Result<(), Self::Error> {
        // Combine all buffers for execution
        let all_buffers: Vec<&MetalNativeBuffer> = inputs
            .iter()
            .copied()
            .chain(outputs.iter().map(|b| &**b))
            .collect();

        self.execute_with_buffers(&all_buffers)
    }
}

impl MetalNativeKernel {
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
    pub fn execute_with_buffers(
        &self,
        buffers: &[&MetalNativeBuffer],
    ) -> Result<(), MetalNativeError> {
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
        let gws = self.config.global_work_size;

        let threadgroup_size = if let Some(lws) = self.config.local_work_size {
            MTLSize::new(lws[0] as u64, lws[1] as u64, lws[2] as u64)
        } else {
            // Use default threadgroup size based on pipeline
            let max_threads = self.pipeline.max_total_threads_per_threadgroup();
            // Use a reasonable default that works for most cases
            let size = (max_threads as f64).sqrt() as u64;
            MTLSize::new(size.max(1), size.max(1), 1)
        };

        // Grid size is the total number of threads we want to dispatch
        let grid_size = MTLSize::new(gws[0] as u64, gws[1] as u64, gws[2] as u64);

        // Dispatch the compute kernel
        // dispatch_threads handles non-uniform threadgroups automatically
        encoder.dispatch_threads(grid_size, threadgroup_size);

        // End encoding and commit
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check for errors
        if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
            return Err(MetalNativeError::from(
                "Metal command buffer execution failed",
            ));
        }

        Ok(())
    }
}

// Safety: Metal kernel is thread-safe
unsafe impl Send for MetalNativeKernel {}
unsafe impl Sync for MetalNativeKernel {}
