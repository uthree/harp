//! Metal native kernel

use super::buffer::MetalBuffer;
use super::device::MetalError;
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::{Buffer, Kernel, KernelConfig};
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
    fn clone_kernel(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn execute(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Downcast dyn Buffer to MetalBuffer
        let metal_inputs: Vec<&MetalBuffer> = inputs
            .iter()
            .map(|b| {
                b.as_any()
                    .downcast_ref::<MetalBuffer>()
                    .expect("Buffer type mismatch: expected MetalBuffer")
            })
            .collect();

        let metal_outputs: Vec<&MetalBuffer> = outputs
            .iter()
            .map(|b| {
                b.as_any()
                    .downcast_ref::<MetalBuffer>()
                    .expect("Buffer type mismatch: expected MetalBuffer")
            })
            .collect();

        // Combine all buffers for execution
        let all_buffers: Vec<&MetalBuffer> =
            metal_inputs.into_iter().chain(metal_outputs).collect();

        self.execute_with_buffers(&all_buffers)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn execute_with_sizes(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
        grid_size: [usize; 3],
        local_size: [usize; 3],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Downcast dyn Buffer to MetalBuffer
        let metal_inputs: Vec<&MetalBuffer> = inputs
            .iter()
            .map(|b| {
                b.as_any()
                    .downcast_ref::<MetalBuffer>()
                    .expect("Buffer type mismatch: expected MetalBuffer")
            })
            .collect();

        let metal_outputs: Vec<&MetalBuffer> = outputs
            .iter()
            .map(|b| {
                b.as_any()
                    .downcast_ref::<MetalBuffer>()
                    .expect("Buffer type mismatch: expected MetalBuffer")
            })
            .collect();

        // Combine all buffers for execution
        let all_buffers: Vec<&MetalBuffer> =
            metal_inputs.into_iter().chain(metal_outputs).collect();

        self.execute_with_buffers_and_sizes(&all_buffers, grid_size, local_size)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

        // Set shape variable arguments (after buffer parameters)
        // Sort by name to ensure deterministic order (matches kernel parameter order)
        let mut shape_vars: Vec<_> = self.config.shape_vars.iter().collect();
        shape_vars.sort_by_key(|(name, _)| *name);

        let mut arg_index = buffers.len();
        for (_, value) in shape_vars {
            // Set i64 value as bytes (8 bytes for long)
            encoder.set_bytes(
                arg_index as u64,
                std::mem::size_of::<i64>() as u64,
                value as *const i64 as *const std::ffi::c_void,
            );
            arg_index += 1;
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
