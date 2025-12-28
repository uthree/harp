//! OpenCL native kernel

use super::buffer::OpenCLBuffer;
use super::device::OpenCLError;
use crate::backend::global::DeviceKind;
use crate::backend::traits::{Buffer, Kernel, KernelConfig};
use ocl::core::{ProgramInfo, ProgramInfoResult};
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
    fn clone_kernel(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::OpenCL
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Downcast dyn Buffer to OpenCLBuffer
        let opencl_inputs: Vec<&OpenCLBuffer> = inputs
            .iter()
            .map(|b| {
                b.as_any()
                    .downcast_ref::<OpenCLBuffer>()
                    .expect("Buffer type mismatch: expected OpenCLBuffer")
            })
            .collect();

        let opencl_outputs: Vec<&mut OpenCLBuffer> = outputs
            .iter_mut()
            .map(|b| {
                b.as_any_mut()
                    .downcast_mut::<OpenCLBuffer>()
                    .expect("Buffer type mismatch: expected OpenCLBuffer")
            })
            .collect();

        self.execute_kernel_internal(
            &opencl_inputs,
            opencl_outputs,
            self.config.global_work_size,
            self.config.local_work_size,
        )
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn execute_with_sizes(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
        grid_size: [usize; 3],
        local_size: [usize; 3],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Downcast dyn Buffer to OpenCLBuffer
        let opencl_inputs: Vec<&OpenCLBuffer> = inputs
            .iter()
            .map(|b| {
                b.as_any()
                    .downcast_ref::<OpenCLBuffer>()
                    .expect("Buffer type mismatch: expected OpenCLBuffer")
            })
            .collect();

        let opencl_outputs: Vec<&mut OpenCLBuffer> = outputs
            .iter_mut()
            .map(|b| {
                b.as_any_mut()
                    .downcast_mut::<OpenCLBuffer>()
                    .expect("Buffer type mismatch: expected OpenCLBuffer")
            })
            .collect();

        self.execute_kernel_internal(&opencl_inputs, opencl_outputs, grid_size, Some(local_size))
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
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
        outputs: Vec<&mut OpenCLBuffer>,
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

    /// コンパイル済みバイナリを取得
    ///
    /// ディスクキャッシュ用にプログラムのバイナリデータを抽出する。
    pub fn get_binary(&self) -> Result<Vec<u8>, OpenCLError> {
        let info = ocl::core::get_program_info(self.program.as_core(), ProgramInfo::Binaries)?;

        if let ProgramInfoResult::Binaries(binaries) = info {
            // 単一デバイス向けなので最初のバイナリを返す
            binaries
                .into_iter()
                .next()
                .ok_or_else(|| OpenCLError::from("No binary found in program"))
        } else {
            Err(OpenCLError::from("Unexpected program info result"))
        }
    }

    /// プログラムへの参照を取得（内部用）
    #[allow(dead_code)]
    pub(crate) fn program(&self) -> &Program {
        &self.program
    }
}

// Safety: OpenCL kernel is thread-safe
unsafe impl Send for OpenCLKernel {}
unsafe impl Sync for OpenCLKernel {}
