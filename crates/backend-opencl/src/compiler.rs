//! OpenCL native compiler

use super::device::{OpenCLDevice, OpenCLError};
use super::kernel::OpenCLKernel;
use harp_core::backend::traits::{Compiler, KernelConfig};
use ocl::Program;
use std::sync::Arc;

/// OpenCL native compiler
///
/// Compiles OpenCL C source code into executable kernels.
pub struct OpenCLCompiler;

impl Compiler for OpenCLCompiler {
    type Dev = OpenCLDevice;
    type Kernel = OpenCLKernel;
    type Error = OpenCLError;

    fn new() -> Self {
        Self
    }

    fn compile(
        &self,
        device: &Self::Dev,
        source: &str,
        config: KernelConfig,
    ) -> Result<Self::Kernel, Self::Error> {
        // Build program from source
        let program = Program::builder()
            .src(source)
            .devices(device.ocl_device())
            .build(device.ocl_context())?;

        Ok(OpenCLKernel::new(
            program,
            Arc::clone(&device.queue),
            config,
        ))
    }
}

impl OpenCLCompiler {
    /// Compile with build options
    pub fn compile_with_options(
        &self,
        context: &OpenCLDevice,
        source: &str,
        config: KernelConfig,
        build_options: &str,
    ) -> Result<OpenCLKernel, OpenCLError> {
        // Build program from source with options
        let program = Program::builder()
            .src(source)
            .devices(context.ocl_device())
            .cmplr_opt(build_options)
            .build(context.ocl_context())?;

        Ok(OpenCLKernel::new(
            program,
            Arc::clone(&context.queue),
            config,
        ))
    }
}
