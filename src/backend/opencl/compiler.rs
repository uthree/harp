//! OpenCL native compiler

use super::context::{OpenCLNativeContext, OpenCLNativeError};
use super::kernel::OpenCLNativeKernel;
use crate::backend::traits::{KernelConfig, NativeCompiler};
use ocl::Program;
use std::sync::Arc;

/// OpenCL native compiler
///
/// Compiles OpenCL C source code into executable kernels.
pub struct OpenCLNativeCompiler;

impl NativeCompiler for OpenCLNativeCompiler {
    type Context = OpenCLNativeContext;
    type Kernel = OpenCLNativeKernel;
    type Error = OpenCLNativeError;

    fn new() -> Self {
        Self
    }

    fn compile(
        &self,
        context: &Self::Context,
        source: &str,
        config: KernelConfig,
    ) -> Result<Self::Kernel, Self::Error> {
        // Build program from source
        let program = Program::builder()
            .src(source)
            .devices(context.ocl_device())
            .build(context.ocl_context())?;

        Ok(OpenCLNativeKernel::new(
            program,
            Arc::clone(&context.queue),
            config,
        ))
    }
}

impl OpenCLNativeCompiler {
    /// Compile with build options
    pub fn compile_with_options(
        &self,
        context: &OpenCLNativeContext,
        source: &str,
        config: KernelConfig,
        build_options: &str,
    ) -> Result<OpenCLNativeKernel, OpenCLNativeError> {
        // Build program from source with options
        let program = Program::builder()
            .src(source)
            .devices(context.ocl_device())
            .cmplr_opt(build_options)
            .build(context.ocl_context())?;

        Ok(OpenCLNativeKernel::new(
            program,
            Arc::clone(&context.queue),
            config,
        ))
    }
}
