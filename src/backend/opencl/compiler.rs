//! OpenCL native compiler

use super::context::{OpenCLContext, OpenCLError};
use super::kernel::OpenCLKernel;
use crate::backend::traits::{Compiler, KernelConfig};
use ocl::Program;
use std::sync::Arc;

/// OpenCL native compiler
///
/// Compiles OpenCL C source code into executable kernels.
pub struct OpenCLCompiler;

impl Compiler for OpenCLCompiler {
    type Context = OpenCLContext;
    type Kernel = OpenCLKernel;
    type Error = OpenCLError;

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

        Ok(OpenCLKernel::new(
            program,
            Arc::clone(&context.queue),
            config,
        ))
    }
}

impl OpenCLCompiler {
    /// Compile with build options
    pub fn compile_with_options(
        &self,
        context: &OpenCLContext,
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
