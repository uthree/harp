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
    /// OpenCL fast math compilation options
    pub const FAST_MATH_OPTIONS: &'static str =
        "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations";

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

    /// Compile with fast math optimizations enabled
    ///
    /// This enables `-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations`
    /// which can significantly improve performance but may affect numerical precision.
    pub fn compile_with_fast_math(
        &self,
        context: &OpenCLDevice,
        source: &str,
        config: KernelConfig,
    ) -> Result<OpenCLKernel, OpenCLError> {
        self.compile_with_options(context, source, config, Self::FAST_MATH_OPTIONS)
    }

    /// Compile with optional fast math based on configuration
    pub fn compile_configurable(
        &self,
        context: &OpenCLDevice,
        source: &str,
        config: KernelConfig,
        fast_math: bool,
    ) -> Result<OpenCLKernel, OpenCLError> {
        if fast_math {
            self.compile_with_fast_math(context, source, config)
        } else {
            self.compile(context, source, config)
        }
    }
}
