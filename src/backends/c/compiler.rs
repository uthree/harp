use crate::backends::{Compiler, Kernel, KernelMetadata};
use crate::tensor::Variable;
use std::error::Error;
use std::process::Command;
use std::sync::Arc;
use tempfile::Builder;
use std::io::Write;
use libloading::{Library, Symbol};
use std::mem;

type RawBuffer = *mut u8;

#[derive(Clone, Default, Debug)]
pub struct GccCompileOptions {
    pub optimization_level: u8,
    pub debug_info: bool,
    pub use_fast_math: bool,
}

#[derive(Debug)]
pub struct GccCompiler;
impl Compiler for GccCompiler {
    type Options = GccCompileOptions;

    fn is_available(&self) -> bool {
        Command::new("gcc").arg("--version").output().is_ok()
    }

    fn compile(
        &self,
        source_code: &str,
        options: &Self::Options,
    ) -> Result<Arc<dyn Kernel>, Box<dyn Error>> {
        let c_file = Builder::new().prefix("kernel").suffix(".c").tempfile()?;
        let so_file = Builder::new().prefix("kernel").suffix(".so").tempfile()?;
        write!(c_file.as_file(), "{}", source_code)?;

        let opt_level = format!("-O{}", options.optimization_level);
        let mut args = vec![
            "-shared", "-fPIC", &opt_level,
            c_file.path().to_str().unwrap(),
            "-o", so_file.path().to_str().unwrap(),
        ];
        if options.debug_info { args.push("-g"); }
        if options.use_fast_math { args.push("-ffast-math"); }

        let output = Command::new("gcc").args(&args).output()?;
        if !output.status.success() {
            return Err(format!(
                "gcc compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ).into());
        }

        unsafe {
            let lib = Arc::new(Library::new(so_file.path())?);
            type KernelFunc = unsafe extern "C" fn(*const RawBuffer, *const i32);
            let func: Symbol<KernelFunc> = lib.get(b"kernel_main")?;

            let metadata = KernelMetadata {
                args_info: vec![], // TODO
                global_work_size: 1,
                local_work_size: 1,
            };

            let func = mem::transmute::<Symbol<KernelFunc>, Symbol<'static, KernelFunc>>(func);

            Ok(Arc::new(CpuKernel {
                lib,
                func,
                metadata,
                _so_file: so_file,
            }))
        }
    }
}

pub struct CpuKernel {
    lib: Arc<Library>,
    func: Symbol<'static, unsafe extern "C" fn(*const RawBuffer, *const i32)>,
    metadata: KernelMetadata,
    _so_file: tempfile::NamedTempFile,
}

impl Kernel for CpuKernel {
    fn exec(&self, args: &[&Variable]) {
        let raw_buffers: Vec<RawBuffer> = args.iter().map(|v| v.0.backend.get_buffer_ptr(v.0.id)).collect();
        let int_args: Vec<i32> = vec![10]; // 仮のN

        unsafe {
            (self.func)(raw_buffers.as_ptr(), int_args.as_ptr());
        }
    }
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

