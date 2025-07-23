use crate::backends::{Compiler, Kernel, KernelMetadata, Variable};
use log::debug;
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

#[derive(Clone, Copy, Debug)]
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
        write!(c_file.as_file(), "{source_code}")?;

        let opt_level = format!("-O{}", options.optimization_level);
        let mut args = vec![
            "-shared", "-fPIC", &opt_level,
            c_file.path().to_str().unwrap(),
            "-o", so_file.path().to_str().unwrap(),
        ];
        if options.debug_info { args.push("-g"); }
        if options.use_fast_math { args.push("-ffast-math"); }

        debug!("Compiling with gcc, args: {args:?}");
        let output = Command::new("gcc").args(&args).output()?;
        if !output.status.success() {
            return Err(format!(
                "gcc compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ).into());
        }

        let lib = Arc::new(unsafe { Library::new(so_file.path())? });
        let metadata = KernelMetadata {
            args_info: vec![], // TODO
            global_work_size: 1,
            local_work_size: 1,
        };

        Ok(Arc::new(CpuKernel {
            lib,
            metadata,
        }))
    }
}

pub struct CpuKernel {
    lib: Arc<Library>,
    metadata: KernelMetadata,
}


impl Kernel for CpuKernel {
    fn exec(&self, bufs: &mut [*mut std::ffi::c_void], ints: &[i32]) {
        unsafe {
            let main_fn: libloading::Symbol<unsafe extern "C" fn(*mut *mut std::ffi::c_void, *const i32)> =
                self.lib.get(b"kernel_main").unwrap();
            main_fn(bufs.as_mut_ptr(), ints.as_ptr());
        }
    }

    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}


