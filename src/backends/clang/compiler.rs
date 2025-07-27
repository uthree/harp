use crate::backends::{Buffer, Compiler, Kernel, KernelMetadata};
use libloading::{Library, Symbol};
use log::debug;
use std::error::Error;
use std::io::Write;
use std::mem;
use std::process::Command;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tempfile::{Builder, TempPath};

/// A raw pointer to a buffer, used for FFI.
type RawBuffer = *mut u8;

/// Options for compiling with Clang.
#[derive(Clone, Debug)]
pub struct ClangCompileOptions {
    /// The optimization level to use (e.g., 0, 1, 2, 3).
    pub optimization_level: u8,
    /// Whether to include debug information in the compiled artifact.
    pub debug_info: bool,
    /// Whether to enable fast math optimizations.
    pub use_fast_math: bool,
}

impl Default for ClangCompileOptions {
    fn default() -> Self {
        Self {
            optimization_level: 3,
            debug_info: false,
            use_fast_math: true,
        }
    }
}

/// A `Compiler` implementation that uses Clang to compile C code.
#[derive(Debug)]
pub struct ClangCompiler;

impl Compiler for ClangCompiler {
    type Options = ClangCompileOptions;

    fn is_available(&self) -> bool {
        Command::new("clang").arg("--version").output().is_ok()
    }

    fn compile(
        &self,
        source_code: &str,
        options: &Self::Options,
    ) -> Result<Rc<dyn Kernel>, Box<dyn Error>> {
        // Create temporary files for the C source and the shared library output.
        let c_file = Builder::new().prefix("kernel").suffix(".c").tempfile()?;
        let so_file = Builder::new().prefix("kernel").suffix(".so").tempfile()?;
        write!(c_file.as_file(), "{source_code}")?;

        // Build the command-line arguments for Clang.
        let opt_level = format!("-O{}", options.optimization_level);
        let mut args = vec![
            "-shared",
            "-fPIC",
            &opt_level,
            c_file.path().to_str().unwrap(),
            "-o",
            so_file.path().to_str().unwrap(),
        ];
        if options.debug_info {
            args.push("-g");
        }
        if options.use_fast_math {
            args.push("-ffast-math");
        }

        debug!("Compiling with clang, args: {args:?}");
        let output = Command::new("clang").args(&args).output()?;
        if !output.status.success() {
            return Err(format!(
                "clang compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        // Persist the temporary shared library file, so it's not deleted on drop.
        let so_path = so_file.into_temp_path();

        // Load the compiled shared library and get a handle to the kernel function.
        unsafe {
            let lib = Rc::new(Library::new(&so_path)?);
            type KernelFunc = unsafe extern "C" fn(*const RawBuffer, *const usize);
            let func: Symbol<KernelFunc> = lib.get(b"kernel_main")?;

            // TODO: Populate metadata properly.
            let metadata = KernelMetadata {
                args_info: vec![],
                global_work_size: 1,
                local_work_size: 1,
            };

            // Transmute the lifetime of the function symbol to 'static.
            // This is a common pattern with libloading, but it's unsafe because
            // we must ensure the Library (`_lib`) lives as long as the symbol.
            let func = mem::transmute::<Symbol<KernelFunc>, Symbol<'static, KernelFunc>>(func);

            Ok(Rc::new(ClangKernel {
                _lib: lib,
                func,
                metadata,
                _so_path: so_path,
            }))
        }
    }
}

/// A `Kernel` implementation for a compiled Clang function.
///
/// This struct holds the loaded library, the function symbol, and metadata.
/// The `_so_path` is kept to ensure the temporary file is not deleted until
/// the `ClangKernel` is dropped.
pub struct ClangKernel {
    _lib: Rc<Library>,
    func: Symbol<'static, unsafe extern "C" fn(*const RawBuffer, *const usize)>,
    metadata: KernelMetadata,
    _so_path: TempPath,
}

impl Kernel for ClangKernel {
    fn exec(&self, args: &[&Buffer], shape_args: &[usize]) -> Duration {
        let raw_buffers: Vec<RawBuffer> = args
            .iter()
            .map(|v| v.backend.get_buffer_ptr(v.id))
            .collect();

        // Execute the external C function and measure the time.
        let start = Instant::now();
        unsafe {
            (self.func)(raw_buffers.as_ptr(), shape_args.as_ptr());
        }
        start.elapsed()
    }

    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}
