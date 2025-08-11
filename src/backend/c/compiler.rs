use super::{CBuffer, CKernel};
use crate::backend::{Compiler, KernelDetails};
use libloading::Library;
use log::debug;
use std::sync::Arc;

/// A C compiler that uses shell commands to compile C code into a dynamic library.
#[derive(Default)]
pub struct CCompiler {
    // Options for the C compiler can be added here.
}

impl CCompiler {
    /// Checks if a C compiler is available on the system by running `cc --version`.
    pub fn check_availability(&self) -> bool {
        let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let result = std::process::Command::new(compiler)
            .arg("--version")
            .output();
        match result {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }
}

impl Compiler<CBuffer> for CCompiler {
    type KernelType = CKernel;

    fn new() -> Self {
        CCompiler::default()
    }

    fn is_available(&self) -> bool {
        self.check_availability()
    }

    fn with_option(&mut self, _option: ()) {
        unimplemented!();
    }

    fn compile(&mut self, code: &String, details: KernelDetails) -> Self::KernelType {
        let mut source_file = tempfile::Builder::new()
            .prefix("kernel")
            .suffix(".c")
            .tempfile_in("/tmp")
            .unwrap();
        std::io::Write::write_all(&mut source_file, code.as_bytes()).unwrap();

        let out_dir = tempfile::tempdir_in("/tmp").unwrap();

        let (lib_name, compiler) = if cfg!(target_os = "macos") {
            ("kernel.dylib", "clang")
        } else {
            ("kernel.so", "gcc")
        };
        let lib_path = out_dir.path().join(lib_name);

        debug!(
            "Running compile command: {} -shared -fPIC -O3 -o {} {}",
            compiler,
            lib_path.to_str().unwrap(),
            source_file.path().to_str().unwrap()
        );

        let output = std::process::Command::new(compiler)
            .arg("-shared")
            .arg("-fPIC")
            .arg("-O3")
            .arg("-o")
            .arg(&lib_path)
            .arg(source_file.path())
            .arg("-lm")
            .output()
            .expect("Failed to execute compiler");

        if !output.status.success() {
            panic!(
                "Compiler failed with status {:?}:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let library =
            Arc::new(unsafe { Library::new(&lib_path).expect("Failed to load dynamic library") });

        let func_name = "kernel_main".to_string();

        CKernel {
            library,
            func_name,
            details,
        }
    }
}
