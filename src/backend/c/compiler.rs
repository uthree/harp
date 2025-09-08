use super::{CBuffer, CKernel};
use crate::backend::Compiler;
use crate::graph::GraphSignature;
use libloading::Library;
use log::debug;
use std::sync::Arc;

#[derive(Default)]
pub struct CCompiler {}

impl CCompiler {
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

impl Compiler for CCompiler {
    type CodeRepr = String;
    type Buffer = CBuffer;
    type KernelType = CKernel;
    type Option = ();

    fn new() -> Self {
        CCompiler::default()
    }

    fn is_available(&self) -> bool {
        self.check_availability()
    }

    fn with_option(&mut self, _option: ()) {}

    fn compile(&mut self, code: &String, signature: GraphSignature) -> Self::KernelType {
        let mut source_file = tempfile::Builder::new()
            .prefix("kernel")
            .suffix(".c")
            .tempfile_in("/tmp")
            .unwrap();
        std::io::Write::write_all(&mut source_file, code.as_bytes()).unwrap();

        let out_dir = tempfile::tempdir_in("/tmp").unwrap();

        let (lib_name, compiler, args) = if cfg!(target_os = "macos") {
            let mut base_args = vec![
                "-shared".to_string(),
                "-fPIC".to_string(),
                "-O3".to_string(),
                "-Xpreprocessor".to_string(),
                "-fopenmp".to_string(),
            ];
            let omp_path_str = if cfg!(target_arch = "aarch64") {
                "/opt/homebrew/opt/libomp"
            } else {
                "/usr/local/opt/libomp"
            };
            let omp_path = std::path::Path::new(omp_path_str);
            if omp_path.exists() {
                base_args.push("-I".to_string());
                base_args.push(omp_path.join("include").to_str().unwrap().to_string());
                base_args.push("-L".to_string());
                base_args.push(omp_path.join("lib").to_str().unwrap().to_string());
            }
            base_args.push("-lomp".to_string());
            ("kernel.dylib", "clang", base_args)
        } else {
            (
                "kernel.so",
                "gcc",
                vec![
                    "-shared".to_string(),
                    "-fPIC".to_string(),
                    "-O3".to_string(),
                    "-fopenmp".to_string(),
                ],
            )
        };

        let lib_path = out_dir.path().join(lib_name);

        debug!(
            "Running compile command: {} {} -o {} {}",
            compiler,
            args.join(" "),
            lib_path.to_str().unwrap(),
            source_file.path().to_str().unwrap()
        );

        let output = std::process::Command::new(compiler)
            .args(&args)
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

        let library = Arc::new(unsafe { Library::new(&lib_path).expect("Failed to load dynamic library") });

        CKernel {
            library,
            func_name: "kernel_main".to_string(),
            signature,
        }
    }
}
