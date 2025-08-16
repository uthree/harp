use super::{CBuffer, CKernel};
use crate::backend::Compiler;
use crate::graph::GraphSignature;
use libloading::Library;
use log::debug;
use std::sync::Arc;

/// A trait for defining platform-specific C compilation strategies.
trait CompilationStrategy {
    /// Returns the name of the dynamic library file.
    fn lib_name(&self) -> &str;

    /// Returns the name of the C compiler to use.
    fn compiler(&self) -> &str;

    /// Returns a vector of compiler arguments.
    fn args(&self) -> Vec<String>;
}

/// A compilation strategy for macOS.
struct MacOsStrategy;

impl CompilationStrategy for MacOsStrategy {
    fn lib_name(&self) -> &str {
        "kernel.dylib"
    }

    fn compiler(&self) -> &str {
        "clang"
    }

    fn args(&self) -> Vec<String> {
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
        base_args
    }
}

/// A compilation strategy for Linux.
struct LinuxStrategy;

impl CompilationStrategy for LinuxStrategy {
    fn lib_name(&self) -> &str {
        "kernel.so"
    }

    fn compiler(&self) -> &str {
        "gcc"
    }

    fn args(&self) -> Vec<String> {
        vec![
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-O3".to_string(),
            "-fopenmp".to_string(),
        ]
    }
}

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

    /// Returns the appropriate compilation strategy for the current platform.
    fn get_strategy(&self) -> Box<dyn CompilationStrategy> {
        if cfg!(target_os = "macos") {
            Box::new(MacOsStrategy)
        } else {
            Box::new(LinuxStrategy)
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
        // This compiler does not have options yet.
    }

    fn compile(&mut self, code: &String, details: GraphSignature) -> Self::KernelType {
        let mut source_file = tempfile::Builder::new()
            .prefix("kernel")
            .suffix(".c")
            .tempfile_in("/tmp")
            .unwrap();
        std::io::Write::write_all(&mut source_file, code.as_bytes()).unwrap();

        let out_dir = tempfile::tempdir_in("/tmp").unwrap();
        let strategy = self.get_strategy();
        let lib_name = strategy.lib_name();
        let compiler = strategy.compiler();
        let args = strategy.args();
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

        let library =
            Arc::new(unsafe { Library::new(&lib_path).expect("Failed to load dynamic library") });

        // We assume the function name is the first output name in the signature for now.
        // A more robust solution would be needed for multi-output graphs.
        let func_name = "kernel_main".to_string(); // Simplified

        CKernel {
            library,
            func_name,
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::backend::{Buffer, Kernel};

    #[test]
    fn test_compile_and_run_vector_add() {
        let mut compiler = CCompiler::new();
        if !compiler.is_available() {
            eprintln!("C compiler not available, skipping test.");
            return;
        }

        let code = r#"#
            void vector_add(float* out, float* a, float* b) {
                for (int i = 0; i < 10; i++) { // Hardcoded size for simplicity
                    out[i] = a[i] + b[i];
                }
            }
        "#
        .to_string();

        // The GraphSignature is a placeholder here, as the C code is manually written.
        // In a real scenario, this would be derived from the graph.
        let details = GraphSignature::new();

        let mut kernel = compiler.compile(&code, details);
        // We need to manually set the function name because the GraphSignature is empty.
        kernel.func_name = "vector_add".to_string();

        let n = 10;
        let shape = vec![n];
        let dtype = DType::F32;

        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

        let a_buffer = CBuffer::from_slice(&a_data, &shape, dtype.clone());
        let b_buffer = CBuffer::from_slice(&b_data, &shape, dtype.clone());
        let out_buffer = CBuffer::allocate(dtype, shape);

        let buffers = vec![out_buffer, a_buffer, b_buffer];
        let result_buffers = kernel.call(buffers, &[]);

        let result_data = result_buffers[0].to_vec::<f32>();
        let expected_data: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a + b)
            .collect();

        assert_eq!(result_data, expected_data);
    }
}
