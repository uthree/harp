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
            let args = vec!["-shared".to_string(), "-fPIC".to_string()];
            ("kernel.dylib", "clang", args)
        } else {
            let args = vec!["-shared".to_string(), "-fPIC".to_string()];
            ("kernel.so", "gcc", args)
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
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            panic!(
                "Compiler failed with status {:?}:\nSTDERR:\n{}\nSTDOUT:\n{}\nCommand: {} {}",
                output.status,
                stderr,
                stdout,
                compiler,
                args.join(" ")
            );
        }

        let library =
            Arc::new(unsafe { Library::new(&lib_path).expect("Failed to load dynamic library") });

        CKernel {
            library,
            func_name: "kernel_main".to_string(),
            signature,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::backend::Kernel;
    use crate::graph::{GraphSignature, TensorSignature};

    #[test]
    fn test_compiler_availability() {
        let compiler = CCompiler::new();
        // This test assumes a C compiler is available in the environment.
        assert!(compiler.is_available());
    }

    #[test]
    fn test_compile_and_run_simple_kernel() {
        let _ = env_logger::try_init();
        let c_code = r#"
#include <stddef.h>
#include <stdint.h>

void kernel_impl(float* a, float* b, float* c) {
    for (size_t i = 0; i < 4; ++i) {
        c[i] = a[i] + b[i];
    }
}

void kernel_main(void** buffers, size_t* shape_vars) {
    float* var0 = (float*)buffers[0];
    float* var1 = (float*)buffers[1];
    float* var2 = (float*)buffers[2];
    kernel_impl(var0, var1, var2);
}
"#
        .to_string();

        let mut compiler = CCompiler::new();
        let signature = GraphSignature {
            shape_variables: vec![],
            inputs: vec![
                TensorSignature {
                    dtype: DType::F32,
                    shape: vec![4.into()],
                },
                TensorSignature {
                    dtype: DType::F32,
                    shape: vec![4.into()],
                },
            ],
            outputs: vec![TensorSignature {
                dtype: DType::F32,
                shape: vec![4.into()],
            }],
        };

        let mut kernel = compiler.compile(&c_code, signature);

        let input_a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let output_data: Vec<f32> = vec![0.0; 4];

        let buffer_a = CBuffer::from_slice(&input_a_data, &[4], DType::F32);
        let buffer_b = CBuffer::from_slice(&input_b_data, &[4], DType::F32);
        let buffer_c = CBuffer::from_slice(&output_data, &[4], DType::F32);

        let buffers = vec![buffer_a, buffer_b, buffer_c];
        let shape_vars = vec![];

        let result_buffers = kernel.call(buffers, &shape_vars);

        let result_vec = result_buffers[2].to_vec::<f32>();
        let expected_vec: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];

        assert_eq!(result_vec, expected_vec);
    }
}
