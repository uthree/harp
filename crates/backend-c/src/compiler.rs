//! C compiler implementation for C backend
//!
//! This module compiles C code to a shared library for runtime execution.

use crate::CDevice;
use crate::kernel::CKernel;
use eclat::backend::KernelConfig;
use eclat::backend::traits::Compiler;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

/// Error type for CCompiler operations
#[derive(Debug)]
pub enum CCompilerError {
    /// IO error
    Io(std::io::Error),
    /// Compilation failed
    CompilationFailed(String),
    /// Library loading error
    LibraryError(String),
}

impl fmt::Display for CCompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CCompilerError::Io(e) => write!(f, "IO error: {}", e),
            CCompilerError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            CCompilerError::LibraryError(msg) => write!(f, "Library error: {}", msg),
        }
    }
}

impl Error for CCompilerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CCompilerError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CCompilerError {
    fn from(e: std::io::Error) -> Self {
        CCompilerError::Io(e)
    }
}

/// C compiler for runtime execution
///
/// Compiles C code to a shared library using the system C compiler (cc).
#[derive(Debug, Clone, Default)]
pub struct CCompiler;

impl CCompiler {
    /// Find the system C compiler
    fn find_compiler() -> &'static str {
        // Try common compilers
        #[cfg(target_os = "macos")]
        {
            "clang"
        }
        #[cfg(target_os = "linux")]
        {
            "gcc"
        }
        #[cfg(target_os = "windows")]
        {
            "cl"
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            "cc"
        }
    }

    /// Get the shared library extension for the current platform
    fn lib_extension() -> &'static str {
        #[cfg(target_os = "macos")]
        {
            "dylib"
        }
        #[cfg(target_os = "linux")]
        {
            "so"
        }
        #[cfg(target_os = "windows")]
        {
            "dll"
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            "so"
        }
    }

    /// Compile C code to a shared library
    fn compile_to_library(
        &self,
        code: &str,
        entry_point: &str,
    ) -> Result<(TempDir, PathBuf), CCompilerError> {
        // Create temporary directory for compilation
        let temp_dir = TempDir::new()?;
        let source_path = temp_dir.path().join("kernel.c");
        let lib_name = format!("lib{}.{}", entry_point, Self::lib_extension());
        let lib_path = temp_dir.path().join(&lib_name);

        // Write source code
        fs::write(&source_path, code)?;

        // Compile to shared library
        let compiler = Self::find_compiler();
        let output = Command::new(compiler)
            .args(self.compiler_args(&source_path, &lib_path))
            .output()
            .map_err(|e| {
                CCompilerError::CompilationFailed(format!(
                    "Failed to run compiler '{}': {}",
                    compiler, e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(CCompilerError::CompilationFailed(format!(
                "stderr: {}\nstdout: {}",
                stderr, stdout
            )));
        }

        Ok((temp_dir, lib_path))
    }

    /// Get compiler arguments for the current platform
    #[cfg(target_os = "macos")]
    fn compiler_args(&self, source: &Path, output: &Path) -> Vec<String> {
        vec![
            "-O3".to_string(),
            "-ffast-math".to_string(),
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-o".to_string(),
            output.to_string_lossy().to_string(),
            source.to_string_lossy().to_string(),
        ]
    }

    #[cfg(target_os = "linux")]
    fn compiler_args(&self, source: &Path, output: &Path) -> Vec<String> {
        vec![
            "-O3".to_string(),
            "-ffast-math".to_string(),
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-o".to_string(),
            output.to_string_lossy().to_string(),
            source.to_string_lossy().to_string(),
            "-lm".to_string(),
        ]
    }

    #[cfg(target_os = "windows")]
    fn compiler_args(&self, source: &Path, output: &Path) -> Vec<String> {
        vec![
            "/O2".to_string(),
            "/LD".to_string(),
            format!("/Fe:{}", output.to_string_lossy()),
            source.to_string_lossy().to_string(),
        ]
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    fn compiler_args(&self, source: &Path, output: &Path) -> Vec<String> {
        vec![
            "-O3".to_string(),
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-o".to_string(),
            output.to_string_lossy().to_string(),
            source.to_string_lossy().to_string(),
        ]
    }
}

impl Compiler for CCompiler {
    type Dev = CDevice;
    type Kernel = CKernel;
    type Error = CCompilerError;

    fn new() -> Self {
        Self
    }

    fn compile(
        &self,
        _device: &Self::Dev,
        source: &str,
        config: KernelConfig,
    ) -> Result<Self::Kernel, Self::Error> {
        let entry_point = &config.entry_point;

        // Compile to shared library
        let (temp_dir, lib_path) = self.compile_to_library(source, entry_point)?;

        // Create kernel
        let kernel = CKernel::new(temp_dir, lib_path, config)
            .map_err(|e| CCompilerError::LibraryError(e.to_string()))?;

        Ok(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_new() {
        let _compiler = CCompiler::new();
        // Just ensure creation doesn't panic
    }

    #[test]
    fn test_find_compiler() {
        let compiler = CCompiler::find_compiler();
        assert!(!compiler.is_empty());
    }

    #[test]
    fn test_lib_extension() {
        let ext = CCompiler::lib_extension();
        assert!(!ext.is_empty());
    }

    #[test]
    fn test_compile_simple_function() {
        let compiler = CCompiler::new();
        let code = r#"
void test_entry(float* out, float* in) {
    out[0] = in[0] * 2.0f;
}
"#;
        let (temp_dir, lib_path) = compiler.compile_to_library(code, "test_entry").unwrap();
        assert!(lib_path.exists());
        drop(temp_dir); // Clean up
    }
}
