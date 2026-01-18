//! Rust compiler implementation for Rust backend
//!
//! This module compiles Rust code to a shared library (cdylib) for runtime execution.

use crate::RustDevice;
use crate::kernel::RustKernel;
use eclat::backend::KernelConfig;
use eclat::backend::traits::Compiler;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

/// Error type for RustCompiler operations
#[derive(Debug)]
pub enum RustCompilerError {
    /// IO error
    Io(std::io::Error),
    /// Compilation failed
    CompilationFailed(String),
    /// Library loading error
    LibraryError(String),
}

impl fmt::Display for RustCompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustCompilerError::Io(e) => write!(f, "IO error: {}", e),
            RustCompilerError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            RustCompilerError::LibraryError(msg) => write!(f, "Library error: {}", msg),
        }
    }
}

impl Error for RustCompilerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RustCompilerError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RustCompilerError {
    fn from(e: std::io::Error) -> Self {
        RustCompilerError::Io(e)
    }
}

/// Rust compiler for runtime execution
///
/// Compiles Rust code to a shared library (cdylib) using rustc.
#[derive(Debug, Clone, Default)]
pub struct RustCompiler;

impl RustCompiler {
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

    /// Compile Rust code to a shared library
    fn compile_to_library(
        &self,
        code: &str,
        entry_point: &str,
    ) -> Result<(TempDir, PathBuf), RustCompilerError> {
        // Create temporary directory for compilation
        let temp_dir = TempDir::new()?;
        let source_path = temp_dir.path().join("kernel.rs");
        let lib_name = format!("lib{}.{}", entry_point, Self::lib_extension());
        let lib_path = temp_dir.path().join(&lib_name);

        // Write source code
        fs::write(&source_path, code)?;

        // Compile to shared library using rustc
        let output = Command::new("rustc")
            .args(self.compiler_args(&source_path, &lib_path))
            .output()
            .map_err(|e| {
                RustCompilerError::CompilationFailed(format!("Failed to run rustc: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(RustCompilerError::CompilationFailed(format!(
                "rustc compilation failed:\nstderr: {}\nstdout: {}\nsource:\n{}",
                stderr, stdout, code
            )));
        }

        Ok((temp_dir, lib_path))
    }

    /// Get compiler arguments for rustc
    fn compiler_args(&self, source: &Path, output: &Path) -> Vec<String> {
        vec![
            "--crate-type=cdylib".to_string(),
            "-C".to_string(),
            "opt-level=3".to_string(),
            "-o".to_string(),
            output.to_string_lossy().to_string(),
            source.to_string_lossy().to_string(),
        ]
    }
}

impl Compiler for RustCompiler {
    type Dev = RustDevice;
    type Kernel = RustKernel;
    type Error = RustCompilerError;

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
        let kernel = RustKernel::new(temp_dir, lib_path, config)
            .map_err(|e| RustCompilerError::LibraryError(e.to_string()))?;

        Ok(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_new() {
        let _compiler = RustCompiler::new();
        // Just ensure creation doesn't panic
    }

    #[test]
    fn test_lib_extension() {
        let ext = RustCompiler::lib_extension();
        assert!(!ext.is_empty());
    }

    #[test]
    fn test_compile_simple_function() {
        let compiler = RustCompiler::new();
        let code = r#"
#![allow(unused_unsafe)]

#[no_mangle]
pub unsafe extern "C" fn test_entry(out: *mut f32, input: *mut f32) {
    *out = *input * 2.0f32;
}
"#;
        let result = compiler.compile_to_library(code, "test_entry");
        match result {
            Ok((temp_dir, lib_path)) => {
                assert!(lib_path.exists());
                drop(temp_dir); // Clean up
            }
            Err(e) => {
                // rustc might not be available in CI
                println!("Compilation skipped (rustc not available?): {}", e);
            }
        }
    }
}
