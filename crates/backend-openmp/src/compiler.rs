//! OpenMP compiler implementation
//!
//! This module compiles C code with OpenMP pragmas to a shared library for parallel execution.

use crate::kernel::OpenMPKernel;
use crate::OpenMPDevice;
use eclat::backend::traits::Compiler;
use eclat::backend::KernelConfig;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

/// Error type for OpenMPCompiler operations
#[derive(Debug)]
pub enum OpenMPCompilerError {
    /// IO error
    Io(std::io::Error),
    /// Compilation failed
    CompilationFailed(String),
    /// Library loading error
    LibraryError(String),
}

impl fmt::Display for OpenMPCompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenMPCompilerError::Io(e) => write!(f, "IO error: {}", e),
            OpenMPCompilerError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            OpenMPCompilerError::LibraryError(msg) => write!(f, "Library error: {}", msg),
        }
    }
}

impl Error for OpenMPCompilerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OpenMPCompilerError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OpenMPCompilerError {
    fn from(e: std::io::Error) -> Self {
        OpenMPCompilerError::Io(e)
    }
}

/// OpenMP compiler for parallel CPU execution
///
/// Compiles C code with OpenMP pragmas to a shared library using
/// the system C compiler with -fopenmp flag.
#[derive(Debug, Clone, Default)]
pub struct OpenMPCompiler;

impl OpenMPCompiler {
    /// Find the system C compiler
    fn find_compiler() -> &'static str {
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
    ) -> Result<(TempDir, PathBuf), OpenMPCompilerError> {
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
                OpenMPCompilerError::CompilationFailed(format!(
                    "Failed to run compiler '{}': {}",
                    compiler, e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(OpenMPCompilerError::CompilationFailed(format!(
                "stderr: {}\nstdout: {}",
                stderr, stdout
            )));
        }

        Ok((temp_dir, lib_path))
    }

    /// Get compiler arguments for the current platform (with OpenMP support)
    #[cfg(target_os = "macos")]
    fn compiler_args(&self, source: &Path, output: &Path) -> Vec<String> {
        vec![
            "-O3".to_string(),
            "-ffast-math".to_string(),
            "-Xpreprocessor".to_string(),
            "-fopenmp".to_string(),
            "-lomp".to_string(),
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
            "-fopenmp".to_string(),
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
            "/openmp".to_string(),
            "/LD".to_string(),
            format!("/Fe:{}", output.to_string_lossy()),
            source.to_string_lossy().to_string(),
        ]
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    fn compiler_args(&self, source: &Path, output: &Path) -> Vec<String> {
        vec![
            "-O3".to_string(),
            "-fopenmp".to_string(),
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-o".to_string(),
            output.to_string_lossy().to_string(),
            source.to_string_lossy().to_string(),
        ]
    }
}

impl Compiler for OpenMPCompiler {
    type Dev = OpenMPDevice;
    type Kernel = OpenMPKernel;
    type Error = OpenMPCompilerError;

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
        let kernel = OpenMPKernel::new(temp_dir, lib_path, config)
            .map_err(|e| OpenMPCompilerError::LibraryError(e.to_string()))?;

        Ok(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_new() {
        let _compiler = OpenMPCompiler::new();
    }

    #[test]
    fn test_find_compiler() {
        let compiler = OpenMPCompiler::find_compiler();
        assert!(!compiler.is_empty());
    }

    #[test]
    fn test_lib_extension() {
        let ext = OpenMPCompiler::lib_extension();
        assert!(!ext.is_empty());
    }
}
