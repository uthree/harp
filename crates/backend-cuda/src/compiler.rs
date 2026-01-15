//! CUDA compiler implementation
//!
//! Compiles CUDA source code to PTX using nvcc.

use crate::device::CudaDevice;
use crate::kernel::CudaKernel;
use eclat::backend::traits::Compiler;
use eclat::backend::KernelConfig;
use std::io::Write;
use std::process::Command;
use tempfile::TempDir;

/// CUDA compiler that uses nvcc to compile kernels
pub struct CudaCompiler {
    /// Target GPU architecture (e.g., "sm_50", "sm_70")
    arch: String,
    /// Optimization level
    optimization_level: OptimizationLevel,
}

/// Optimization level for CUDA compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization (-O0)
    None,
    /// Basic optimization (-O1)
    Basic,
    /// Standard optimization (-O2)
    Standard,
    /// Aggressive optimization (-O3)
    Aggressive,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Aggressive
    }
}

impl CudaCompiler {
    /// Create a CUDA compiler with a specific architecture
    pub fn with_arch(arch: &str) -> Self {
        Self {
            arch: arch.to_string(),
            optimization_level: OptimizationLevel::Aggressive,
        }
    }

    /// Set the optimization level
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Get the optimization flag string
    fn optimization_flag(&self) -> &str {
        match self.optimization_level {
            OptimizationLevel::None => "-O0",
            OptimizationLevel::Basic => "-O1",
            OptimizationLevel::Standard => "-O2",
            OptimizationLevel::Aggressive => "-O3",
        }
    }

    /// Compile CUDA source code to PTX
    ///
    /// Returns the temporary directory (to keep PTX file alive) and the PTX content.
    pub fn compile_to_ptx(
        &self,
        source: &str,
        entry_point: &str,
    ) -> Result<(TempDir, Vec<u8>), CudaCompilerError> {
        // Create temporary directory
        let temp_dir = TempDir::new()
            .map_err(|e| CudaCompilerError::IoError(format!("Failed to create temp dir: {}", e)))?;

        let source_path = temp_dir.path().join(format!("{}.cu", entry_point));
        let ptx_path = temp_dir.path().join(format!("{}.ptx", entry_point));

        // Write source code
        let mut file = std::fs::File::create(&source_path)
            .map_err(|e| CudaCompilerError::IoError(format!("Failed to create source file: {}", e)))?;
        file.write_all(source.as_bytes())
            .map_err(|e| CudaCompilerError::IoError(format!("Failed to write source: {}", e)))?;

        log::debug!("Compiling CUDA source:\n{}", source);
        log::debug!("Source path: {:?}", source_path);
        log::debug!("PTX path: {:?}", ptx_path);

        // Run nvcc
        let output = Command::new("nvcc")
            .args(&[
                "-ptx",                    // Generate PTX
                self.optimization_flag(),  // Optimization level
                "-arch",
                &self.arch,                // Target architecture
                "--use_fast_math",         // Enable fast math
                "-o",
                ptx_path.to_str().unwrap(),
                source_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| CudaCompilerError::NvccNotFound(format!("Failed to run nvcc: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(CudaCompilerError::CompilationError(format!(
                "nvcc failed:\nstdout: {}\nstderr: {}",
                stdout, stderr
            )));
        }

        // Read PTX content
        let ptx_content = std::fs::read(&ptx_path)
            .map_err(|e| CudaCompilerError::IoError(format!("Failed to read PTX: {}", e)))?;

        log::debug!("PTX compilation successful ({} bytes)", ptx_content.len());

        Ok((temp_dir, ptx_content))
    }
}

impl Default for CudaCompiler {
    fn default() -> Self {
        Self {
            arch: "sm_50".to_string(),
            optimization_level: OptimizationLevel::Aggressive,
        }
    }
}

impl Compiler for CudaCompiler {
    type Dev = CudaDevice;
    type Kernel = CudaKernel;
    type Error = CudaCompilerError;

    fn new() -> Self {
        Self::default()
    }

    fn compile(
        &self,
        device: &Self::Dev,
        source: &str,
        config: KernelConfig,
    ) -> Result<Self::Kernel, Self::Error> {
        // Compile to PTX
        let (temp_dir, ptx_content) = self.compile_to_ptx(source, &config.entry_point)?;

        // Create kernel from PTX
        CudaKernel::from_ptx(device, temp_dir, ptx_content, config)
            .map_err(|e| CudaCompilerError::KernelCreationError(e.to_string()))
    }
}

/// Error types for CUDA compilation
#[derive(Debug)]
pub enum CudaCompilerError {
    /// nvcc not found
    NvccNotFound(String),
    /// Compilation failed
    CompilationError(String),
    /// IO error
    IoError(String),
    /// Kernel creation error
    KernelCreationError(String),
}

impl std::fmt::Display for CudaCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaCompilerError::NvccNotFound(msg) => write!(f, "nvcc not found: {}", msg),
            CudaCompilerError::CompilationError(msg) => write!(f, "CUDA compilation error: {}", msg),
            CudaCompilerError::IoError(msg) => write!(f, "IO error: {}", msg),
            CudaCompilerError::KernelCreationError(msg) => write!(f, "Kernel creation error: {}", msg),
        }
    }
}

impl std::error::Error for CudaCompilerError {}
