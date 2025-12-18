//! OpenCL backend
//!
//! This module provides OpenCL kernel source rendering
//! and native GPU execution using the `ocl` crate.

pub mod renderer;

// Native execution components (requires opencl feature)
#[cfg(feature = "opencl")]
mod buffer;
#[cfg(feature = "opencl")]
mod compiler;
#[cfg(feature = "opencl")]
mod device;
#[cfg(feature = "opencl")]
mod kernel;

pub use renderer::OpenCLRenderer;

// Re-export native execution types when feature is enabled
#[cfg(feature = "opencl")]
pub use buffer::OpenCLBuffer;
#[cfg(feature = "opencl")]
pub use compiler::OpenCLCompiler;
#[cfg(feature = "opencl")]
pub use device::{OpenCLDevice, OpenCLError};
#[cfg(feature = "opencl")]
pub use kernel::OpenCLKernel;

// OptimizationLevelはc_likeモジュールからre-export
pub use crate::backend::c_like::OptimizationLevel;

/// OpenCL Cコードを表す型
///
/// newtype pattern を使用して、型システムで OpenCL 専用のコードとして扱う。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCLCode(String);

impl OpenCLCode {
    /// 新しい OpenCLCode を作成
    pub fn new(code: String) -> Self {
        Self(code)
    }

    /// 内部の String への参照を取得
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// 内部の String を取得（所有権を移動）
    pub fn into_inner(self) -> String {
        self.0
    }

    /// コードのバイト数を取得
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// コードが空かどうか
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// 指定した文字列が含まれているかチェック
    pub fn contains(&self, pat: &str) -> bool {
        self.0.contains(pat)
    }
}

impl From<String> for OpenCLCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<OpenCLCode> for String {
    fn from(code: OpenCLCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for OpenCLCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for OpenCLCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
