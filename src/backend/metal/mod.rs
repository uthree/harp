//! Metal backend
//!
//! This module provides Metal Shading Language kernel source rendering
//! and native GPU execution using the `metal` crate.

pub mod renderer;

// Native execution components (requires metal feature and macOS)
#[cfg(all(feature = "metal", target_os = "macos"))]
mod buffer;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod compiler;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod context;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod kernel;

pub use renderer::MetalRenderer;

// Re-export native execution types when feature is enabled
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use buffer::MetalBuffer;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use compiler::MetalCompiler;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use context::{MetalContext, MetalError};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use kernel::MetalKernel;

// OptimizationLevelはc_likeモジュールからre-export
pub use crate::backend::c_like::OptimizationLevel;

/// Metal Shading Language のソースコードを表す型
///
/// newtype pattern を使用して、型システムで Metal 専用のコードとして扱う。
/// これにより、誤って他のバックエンドにコードを渡すことを防ぐ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalCode(String);

impl MetalCode {
    /// 新しい MetalCode を作成
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

impl From<String> for MetalCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<MetalCode> for String {
    fn from(code: MetalCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for MetalCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for MetalCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
