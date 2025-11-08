pub mod renderer;

#[cfg(target_os = "macos")]
pub mod buffer;

#[cfg(target_os = "macos")]
pub mod kernel;

#[cfg(target_os = "macos")]
pub mod compiler;

#[cfg(target_os = "macos")]
pub use buffer::MetalBuffer;

#[cfg(target_os = "macos")]
pub use kernel::MetalKernel;

#[cfg(target_os = "macos")]
pub use compiler::MetalCompiler;

pub use renderer::MetalRenderer;

/// MetalRenderer と MetalCompiler を組み合わせた Pipeline
#[cfg(target_os = "macos")]
pub type MetalPipeline = crate::backend::GenericPipeline<MetalRenderer, MetalCompiler>;

/// Metal Shading Language のソースコードを表す型
///
/// new type pattern を使用して、型システムで Metal 専用のコードとして扱う。
/// これにより、誤って他のバックエンドにコードを渡すことを防ぐ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalCode {
    code: String,
    signature: crate::backend::KernelSignature,
}

impl MetalCode {
    /// 新しい MetalCode を作成（シグネチャなし）
    pub fn new(code: String) -> Self {
        Self {
            code,
            signature: crate::backend::KernelSignature::empty(),
        }
    }

    /// シグネチャ付きで新しい MetalCode を作成
    pub fn with_signature(code: String, signature: crate::backend::KernelSignature) -> Self {
        Self { code, signature }
    }

    /// 内部の String への参照を取得
    pub fn as_str(&self) -> &str {
        &self.code
    }

    /// 内部の String を取得（所有権を移動）
    pub fn into_inner(self) -> String {
        self.code
    }

    /// シグネチャへの参照を取得
    pub fn signature(&self) -> &crate::backend::KernelSignature {
        &self.signature
    }

    /// コードのバイト数を取得
    pub fn len(&self) -> usize {
        self.code.len()
    }

    /// コードが空かどうか
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// 指定した文字列が含まれているかチェック
    pub fn contains(&self, pat: &str) -> bool {
        self.code.contains(pat)
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
        write!(f, "{}", self.code)
    }
}
