pub mod buffer;
pub mod compiler;
pub mod kernel;
pub mod renderer;

pub use buffer::OpenCLBuffer;
pub use compiler::OpenCLCompiler;
pub use kernel::OpenCLKernel;
pub use renderer::OpenCLRenderer;

/// libloading用のラッパー関数名
pub const LIBLOADING_WRAPPER_NAME: &str = "__harp_entry";

/// OpenCLRenderer と OpenCLCompiler を組み合わせた Pipeline
pub type OpenCLPipeline = crate::backend::GenericPipeline<OpenCLRenderer, OpenCLCompiler>;

/// OpenCL Cコードを表す型
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCLCode {
    code: String,
    signature: crate::backend::KernelSignature,
}

impl OpenCLCode {
    /// 新しい OpenCLCode を作成
    pub fn new(code: String) -> Self {
        Self {
            code,
            signature: crate::backend::KernelSignature::empty(),
        }
    }

    /// シグネチャ付きで新しい OpenCLCode を作成
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
        write!(f, "{}", self.code)
    }
}
