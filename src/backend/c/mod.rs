pub mod buffer;
pub mod compiler;
pub mod kernel;
pub mod renderer;

pub use buffer::CBuffer;
pub use compiler::CCompiler;
pub use kernel::CKernel;
pub use renderer::CRenderer;

/// libloading用のラッパー関数名
///
/// libloadingは固定シグネチャを期待するため、エントリーポイント関数を
/// ラップする関数を生成する。この定数はレンダラーとコンパイラの両方で使用される。
pub const LIBLOADING_WRAPPER_NAME: &str = "__harp_entry";

/// C言語バックエンド専用のPipeline
///
/// GenericPipelineの特殊化として定義。
/// グラフ最適化は常に有効です（LoweringSuggesterによるCustomノード変換が必須）。
pub type CPipeline = crate::backend::GenericPipeline<CRenderer, CCompiler>;

/// C言語（シングルスレッド）のソースコードを表す型
///
/// new type pattern を使用して、型システムで C 専用のコードとして扱う。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CCode {
    code: String,
    signature: crate::backend::KernelSignature,
}

impl CCode {
    /// 新しい CCode を作成（シグネチャなし）
    pub fn new(code: String) -> Self {
        Self {
            code,
            signature: crate::backend::KernelSignature::empty(),
        }
    }

    /// シグネチャ付きで新しい CCode を作成
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

impl From<String> for CCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<CCode> for String {
    fn from(code: CCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for CCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for CCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code)
    }
}

impl crate::backend::SignedCode for CCode {
    fn with_signature(code: String, signature: crate::backend::KernelSignature) -> Self {
        Self { code, signature }
    }

    fn signature(&self) -> &crate::backend::KernelSignature {
        &self.signature
    }
}
