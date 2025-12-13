use crate::backend::c::CBuffer;
use crate::backend::{Kernel, KernelSignature};
use libloading::{Library, Symbol};
use std::path::PathBuf;
use tempfile::NamedTempFile;

/// Cカーネルの実行関数の型
///
/// カーネル関数は複数のポインタを引数として受け取る
/// 例: void kernel_0(float* in0, float* in1, float* out0)
type KernelFn = unsafe extern "C" fn(*mut *mut u8);

/// C用のカーネル（シングルスレッド実行専用）
///
/// 一時ファイル（コンパイルされた動的ライブラリ）を保持し、
/// CKernelがDropされると自動的に削除される
pub struct CKernel {
    _library: Library,
    signature: KernelSignature,
    entry_point: String,
    /// 一時ファイルを保持（Dropで自動削除）
    _temp_file: NamedTempFile,
}

impl CKernel {
    /// 新しいCKernelを作成
    pub fn new(
        library: Library,
        signature: KernelSignature,
        entry_point: String,
        temp_file: NamedTempFile,
    ) -> Self {
        Self {
            _library: library,
            signature,
            entry_point,
            _temp_file: temp_file,
        }
    }

    /// カーネルを実行
    ///
    /// # Safety
    /// この関数はunsafeです。呼び出し側は以下を保証する必要があります：
    /// - バッファの数と順序が署名と一致している
    /// - バッファのサイズが十分である
    pub unsafe fn execute(&self, buffers: &mut [&mut CBuffer]) -> Result<(), String> {
        // 動的ライブラリを再ロード（関数ポインタを取得するため）
        let lib = unsafe {
            Library::new(self._temp_file.path())
                .map_err(|e| format!("Failed to load library: {}", e))?
        };

        // エントリーポイント関数を取得
        let kernel_fn: Symbol<KernelFn> = unsafe {
            lib.get(self.entry_point.as_bytes()).map_err(|e| {
                format!(
                    "Failed to get kernel function '{}': {}",
                    self.entry_point, e
                )
            })?
        };

        // バッファポインタの配列を作成
        let mut buffer_ptrs: Vec<*mut u8> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();

        // カーネル関数を呼び出し
        unsafe {
            kernel_fn(buffer_ptrs.as_mut_ptr());
        }

        Ok(())
    }

    /// ライブラリパスを取得
    pub fn library_path(&self) -> PathBuf {
        self._temp_file.path().to_path_buf()
    }

    /// エントリーポイント名を取得
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
}

impl Kernel for CKernel {
    type Buffer = CBuffer;

    fn signature(&self) -> KernelSignature {
        self.signature.clone()
    }

    unsafe fn execute(&self, buffers: &mut [&mut Self::Buffer]) -> Result<(), String> {
        // 動的ライブラリを再ロード（関数ポインタを取得するため）
        let lib = unsafe {
            Library::new(self._temp_file.path())
                .map_err(|e| format!("Failed to load library: {}", e))?
        };

        // エントリーポイント関数を取得
        let kernel_fn: Symbol<KernelFn> = unsafe {
            lib.get(self.entry_point.as_bytes()).map_err(|e| {
                format!(
                    "Failed to get kernel function '{}': {}",
                    self.entry_point, e
                )
            })?
        };

        // バッファポインタの配列を作成
        let mut buffer_ptrs: Vec<*mut u8> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();

        // カーネル関数を呼び出し
        unsafe {
            kernel_fn(buffer_ptrs.as_mut_ptr());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ckernel_signature() {
        // 空のシグネチャでテスト
        let signature = KernelSignature::empty();

        // Note: 実際のLibraryを作成せずにテストするのは困難なので、
        // ここでは署名の取得のみテスト
        // 実際の実行テストはコンパイラのテストで行う
        assert_eq!(signature.inputs.len(), 0);
        assert_eq!(signature.outputs.len(), 0);
    }
}
