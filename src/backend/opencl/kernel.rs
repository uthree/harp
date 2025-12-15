use crate::backend::opencl::OpenCLBuffer;
use crate::backend::{Kernel, KernelSignature};
use libloading::{Library, Symbol};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::NamedTempFile;

/// OpenCLカーネルの実行関数の型
///
/// カーネル関数は以下の引数を受け取る:
/// - buffers: バッファポインタの配列
/// - sizes: 各バッファのバイトサイズの配列
/// - is_outputs: 出力バッファかどうかのフラグ配列 (0: 入力, 1: 出力)
/// - num_buffers: バッファの数
///
/// 例: `void __harp_entry(void** buffers, size_t* sizes, int* is_outputs, int num_buffers)`
type KernelFn = unsafe extern "C" fn(*mut *mut u8, *mut usize, *mut i32, i32);

/// OpenCL用のカーネル
///
/// 一時ファイル（コンパイルされた動的ライブラリ）を保持し、
/// OpenCLKernelがDropされると自動的に削除される
///
/// `Clone`を実装しており、複数のインスタンスがライブラリを共有できます。
/// 実際の実行時にはライブラリを再ロードするため、共有は安全です。
#[derive(Clone)]
pub struct OpenCLKernel {
    signature: KernelSignature,
    entry_point: String,
    /// 一時ファイルを保持（Dropで自動削除、Arcで共有可能）
    _temp_file: Arc<NamedTempFile>,
}

impl OpenCLKernel {
    /// 新しいOpenCLKernelを作成
    ///
    /// # Arguments
    /// * `_library` - 動的ライブラリ（下位互換性のため受け取るが内部では使用しない）
    /// * `signature` - カーネルシグネチャ
    /// * `entry_point` - エントリーポイント関数名
    /// * `temp_file` - 一時ファイル（動的ライブラリ）
    pub fn new(
        _library: Library,
        signature: KernelSignature,
        entry_point: String,
        temp_file: NamedTempFile,
    ) -> Self {
        // _libraryは使用せず、必要時に再ロードする
        // これによりCloneが可能になる
        Self {
            signature,
            entry_point,
            _temp_file: Arc::new(temp_file),
        }
    }

    /// カーネルを実行
    ///
    /// # Safety
    /// この関数はunsafeです。呼び出し側は以下を保証する必要があります：
    /// - バッファの数と順序が署名と一致している
    /// - バッファのサイズが十分である
    pub unsafe fn execute(&self, buffers: &mut [&mut OpenCLBuffer]) -> Result<(), String> {
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

        // バッファサイズの配列を作成
        let mut sizes: Vec<usize> = buffers.iter().map(|b| b.byte_len()).collect();

        // 出力フラグの配列を作成
        // シグネチャから入力と出力を判別
        let num_inputs = self.signature.inputs.len();
        let mut is_outputs: Vec<i32> = buffers
            .iter()
            .enumerate()
            .map(|(i, _)| if i >= num_inputs { 1 } else { 0 })
            .collect();

        let num_buffers = buffers.len() as i32;

        // カーネル関数を呼び出し
        unsafe {
            kernel_fn(
                buffer_ptrs.as_mut_ptr(),
                sizes.as_mut_ptr(),
                is_outputs.as_mut_ptr(),
                num_buffers,
            );
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

impl Kernel for OpenCLKernel {
    type Buffer = OpenCLBuffer;

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

        // バッファサイズの配列を作成
        let mut sizes: Vec<usize> = buffers.iter().map(|b| b.byte_len()).collect();

        // 出力フラグの配列を作成
        // シグネチャから入力と出力を判別
        let num_inputs = self.signature.inputs.len();
        let mut is_outputs: Vec<i32> = buffers
            .iter()
            .enumerate()
            .map(|(i, _)| if i >= num_inputs { 1 } else { 0 })
            .collect();

        let num_buffers = buffers.len() as i32;

        // カーネル関数を呼び出し
        unsafe {
            kernel_fn(
                buffer_ptrs.as_mut_ptr(),
                sizes.as_mut_ptr(),
                is_outputs.as_mut_ptr(),
                num_buffers,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_kernel_signature() {
        // 空のシグネチャでテスト
        let signature = KernelSignature::empty();

        // Note: 実際のLibraryを作成せずにテストするのは困難なので、
        // ここでは署名の取得のみテスト
        // 実際の実行テストはコンパイラのテストで行う
        assert_eq!(signature.inputs.len(), 0);
        assert_eq!(signature.outputs.len(), 0);
    }
}
