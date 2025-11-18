use crate::backend::Compiler;
use crate::backend::metal::{LIBLOADING_WRAPPER_NAME, MetalBuffer, MetalCode, MetalKernel};
use libloading::Library;
use std::path::PathBuf;
use std::process::Command;

/// Metalコンパイラ（C++ラッパー方式）
#[derive(Debug)]
pub struct MetalCompiler {
    /// コンパイラのパス（デフォルトは "clang++"）
    compiler_path: String,
    /// 追加のコンパイラフラグ
    extra_flags: Vec<String>,
    /// 一時ファイルを保存するディレクトリ
    temp_dir: PathBuf,
}

impl MetalCompiler {
    /// 新しいMetalCompilerを作成
    pub fn new() -> Self {
        Self {
            compiler_path: Self::detect_compiler(),
            extra_flags: vec![],
            temp_dir: std::env::temp_dir(),
        }
    }

    /// コンパイラを自動検出
    fn detect_compiler() -> String {
        // C++コンパイラを検出
        if Command::new("clang++").arg("--version").output().is_ok() {
            "clang++".to_string()
        } else if Command::new("g++").arg("--version").output().is_ok() {
            "g++".to_string()
        } else {
            "c++".to_string()
        }
    }

    /// コンパイラパスを設定
    pub fn with_compiler(mut self, path: String) -> Self {
        self.compiler_path = path;
        self
    }

    /// 追加のコンパイラフラグを設定
    pub fn with_flags(mut self, flags: Vec<String>) -> Self {
        self.extra_flags = flags;
        self
    }

    /// 一時ディレクトリを設定
    pub fn with_temp_dir(mut self, dir: PathBuf) -> Self {
        self.temp_dir = dir;
        self
    }

    /// Metal C++コードをコンパイルして動的ライブラリを生成
    ///
    /// 標準入力経由でコードを渡すことで一時ファイルの作成を回避
    fn compile_to_library(&self, code: &str, output_path: &PathBuf) -> Result<(), String> {
        use std::io::Write;
        use std::process::Stdio;

        // コンパイルコマンドを構築
        let mut cmd = Command::new(&self.compiler_path);

        // 基本的なフラグ
        cmd.arg("-x")
            .arg("objective-c++") // Objective-C++として扱う
            .arg("-") // 標準入力から読み込み
            .arg("-shared") // 共有ライブラリとしてコンパイル
            .arg("-fPIC") // Position Independent Code
            .arg("-O2") // 最適化レベル2
            .arg("-std=c++17") // C++17標準を使用
            .arg("-o")
            .arg(output_path);

        // Metalフレームワークをリンク（macOS専用）
        #[cfg(target_os = "macos")]
        {
            cmd.arg("-framework").arg("Metal");
            cmd.arg("-framework").arg("Foundation");
        }

        // 追加のフラグ
        for flag in &self.extra_flags {
            cmd.arg(flag);
        }

        // 標準入力を設定
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // プロセスを起動
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to execute compiler: {}", e))?;

        // 標準入力にコードを書き込み
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(code.as_bytes())
                .map_err(|e| format!("Failed to write to compiler stdin: {}", e))?;
        }

        // コンパイル結果を待機
        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for compiler: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Compilation failed:\n{}", stderr));
        }

        Ok(())
    }
}

impl Default for MetalCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler for MetalCompiler {
    type CodeRepr = MetalCode;
    type Buffer = MetalBuffer;
    type Kernel = MetalKernel;
    type Option = ();

    fn new() -> Self {
        MetalCompiler::new()
    }

    fn is_available(&self) -> bool {
        // コンパイラが実行可能かチェック
        Command::new(&self.compiler_path)
            .arg("--version")
            .output()
            .is_ok()
    }

    fn compile(&mut self, code: &Self::CodeRepr) -> Self::Kernel {
        // 一時ファイルを作成（適切な拡張子を設定）
        #[cfg(target_os = "macos")]
        let lib_suffix = ".dylib";
        #[cfg(target_os = "linux")]
        let lib_suffix = ".so";
        #[cfg(target_os = "windows")]
        let lib_suffix = ".dll";

        let temp_file = tempfile::Builder::new()
            .prefix("harp_metal_kernel_")
            .suffix(lib_suffix)
            .tempfile_in(&self.temp_dir)
            .expect("Failed to create temporary file");

        let lib_path = temp_file.path().to_path_buf();

        // コンパイル
        self.compile_to_library(code.as_str(), &lib_path)
            .expect("Failed to compile Metal C++ code");

        // ライブラリをロード
        let library = unsafe { Library::new(&lib_path).expect("Failed to load compiled library") };

        // シグネチャを取得
        let signature = code.signature().clone();

        // MetalKernelを作成
        MetalKernel::new(
            library,
            signature,
            LIBLOADING_WRAPPER_NAME.to_string(),
            temp_file,
        )
    }

    fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> Self::Buffer {
        MetalBuffer::new(shape, element_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Buffer;

    #[test]
    fn test_compiler_detection() {
        let compiler = MetalCompiler::new();
        // コンパイラが検出されることを確認
        assert!(!compiler.compiler_path.is_empty());
    }

    #[test]
    fn test_compiler_availability() {
        let compiler = MetalCompiler::new();
        // コンパイラが利用可能であることを確認
        assert!(compiler.is_available());
    }

    #[test]
    fn test_buffer_creation() {
        let compiler = MetalCompiler::new();
        let buffer = compiler.create_buffer(vec![10, 20], 4);
        assert_eq!(buffer.shape(), vec![10, 20]);
        assert_eq!(buffer.byte_len(), 10 * 20 * 4);
    }
}
