use crate::backend::Compiler;
use crate::backend::opencl::{LIBLOADING_WRAPPER_NAME, OpenCLBuffer, OpenCLCode, OpenCLKernel};
use libloading::Library;
use std::path::PathBuf;
use std::process::Command;

/// OpenCLコンパイラ
#[derive(Debug)]
pub struct OpenCLCompiler {
    /// コンパイラのパス（デフォルトは "gcc" または "clang"）
    compiler_path: String,
    /// 追加のコンパイラフラグ
    extra_flags: Vec<String>,
    /// 一時ファイルを保存するディレクトリ
    temp_dir: PathBuf,
}

impl OpenCLCompiler {
    /// 新しいOpenCLCompilerを作成
    pub fn new() -> Self {
        Self {
            compiler_path: Self::detect_compiler(),
            extra_flags: vec![],
            temp_dir: std::env::temp_dir(),
        }
    }

    /// コンパイラを自動検出
    fn detect_compiler() -> String {
        // clangが利用可能ならそれを使用、なければgcc
        if Command::new("clang").arg("--version").output().is_ok() {
            "clang".to_string()
        } else if Command::new("gcc").arg("--version").output().is_ok() {
            "gcc".to_string()
        } else {
            // デフォルトはcc（POSIX標準）
            "cc".to_string()
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

    /// OpenCL Cコードをコンパイルして動的ライブラリを生成
    ///
    /// 標準入力経由でコードを渡すことで一時ファイルの作成を回避
    fn compile_to_library(&self, code: &str, output_path: &PathBuf) -> Result<(), String> {
        use std::io::Write;
        use std::process::Stdio;

        // コンパイルコマンドを構築
        let mut cmd = Command::new(&self.compiler_path);

        // 基本的なフラグ
        cmd.arg("-x")
            .arg("c") // 入力言語をCとして指定
            .arg("-") // 標準入力から読み込み
            .arg("-shared") // 共有ライブラリとしてコンパイル
            .arg("-fPIC") // Position Independent Code
            .arg("-O2") // 最適化レベル2
            .arg("-o")
            .arg(output_path);

        // OpenCLライブラリのリンク（プラットフォーム依存）
        #[cfg(target_os = "macos")]
        {
            cmd.arg("-framework").arg("OpenCL");
        }

        #[cfg(target_os = "linux")]
        {
            cmd.arg("-lOpenCL");
        }

        #[cfg(target_os = "windows")]
        {
            cmd.arg("-lOpenCL");
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

impl Default for OpenCLCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler for OpenCLCompiler {
    type CodeRepr = OpenCLCode;
    type Buffer = OpenCLBuffer;
    type Kernel = OpenCLKernel;
    type Option = ();

    fn new() -> Self {
        OpenCLCompiler::new()
    }

    fn is_available(&self) -> bool {
        // コンパイラが実行可能かチェック
        Command::new(&self.compiler_path)
            .arg("--version")
            .output()
            .is_ok()
    }

    fn compile(
        &mut self,
        code: &Self::CodeRepr,
        signature: crate::backend::KernelSignature,
    ) -> Self::Kernel {
        // 一時ファイルを作成（適切な拡張子を設定）
        #[cfg(target_os = "macos")]
        let lib_suffix = ".dylib";
        #[cfg(target_os = "linux")]
        let lib_suffix = ".so";
        #[cfg(target_os = "windows")]
        let lib_suffix = ".dll";

        let temp_file = tempfile::Builder::new()
            .prefix("harp_opencl_kernel_")
            .suffix(lib_suffix)
            .tempfile_in(&self.temp_dir)
            .expect("Failed to create temporary file");

        let lib_path = temp_file.path().to_path_buf();

        // コンパイル
        self.compile_to_library(code.as_str(), &lib_path)
            .expect("Failed to compile OpenCL code");

        // ライブラリをロード
        let library = unsafe { Library::new(&lib_path).expect("Failed to load compiled library") };

        // OpenCLKernelを作成
        OpenCLKernel::new(
            library,
            signature,
            LIBLOADING_WRAPPER_NAME.to_string(),
            temp_file,
        )
    }

    fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> Self::Buffer {
        OpenCLBuffer::new(shape, element_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Buffer;

    #[test]
    fn test_compiler_detection() {
        let compiler = OpenCLCompiler::new();
        // コンパイラが検出されることを確認
        assert!(!compiler.compiler_path.is_empty());
    }

    #[test]
    fn test_compiler_availability() {
        let compiler = OpenCLCompiler::new();
        // コンパイラが利用可能であることを確認
        assert!(compiler.is_available());
    }

    #[test]
    fn test_buffer_creation() {
        let compiler = OpenCLCompiler::new();
        let buffer = compiler.create_buffer(vec![10, 20], 4);
        assert_eq!(buffer.shape(), vec![10, 20]);
        assert_eq!(buffer.element_size(), 4);
    }
}
