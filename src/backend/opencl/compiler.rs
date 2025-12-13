use crate::backend::Compiler;
use crate::backend::c_like::OptimizationLevel;
use crate::backend::opencl::{LIBLOADING_WRAPPER_NAME, OpenCLBuffer, OpenCLCode, OpenCLKernel};
use libloading::Library;
use std::path::PathBuf;
use std::process::Command;

/// OpenCLコンパイラのオプション
#[derive(Debug, Clone)]
pub struct OpenCLCompilerOption {
    /// 最適化レベル
    pub optimization_level: OptimizationLevel,
    /// パイプを使用するか（一時ファイル回避で高速化）
    pub use_pipe: bool,
    /// コンパイラのパス（空文字列の場合は自動検出）
    pub compiler_path: String,
    /// 追加のコンパイラフラグ
    pub extra_flags: Vec<String>,
}

impl Default for OpenCLCompilerOption {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::O0,
            use_pipe: true,
            compiler_path: String::new(),
            extra_flags: Vec::new(),
        }
    }
}

impl OpenCLCompilerOption {
    /// 新しいオプションを作成
    pub fn new() -> Self {
        Self::default()
    }

    /// 最適化レベルを設定
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// パイプ使用の有効/無効を設定
    pub fn with_pipe(mut self, use_pipe: bool) -> Self {
        self.use_pipe = use_pipe;
        self
    }

    /// コンパイラパスを設定
    pub fn with_compiler_path(mut self, path: String) -> Self {
        self.compiler_path = path;
        self
    }

    /// 追加のコンパイラフラグを設定
    pub fn with_extra_flags(mut self, flags: Vec<String>) -> Self {
        self.extra_flags = flags;
        self
    }
}

/// OpenCLコンパイラ
#[derive(Debug, Clone)]
pub struct OpenCLCompiler {
    /// コンパイラのパス（デフォルトは "gcc" または "clang"）
    compiler_path: String,
    /// 追加のコンパイラフラグ
    extra_flags: Vec<String>,
    /// 一時ファイルを保存するディレクトリ
    temp_dir: PathBuf,
    /// 最適化レベル
    optimization_level: OptimizationLevel,
    /// パイプを使用するか（デフォルト: true）
    use_pipe: bool,
}

impl OpenCLCompiler {
    /// 新しいOpenCLCompilerを作成
    pub fn new() -> Self {
        Self {
            compiler_path: Self::detect_compiler(),
            extra_flags: vec![],
            temp_dir: std::env::temp_dir(),
            optimization_level: OptimizationLevel::O0, // デフォルトは最適化なし（コンパイル高速化）
            use_pipe: true,                            // デフォルトでパイプを使用
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

    /// 最適化レベルを設定
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// パイプ使用の有効/無効を設定
    ///
    /// trueの場合、コンパイル時に-pipeフラグを使用し、
    /// 一時ファイルの代わりにパイプを使用します。
    pub fn with_pipe(mut self, use_pipe: bool) -> Self {
        self.use_pipe = use_pipe;
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
            .arg(format!("-O{}", self.optimization_level.as_flag())); // 最適化レベル

        // パイプ使用
        if self.use_pipe {
            cmd.arg("-pipe");
        }

        cmd.arg("-o").arg(output_path);

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
    type Option = OpenCLCompilerOption;

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

    fn with_option(&mut self, option: Self::Option) {
        self.optimization_level = option.optimization_level;
        self.use_pipe = option.use_pipe;
        if !option.compiler_path.is_empty() {
            self.compiler_path = option.compiler_path;
        }
        if !option.extra_flags.is_empty() {
            self.extra_flags = option.extra_flags;
        }
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
