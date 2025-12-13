use crate::backend::Compiler;
use crate::backend::c::{CBuffer, CCode, CKernel, LIBLOADING_WRAPPER_NAME};
use libloading::Library;
use std::path::PathBuf;
use std::process::Command;

/// コンパイラの最適化レベル
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// 最適化なし（-O0）- コンパイル高速、デバッグ向け
    O0,
    /// 基本的な最適化（-O1）
    O1,
    /// 標準的な最適化（-O2）
    O2,
    /// 積極的な最適化（-O3）
    #[default]
    O3,
}

impl OptimizationLevel {
    /// コンパイラフラグとして使用する数値を返す
    pub fn as_flag(&self) -> u8 {
        match self {
            OptimizationLevel::O0 => 0,
            OptimizationLevel::O1 => 1,
            OptimizationLevel::O2 => 2,
            OptimizationLevel::O3 => 3,
        }
    }
}

/// Cコンパイラのオプション
#[derive(Debug, Clone)]
pub struct CCompilerOption {
    /// 最適化レベル
    pub optimization_level: OptimizationLevel,
    /// パイプを使用するか（一時ファイル回避で高速化）
    pub use_pipe: bool,
    /// コンパイラのパス（空文字列の場合は自動検出）
    pub compiler_path: String,
    /// 追加のコンパイラフラグ
    pub extra_flags: Vec<String>,
}

impl Default for CCompilerOption {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::O0,
            use_pipe: true,
            compiler_path: String::new(),
            extra_flags: Vec::new(),
        }
    }
}

impl CCompilerOption {
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

/// Cコンパイラ（シングルスレッド実行専用）
#[derive(Debug)]
pub struct CCompiler {
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

impl CCompiler {
    /// 新しいCCompilerを作成
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

    /// C言語コードをコンパイルして動的ライブラリを生成
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

    /// コードからエントリーポイントを抽出
    ///
    /// 現在は単純に最初の関数名を使用
    /// TODO: Programから適切なエントリーポイントを取得
    fn extract_entry_point(code: &CCode) -> String {
        // 簡易的な実装: "void kernel_" で始まる関数を探す
        for line in code.as_str().lines() {
            if (line.contains("void kernel_") || line.contains("int kernel_"))
                && let Some(start) = line.find("kernel_")
                && let Some(end) = line[start..].find('(')
            {
                return line[start..start + end].to_string();
            }
        }
        // デフォルトはkernel_0
        "kernel_0".to_string()
    }
}

impl Default for CCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler for CCompiler {
    type CodeRepr = CCode;
    type Buffer = CBuffer;
    type Kernel = CKernel;
    type Option = CCompilerOption;

    fn new() -> Self {
        CCompiler::new()
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
            .prefix("harp_kernel_")
            .suffix(lib_suffix)
            .tempfile_in(&self.temp_dir)
            .expect("Failed to create temporary file");

        let lib_path = temp_file.path().to_path_buf();

        // コンパイル
        if let Err(e) = self.compile_to_library(code.as_str(), &lib_path) {
            // エラー時に生成されたコードを出力
            eprintln!(
                "=== Generated C code ===\n{}\n=== End of code ===",
                code.as_str()
            );
            panic!("Failed to compile C code: {:?}", e);
        }

        // ライブラリをロード
        let library = unsafe { Library::new(&lib_path).expect("Failed to load compiled library") };

        // エントリーポイントを抽出（デバッグ用に保存）
        let _original_entry_point = Self::extract_entry_point(code);

        // libloading用のラッパー関数名を使用
        CKernel::new(
            library,
            signature,
            LIBLOADING_WRAPPER_NAME.to_string(),
            temp_file,
        )
    }

    fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> Self::Buffer {
        CBuffer::new(shape, element_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_detection() {
        let compiler = CCompiler::new();
        assert!(!compiler.compiler_path.is_empty());
        println!("Detected compiler: {}", compiler.compiler_path);
    }

    #[test]
    fn test_compiler_availability() {
        let compiler = CCompiler::new();
        assert!(compiler.is_available());
    }

    #[test]
    fn test_buffer_creation() {
        use crate::backend::Buffer;

        let compiler = CCompiler::new();
        let buffer = compiler.create_buffer(vec![10, 20], 4);

        assert_eq!(buffer.shape(), vec![10, 20]);
        assert_eq!(buffer.element_size(), 4);
    }

    /// 結合テスト: 簡単なベクトル加算をコンパイル・実行し、計算結果を検証
    #[test]
    fn test_integration_vector_add() {
        use crate::backend::c::CRenderer;
        use crate::graph::{DType as GraphDType, Graph};
        use crate::lowerer::{create_signature, lower};

        // 1. グラフを作成: a + b
        let mut graph = Graph::new();
        let a = graph.input("a", GraphDType::F32, vec![4]);
        let b = graph.input("b", GraphDType::F32, vec![4]);
        let result = a + b;
        graph.output("result", result);

        // 2. Create signature from graph
        let signature = create_signature(&graph);

        // 3. Lower to AST
        let program = lower(graph);

        // 4. Render to C code
        let mut renderer = CRenderer::new();
        let c_code = renderer.render_program(&program);

        println!("Generated C code:\n{}", c_code.as_str());

        // 5. Compile
        let mut compiler = CCompiler::new();
        let kernel = compiler.compile(&c_code, signature);

        // 6. Create buffers
        let mut input_a = CBuffer::from_f32_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let mut input_b = CBuffer::from_f32_vec(vec![10.0f32, 20.0, 30.0, 40.0]);
        let mut output = CBuffer::new(vec![4], std::mem::size_of::<f32>());

        // 7. Execute
        unsafe {
            kernel
                .execute(&mut [&mut input_a, &mut input_b, &mut output])
                .expect("Kernel execution failed");
        }

        // 8. Verify results
        let result_vec = output.as_f32_slice().unwrap().to_vec();
        assert_eq!(result_vec, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// 結合テスト: 要素ごとの乗算をコンパイル・実行し、計算結果を検証
    #[test]
    fn test_integration_vector_mul() {
        use crate::backend::c::CRenderer;
        use crate::graph::{DType as GraphDType, Graph};
        use crate::lowerer::{create_signature, lower};

        // 1. グラフを作成: a * b
        let mut graph = Graph::new();
        let a = graph.input("a", GraphDType::F32, vec![4]);
        let b = graph.input("b", GraphDType::F32, vec![4]);
        let result = a * b;
        graph.output("result", result);

        // 2. Create signature from graph
        let signature = create_signature(&graph);

        // 3. Lower to AST
        let program = lower(graph);

        // 4. Render to C code
        let mut renderer = CRenderer::new();
        let c_code = renderer.render_program(&program);

        // 5. Compile
        let mut compiler = CCompiler::new();
        let kernel = compiler.compile(&c_code, signature);

        // 6. Create buffers
        let mut input_a = CBuffer::from_f32_vec(vec![2.0f32, 3.0, 4.0, 5.0]);
        let mut input_b = CBuffer::from_f32_vec(vec![10.0f32, 20.0, 30.0, 40.0]);
        let mut output = CBuffer::new(vec![4], std::mem::size_of::<f32>());

        // 7. Execute
        unsafe {
            kernel
                .execute(&mut [&mut input_a, &mut input_b, &mut output])
                .expect("Kernel execution failed");
        }

        // 8. Verify results
        let result_vec = output.as_f32_slice().unwrap().to_vec();
        assert_eq!(result_vec, vec![20.0, 60.0, 120.0, 200.0]);
    }

    /// 結合テスト: 複数ステップの計算 (a + b) * c
    #[test]
    fn test_integration_multi_step_computation() {
        use crate::backend::c::CRenderer;
        use crate::graph::{DType as GraphDType, Graph};
        use crate::lowerer::{create_signature, lower};

        // 1. グラフを作成: (a + b) * c
        let mut graph = Graph::new();
        let a = graph.input("a", GraphDType::F32, vec![4]);
        let b = graph.input("b", GraphDType::F32, vec![4]);
        let c = graph.input("c", GraphDType::F32, vec![4]);
        let sum_ab = a + b;
        let result = sum_ab * c;
        graph.output("result", result);

        // 2. Create signature from graph
        let signature = create_signature(&graph);

        // 3. Lower to AST
        let program = lower(graph);

        // 4. Render to C code
        let mut renderer = CRenderer::new();
        let c_code = renderer.render_program(&program);

        println!("Generated C code for multi-step:\n{}", c_code.as_str());

        // 5. Compile
        let mut compiler = CCompiler::new();
        let kernel = compiler.compile(&c_code, signature);

        // 6. Create buffers
        // Note: lowerer orders parameters alphabetically
        // The generated code expects: harp_main(input0, input1, input2, output)
        // where input0=a, input1=b, input2=c (alphabetically sorted)
        let mut input_a = CBuffer::from_f32_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let mut input_b = CBuffer::from_f32_vec(vec![2.0f32, 3.0, 4.0, 5.0]);
        let mut input_c = CBuffer::from_f32_vec(vec![10.0f32, 10.0, 10.0, 10.0]);
        let mut output = CBuffer::new(vec![4], std::mem::size_of::<f32>());

        // 7. Execute
        unsafe {
            kernel
                .execute(&mut [&mut input_a, &mut input_b, &mut input_c, &mut output])
                .expect("Kernel execution failed");
        }

        // 8. Verify results
        // Computation: (a + b) * c
        // (1+2)*10=30, (2+3)*10=50, (3+4)*10=70, (4+5)*10=90
        let result_vec = output.as_f32_slice().unwrap().to_vec();
        assert_eq!(result_vec, vec![30.0, 50.0, 70.0, 90.0]);
    }

    /// 結合テスト: 2次元配列の加算
    #[test]
    fn test_integration_2d_array_add() {
        use crate::backend::c::CRenderer;
        use crate::graph::{DType as GraphDType, Graph};
        use crate::lowerer::{create_signature, lower};

        // 1. グラフを作成: 2x3 配列の加算
        let mut graph = Graph::new();
        let a = graph.input("a", GraphDType::F32, vec![2, 3]);
        let b = graph.input("b", GraphDType::F32, vec![2, 3]);
        let result = a + b;
        graph.output("result", result);

        // 2. Create signature from graph
        let signature = create_signature(&graph);

        // 3. Lower to AST
        let program = lower(graph);

        // 4. Render to C code
        let mut renderer = CRenderer::new();
        let c_code = renderer.render_program(&program);

        // 5. Compile
        let mut compiler = CCompiler::new();
        let kernel = compiler.compile(&c_code, signature);

        // 6. Create buffers (row-major order)
        let mut input_a = CBuffer::from_f32_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut input_b = CBuffer::from_f32_vec(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let mut output = CBuffer::new(vec![2, 3], std::mem::size_of::<f32>());

        // 7. Execute
        unsafe {
            kernel
                .execute(&mut [&mut input_a, &mut input_b, &mut output])
                .expect("Kernel execution failed");
        }

        // 8. Verify results
        let result_vec = output.as_f32_slice().unwrap().to_vec();
        assert_eq!(result_vec, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    // TODO: reduce_sum テストはレンダラーのバグ（acc変数の宣言がない）を修正後に追加
}
