use crate::backend::Compiler;
use crate::backend::openmp::{CBuffer, CCode, CKernel};
use libloading::Library;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// C/OpenMPコンパイラ
#[derive(Debug)]
pub struct CCompiler {
    /// コンパイラのパス（デフォルトは "gcc" または "clang"）
    compiler_path: String,
    /// 追加のコンパイラフラグ
    extra_flags: Vec<String>,
    /// 一時ファイルを保存するディレクトリ
    temp_dir: PathBuf,
}

impl CCompiler {
    /// 新しいCCompilerを作成
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

    /// C言語コードをコンパイルして動的ライブラリを生成
    fn compile_to_library(&self, code: &str, output_path: &PathBuf) -> Result<(), String> {
        // ソースファイルのパス
        let source_path = output_path.with_extension("c");

        // ソースファイルを書き込み
        fs::write(&source_path, code).map_err(|e| format!("Failed to write source file: {}", e))?;

        // コンパイルコマンドを構築
        let mut cmd = Command::new(&self.compiler_path);

        // 基本的なフラグ
        cmd.arg("-shared") // 共有ライブラリとしてコンパイル
            .arg("-fPIC") // Position Independent Code
            .arg("-fopenmp") // OpenMPサポート
            .arg("-O2") // 最適化レベル2
            .arg("-o")
            .arg(output_path)
            .arg(&source_path);

        // 追加のフラグ
        for flag in &self.extra_flags {
            cmd.arg(flag);
        }

        // コンパイルを実行
        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute compiler: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Compilation failed:\n{}", stderr));
        }

        // ソースファイルを削除（オプション）
        let _ = fs::remove_file(&source_path);

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
    type Option = ();

    fn new() -> Self {
        CCompiler::new()
    }

    fn is_available(&self) -> bool {
        // コンパイラが実行可能かチェック
        Command::new(&self.compiler_path)
            .arg("--version")
            .output()
            .is_ok()
    }

    fn compile(&mut self, code: &Self::CodeRepr) -> Self::Kernel {
        // 一時的なライブラリファイルのパス
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        #[cfg(target_os = "macos")]
        let lib_extension = "dylib";
        #[cfg(target_os = "linux")]
        let lib_extension = "so";
        #[cfg(target_os = "windows")]
        let lib_extension = "dll";

        let lib_path = self
            .temp_dir
            .join(format!("harp_kernel_{}.{}", timestamp, lib_extension));

        // コンパイル
        self.compile_to_library(code.as_str(), &lib_path)
            .expect("Failed to compile C code");

        // ライブラリをロード
        let library = unsafe { Library::new(&lib_path).expect("Failed to load compiled library") };

        // エントリーポイントを抽出
        let entry_point = Self::extract_entry_point(code);

        // シグネチャを取得
        let signature = code.signature().clone();

        CKernel::new(library, signature, entry_point, lib_path)
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
    #[ignore] // OpenMPサポートが環境依存のため、CIでは無効化
    fn test_simple_compilation() {
        let mut compiler = CCompiler::new();

        // 簡単なC言語コード
        let code = r#"
#include <math.h>
#include <omp.h>
#include <stdint.h>

void kernel_0(float** buffers) {
    float* out = buffers[0];
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        out[i] = i * 2.0f;
    }
}
"#;

        let c_code = CCode::new(code.to_string());

        // コンパイル
        let kernel = compiler.compile(&c_code);

        // カーネルが作成されたことを確認
        assert_eq!(kernel.entry_point(), "kernel_0");
    }

    #[test]
    fn test_buffer_creation() {
        use crate::backend::Buffer;

        let compiler = CCompiler::new();
        let buffer = compiler.create_buffer(vec![10, 20], 4);

        assert_eq!(buffer.shape(), vec![10, 20]);
        assert_eq!(buffer.element_size(), 4);
    }
}
