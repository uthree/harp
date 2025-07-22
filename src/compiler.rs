use crate::kernel::{Kernel, KernelMetadata};
use std::error::Error;
use std::process::Command;
use std::sync::Arc;
use tempfile::Builder;
use std::io::Write;
use libloading::{Library, Symbol};
use std::mem;

pub trait Compiler {
    type Options: Default + Clone;
    fn is_available(&self) -> bool;
    fn compile(
        &self,
        source_code: &str,
        options: &Self::Options,
    ) -> Result<Arc<dyn Kernel>, Box<dyn Error>>;
}

#[derive(Clone, Default)]
pub struct GccCompileOptions {
    pub optimization_level: u8,
    pub debug_info: bool,
    pub use_fast_math: bool,
}

pub struct GccCompiler;
impl Compiler for GccCompiler {
    type Options = GccCompileOptions;

    fn is_available(&self) -> bool {
        Command::new("gcc").arg("--version").output().is_ok()
    }

    fn compile(
        &self,
        source_code: &str,
        options: &Self::Options,
    ) -> Result<Arc<dyn Kernel>, Box<dyn Error>> {
        // 1. 一時ファイルを作成
        let c_file = Builder::new().prefix("kernel").suffix(".c").tempfile()?;
        let so_file = Builder::new().prefix("kernel").suffix(".so").tempfile()?;
        write!(c_file.as_file(), "{}", source_code)?;


        // 2. gccのコマンドライン引数を組み立てる
        let opt_level = format!("-O{}", options.optimization_level);
        let mut args = vec![
            "-shared",
            "-fPIC",
            &opt_level,
            c_file.path().to_str().unwrap(),
            "-o",
            so_file.path().to_str().unwrap(),
        ];
        if options.debug_info {
            args.push("-g");
        }
        if options.use_fast_math {
            args.push("-ffast-math");
        }

        // 3. gccを実行
        let output = Command::new("gcc").args(&args).output()?;
        if !output.status.success() {
            return Err(format!(
                "gcc compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        // 4. コンパイルされたライブラリをロード
        unsafe {
            let lib = Arc::new(Library::new(so_file.path())?);
            let func: Symbol<unsafe extern "C" fn()> = lib.get(b"kernel_main")?;

            // TODO: UOpからメタデータを正しく抽出する
            let metadata = KernelMetadata {
                args_info: vec![],
                global_work_size: 1,
                local_work_size: 1,
            };

            // funcのライフタイムを'staticに拡張する。
            // これは、CpuKernelがLibraryの所有権を持つことで、
            // funcが有効な間はlibが有効であることが保証されるため、安全である。
            let func = mem::transmute::<Symbol<unsafe extern "C" fn()>, Symbol<'static, unsafe extern "C" fn()>>(func);

            Ok(Arc::new(CpuKernel {
                lib,
                func,
                metadata,
                _so_file: so_file,
            }))
        }
    }
}

pub struct CpuKernel {
    lib: Arc<Library>,
    func: Symbol<'static, unsafe extern "C" fn()>,
    metadata: KernelMetadata,
    _so_file: tempfile::NamedTempFile,
}

impl Kernel for CpuKernel {
    fn exec(&self, _args: &[&crate::tensor::Variable]) {
        // TODO: argsを正しくセットアップして関数を呼び出す
        unsafe {
            (self.func)();
        }
    }
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

