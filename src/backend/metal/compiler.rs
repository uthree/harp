use crate::backend::Compiler;
use crate::backend::metal::{MetalBuffer, MetalCode, MetalKernel};
use log::{debug, info, trace};
use metal::{CompileOptions, Device, Library, MTLSize};
use std::sync::Arc;

/// デフォルトのグリッドサイズ（1次元）
const DEFAULT_GRID_SIZE: u64 = 1024;

/// Metal コンパイラ
pub struct MetalCompiler {
    device: Arc<Device>,
    command_queue: metal::CommandQueue,
    compile_options: CompileOptions,
}

impl MetalCompiler {
    /// デフォルトのデバイスを使用してコンパイラを作成
    pub fn with_default_device() -> Option<Self> {
        Device::system_default().map(|device| {
            info!("Metal compiler initialized with device: {}", device.name());
            let command_queue = device.new_command_queue();
            let compile_options = CompileOptions::new();

            Self {
                device: Arc::new(device),
                command_queue,
                compile_options,
            }
        })
    }

    /// 指定したデバイスを使用してコンパイラを作成
    pub fn with_device(device: Device) -> Self {
        let command_queue = device.new_command_queue();
        let compile_options = CompileOptions::new();

        Self {
            device: Arc::new(device),
            command_queue,
            compile_options,
        }
    }

    /// デバイスへの参照を取得
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// バッファを作成
    pub fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> MetalBuffer {
        MetalBuffer::new(&self.device, shape, element_size)
    }

    /// ソースコードからライブラリをコンパイル
    fn compile_library(&self, source: &str) -> Result<Library, String> {
        self.device
            .new_library_with_source(source, &self.compile_options)
            .map_err(|e| format!("Metal compilation failed: {}", e))
    }
}

impl Compiler for MetalCompiler {
    type CodeRepr = MetalCode;
    type Buffer = MetalBuffer;
    type Kernel = MetalKernel;
    type Option = ();

    fn new() -> Self {
        Self::with_default_device().expect("Failed to create Metal device")
    }

    fn is_available(&self) -> bool {
        // Metal デバイスが存在すればtrue
        true
    }

    fn compile(&mut self, code: &Self::CodeRepr) -> Self::Kernel {
        info!("Compiling Metal shader source ({} bytes)", code.len());
        trace!("Metal source code:\n{}", code);

        // ソースコードからライブラリをコンパイル
        let library = self.compile_library(code.as_str()).unwrap_or_else(|err| {
            panic!(
                "Failed to compile Metal source code: {}\n\nSource code:\n{}",
                err,
                code.as_str()
            )
        });

        // カーネル関数名を取得（最初の関数を使用）
        // TODO: より適切な関数名の指定方法を実装
        let function_names = library.function_names();
        debug!("Available kernel functions: {:?}", function_names);

        let kernel_name = function_names.first().unwrap_or_else(|| {
            panic!(
                "No kernel function found in compiled library. Available functions: {:?}",
                function_names
            )
        });

        info!("Using kernel function: {}", kernel_name);

        // 関数を取得
        let function = library
            .get_function(kernel_name.as_str(), None)
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to get kernel function '{}' from library",
                    kernel_name
                )
            });

        // パイプラインステートを作成
        debug!("Creating compute pipeline state");
        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to create compute pipeline state for function '{}': {}",
                    kernel_name, err
                )
            });

        // デフォルトのグリッドサイズ（1次元）
        let grid_size = MTLSize::new(DEFAULT_GRID_SIZE, 1, 1);

        // MetalCodeからシグネチャ情報を取得
        let signature = code.signature().clone();

        info!("Metal kernel compiled successfully");
        MetalKernel::new(
            pipeline_state,
            self.command_queue.clone(),
            grid_size,
            signature,
        )
    }

    fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> Self::Buffer {
        MetalBuffer::new(&self.device, shape, element_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_compiler_creation() {
        // Metal が利用可能な環境でのみテスト
        if let Some(compiler) = MetalCompiler::with_default_device() {
            assert!(compiler.is_available());
        }
    }
}
