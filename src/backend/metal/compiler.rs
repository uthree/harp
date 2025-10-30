use crate::backend::metal::MetalCode;
use crate::backend::{Buffer, Compiler, Kernel, KernelSignature};
use log::{debug, info, trace};
use metal::{
    Buffer as MTLBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};
use std::sync::Arc;

/// デフォルトのスレッドグループサイズの最大値
const DEFAULT_MAX_THREAD_GROUP_SIZE: u64 = 256;

/// デフォルトのグリッドサイズ（1次元）
const DEFAULT_GRID_SIZE: u64 = 1024;

/// Metal デバイスバッファのラッパー
pub struct MetalBuffer {
    buffer: MTLBuffer,
    shape: Vec<usize>,
    element_size: usize,
}

impl MetalBuffer {
    /// 新しいバッファを作成
    pub fn new(device: &Device, shape: Vec<usize>, element_size: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let byte_size = total_elements * element_size;

        debug!(
            "Creating Metal buffer: shape={:?}, element_size={}, total_bytes={}",
            shape, element_size, byte_size
        );

        let buffer = device.new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared);

        Self {
            buffer,
            shape,
            element_size,
        }
    }

    /// 既存の MTLBuffer からラップ
    pub fn from_buffer(buffer: MTLBuffer, shape: Vec<usize>, element_size: usize) -> Self {
        Self {
            buffer,
            shape,
            element_size,
        }
    }

    /// 内部の MTLBuffer への参照を取得
    pub fn inner(&self) -> &MTLBuffer {
        &self.buffer
    }

    /// バッファのバイトサイズを取得
    pub fn byte_size(&self) -> usize {
        self.buffer.length() as usize
    }

    /// 要素のサイズを取得（バイト単位）
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// データを CPU から GPU へコピー
    pub fn write_data<T: Copy>(&mut self, data: &[T]) {
        trace!(
            "Writing {} elements to Metal buffer (shape={:?})",
            data.len(),
            self.shape
        );
        let ptr = self.buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        trace!("Write completed");
    }

    /// データを GPU から CPU へコピー
    pub fn read_data<T: Copy>(&self, data: &mut [T]) {
        trace!(
            "Reading {} elements from Metal buffer (shape={:?})",
            data.len(),
            self.shape
        );
        let ptr = self.buffer.contents() as *const T;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data.len());
        }
        trace!("Read completed");
    }
}

impl Buffer for MetalBuffer {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

/// Metal カーネル（コンパイル済み関数）
pub struct MetalKernel {
    pipeline_state: ComputePipelineState,
    command_queue: CommandQueue,
    thread_group_size: MTLSize,
    grid_size: MTLSize,
}

impl MetalKernel {
    /// 新しいカーネルを作成
    pub fn new(
        pipeline_state: ComputePipelineState,
        command_queue: CommandQueue,
        grid_size: MTLSize,
    ) -> Self {
        // スレッドグループサイズを自動決定（最大スレッド数を使用）
        let max_threads = pipeline_state.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(DEFAULT_MAX_THREAD_GROUP_SIZE), 1, 1);

        Self {
            pipeline_state,
            command_queue,
            thread_group_size,
            grid_size,
        }
    }

    /// カーネルを実行
    pub fn dispatch(&self, buffers: &[&MetalBuffer]) -> Result<(), String> {
        info!(
            "Dispatching Metal kernel with {} buffers, grid_size=({},{},{}), thread_group_size=({},{},{})",
            buffers.len(),
            self.grid_size.width,
            self.grid_size.height,
            self.grid_size.depth,
            self.thread_group_size.width,
            self.thread_group_size.height,
            self.thread_group_size.depth
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_state);

        // バッファをバインド
        for (index, buffer) in buffers.iter().enumerate() {
            trace!(
                "Binding buffer {} (shape={:?}, size={})",
                index,
                buffer.shape(),
                buffer.byte_size()
            );
            encoder.set_buffer(index as u64, Some(buffer.inner()), 0);
        }

        // スレッドグリッドサイズとスレッドグループサイズを設定
        encoder.dispatch_thread_groups(self.grid_size, self.thread_group_size);
        encoder.end_encoding();

        debug!("Committing Metal command buffer");
        command_buffer.commit();
        command_buffer.wait_until_completed();
        debug!("Metal kernel execution completed");

        Ok(())
    }

    /// グリッドサイズを設定
    pub fn set_grid_size(&mut self, width: u64, height: u64, depth: u64) {
        self.grid_size = MTLSize::new(width, height, depth);
    }

    /// スレッドグループサイズを設定
    pub fn set_thread_group_size(&mut self, width: u64, height: u64, depth: u64) {
        self.thread_group_size = MTLSize::new(width, height, depth);
    }
}

impl Kernel for MetalKernel {
    type Buffer = MetalBuffer;

    fn signature(&self) -> KernelSignature {
        KernelSignature {}
    }
}

/// Metal コンパイラ
pub struct MetalCompiler {
    device: Arc<Device>,
    command_queue: CommandQueue,
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

        info!("Metal kernel compiled successfully");
        MetalKernel::new(pipeline_state, self.command_queue.clone(), grid_size)
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

    #[test]
    fn test_buffer_creation() {
        if let Some(compiler) = MetalCompiler::with_default_device() {
            let buffer = compiler.create_buffer(vec![10, 20], 4);
            assert_eq!(buffer.shape(), vec![10, 20]);
            assert_eq!(buffer.byte_size(), 10 * 20 * 4);
        }
    }

    #[test]
    fn test_buffer_read_write() {
        if let Some(compiler) = MetalCompiler::with_default_device() {
            let mut buffer = compiler.create_buffer(vec![10], 4);

            // データを書き込み
            let write_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
            buffer.write_data(&write_data);

            // データを読み出し
            let mut read_data = vec![0.0f32; 10];
            buffer.read_data(&mut read_data);

            // 確認
            assert_eq!(write_data, read_data);
        }
    }

    #[test]
    fn test_simple_kernel_compile() {
        if let Some(mut compiler) = MetalCompiler::with_default_device() {
            // 簡単なカーネルをコンパイル
            let source = r#"
                #include <metal_stdlib>
                using namespace metal;

                kernel void test_kernel(
                    device float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]
                ) {
                    output[tid] = input[tid] * 2.0f;
                }
            "#;

            let code = MetalCode::new(source.to_string());
            let _kernel = compiler.compile(&code);
            // コンパイルが成功すれば OK
        }
    }

    #[test]
    fn test_kernel_execution() {
        if let Some(mut compiler) = MetalCompiler::with_default_device() {
            // カーネルをコンパイル
            let source = r#"
                #include <metal_stdlib>
                using namespace metal;

                kernel void double_kernel(
                    device float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]
                ) {
                    output[tid] = input[tid] * 2.0f;
                }
            "#;

            let code = MetalCode::new(source.to_string());
            let mut kernel = compiler.compile(&code);

            // バッファを作成
            let mut input_buffer = compiler.create_buffer(vec![10], 4);
            let output_buffer = compiler.create_buffer(vec![10], 4);

            // 入力データを設定
            let input_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
            input_buffer.write_data(&input_data);

            // グリッドサイズを設定
            kernel.set_grid_size(10, 1, 1);

            // カーネルを実行
            kernel.dispatch(&[&input_buffer, &output_buffer]).unwrap();

            // 結果を読み出し
            let mut output_data = vec![0.0f32; 10];
            output_buffer.read_data(&mut output_data);

            // 確認
            let expected: Vec<f32> = input_data.iter().map(|&x| x * 2.0).collect();
            assert_eq!(output_data, expected);
        }
    }
}
