use crate::backend::metal::MetalBuffer;
use crate::backend::{Buffer, Kernel, KernelSignature};
use log::{debug, info, trace};
use metal::{CommandQueue, ComputePipelineState, MTLSize};

/// デフォルトのスレッドグループサイズの最大値
const DEFAULT_MAX_THREAD_GROUP_SIZE: u64 = 256;

/// Metal カーネル（コンパイル済み関数）
pub struct MetalKernel {
    pipeline_state: ComputePipelineState,
    command_queue: CommandQueue,
    thread_group_size: MTLSize,
    grid_size: MTLSize,
    signature: KernelSignature,
}

impl MetalKernel {
    /// 新しいカーネルを作成
    pub fn new(
        pipeline_state: ComputePipelineState,
        command_queue: CommandQueue,
        grid_size: MTLSize,
        signature: KernelSignature,
    ) -> Self {
        // スレッドグループサイズを自動決定（最大スレッド数を使用）
        let max_threads = pipeline_state.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(DEFAULT_MAX_THREAD_GROUP_SIZE), 1, 1);

        Self {
            pipeline_state,
            command_queue,
            thread_group_size,
            grid_size,
            signature,
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
        self.signature.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::Compiler;
    use crate::backend::metal::{MetalCode, MetalCompiler};

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
