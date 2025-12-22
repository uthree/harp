//! バックエンド別評価実装
//!
//! eval_opencl, eval_metal メソッドを提供します。
//! 各バックエンドはフィーチャーフラグで切り替えられます。

use super::{ArrayElement, ArrayError, ArrayState, Buffer, LazyArray};
use crate::dim::Dimension;
use harp_core::graph::Graph;

// ============================================================================
// OpenCL 評価実装
// ============================================================================

#[cfg(feature = "opencl")]
impl<T: ArrayElement, D: Dimension> LazyArray<T, D> {
    /// OpenCLで評価を実行
    pub(crate) fn eval_opencl(&self) -> Result<(), ArrayError> {
        use crate::execution::opencl::with_opencl_context;
        use std::collections::HashMap;

        let node = self.graph_node();

        // 1. Graphを構築
        let mut graph = Graph::new();
        graph.output("result", node.clone());

        // 2. コンパイル & バッファ確保 & 実行
        let buffer = with_opencl_context(|ctx| {
            // コンパイル（複数カーネル対応）
            let compiled = ctx.compile_program(graph)?;

            // 出力バッファを確保
            let mut output_buf = ctx.allocate_buffer(self.shape.clone(), T::buffer_dtype())?;

            // 入力バッファのマップ（今回は外部入力なし）
            let inputs: HashMap<String, &harp_backend_opencl::OpenCLBuffer> = HashMap::new();

            // 出力バッファのマップ
            let mut outputs: HashMap<String, &mut harp_backend_opencl::OpenCLBuffer> =
                HashMap::new();
            outputs.insert("result".to_string(), &mut output_buf);

            // 実行
            compiled
                .execute(ctx.device(), &inputs, &mut outputs)
                .map_err(|e| ArrayError::Execution(e.to_string()))?;

            Ok(output_buf)
        })?;

        // 3. 状態をMaterializedに遷移
        *self.state.borrow_mut() = ArrayState::Materialized {
            node,
            buffer: Buffer::OpenCL(buffer),
        };

        Ok(())
    }
}

// ============================================================================
// Metal 評価実装
// ============================================================================

#[cfg(feature = "metal")]
impl<T: ArrayElement, D: Dimension> LazyArray<T, D> {
    /// Metalで評価を実行
    pub(crate) fn eval_metal(&self) -> Result<(), ArrayError> {
        use crate::execution::metal::with_metal_context;

        let node = self.graph_node();

        // 1. Graphを構築
        let mut graph = Graph::new();
        graph.output("result", node.clone());

        // 2. コンパイル & バッファ確保 & 実行
        let buffer = with_metal_context(|ctx| {
            // コンパイル
            let compiled = ctx.compile(graph)?;

            // 出力バッファを確保
            let mut output_buf = ctx.allocate_buffer(self.shape.clone(), T::buffer_dtype())?;

            // 実行（入力なし、出力のみ）
            compiled
                .execute(&[], &mut [&mut output_buf])
                .map_err(|e| ArrayError::Execution(e.to_string()))?;

            Ok(output_buf)
        })?;

        // 3. 状態をMaterializedに遷移
        *self.state.borrow_mut() = ArrayState::Materialized {
            node,
            buffer: Buffer::Metal(buffer),
        };

        Ok(())
    }
}
