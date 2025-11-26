//! Cast演算のlowering
//!
//! テンソルの要素を別の型にキャストする演算をカーネル関数に変換します。

use crate::ast::{AstNode, Scope, helper::*};
use crate::graph::{DType as GraphDType, GraphNode};

use super::Lowerer;

impl Lowerer {
    /// Cast演算をカーネル関数に変換
    ///
    /// 入力バッファの各要素をターゲット型にキャストして出力バッファに書き込む
    ///
    /// # 引数
    /// - `node`: Castノード
    /// - `target_dtype`: ターゲット型
    /// - `node_id`: カーネル関数のID
    pub(super) fn lower_cast_kernel(
        &mut self,
        node: &GraphNode,
        target_dtype: &GraphDType,
        node_id: usize,
    ) -> Result<AstNode, String> {
        let input_node = &node.src[0];
        let input_shape = input_node.view.shape();
        let ndim = input_shape.len();

        // パラメータを生成
        let mut params = Vec::new();
        params.push(self.create_input_param(0, &input_node.dtype)?);
        params.push(self.create_output_param(target_dtype)?);
        params.extend(self.extract_shape_params(input_shape));

        // ループ本体の生成
        let body_statements = self.generate_cast_loops(node, target_dtype, ndim)?;

        // カーネル関数を作成
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// Cast演算のループを生成
    fn generate_cast_loops(
        &mut self,
        node: &GraphNode,
        target_dtype: &GraphDType,
        ndim: usize,
    ) -> Result<Vec<AstNode>, String> {
        let input_node = &node.src[0];
        let input_shape = input_node.view.shape();

        // ターゲット型を取得
        let ast_target_dtype = self.graph_dtype_to_ast(target_dtype)?;
        let ast_input_dtype = self.graph_dtype_to_ast(&input_node.dtype)?;

        if ndim == 0 {
            // スカラーの場合
            let input_offset = self.compute_offset_from_view(input_node, &[]);
            let output_offset = self.compute_offset_from_view(node, &[]);

            let value = load(var("input0"), input_offset, ast_input_dtype);
            let casted_value = cast(value, ast_target_dtype);
            return Ok(vec![store(var("output"), output_offset, casted_value)]);
        }

        // 軸インデックスのリスト
        let axes: Vec<usize> = (0..ndim).collect();

        // 最内側のループ本体: output[i] = (target_type)input0[i]
        let input_offset = self.compute_offset_from_view(input_node, &axes);
        let output_offset = self.compute_offset_from_view(node, &axes);

        let value = load(var("input0"), input_offset, ast_input_dtype);
        let casted_value = cast(value, ast_target_dtype);
        let body_statements = vec![store(var("output"), output_offset, casted_value)];

        // ネストされたループを生成（外側から内側へ）
        let mut current = block(body_statements, Scope::new());

        for axis in (0..ndim).rev() {
            let loop_var = format!("ridx{}", axis);
            let shape_expr: AstNode = input_shape[axis].clone().into();
            current = range(&loop_var, const_int(0), const_int(1), shape_expr, current);
        }

        Ok(vec![current])
    }
}
