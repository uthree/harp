//! Arange演算のlowering
//!
//! 連番テンソル [0, 1, 2, ..., n-1] を生成する演算をカーネル関数に変換します。

use crate::ast::{AstNode, DType as AstDType, Scope, helper::*};
use crate::graph::GraphNode;

use super::Lowerer;

impl Lowerer {
    /// Arange演算をカーネル関数に変換
    ///
    /// 出力バッファの各要素に連番値（0, 1, 2, ...）を書き込む
    ///
    /// # 引数
    /// - `node`: Arangeノード
    /// - `node_id`: カーネル関数のID
    pub(super) fn lower_arange_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<AstNode, String> {
        let output_view = &node.view;
        let output_shape = output_view.shape();

        // Arangeは1次元のみサポート
        if output_shape.len() != 1 {
            return Err(format!(
                "Arange only supports 1D tensors, got {}D",
                output_shape.len()
            ));
        }

        // パラメータを生成（出力バッファとShape変数のみ）
        let mut params = Vec::new();
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(output_shape));

        // ループ本体の生成
        let body_statements = self.generate_arange_loop(node)?;

        // カーネル関数を作成
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// 連番初期化ループを生成
    ///
    /// 出力バッファの各要素にインデックス値を書き込む: output[i] = (float)i
    fn generate_arange_loop(&mut self, node: &GraphNode) -> Result<Vec<AstNode>, String> {
        let output_view = &node.view;
        let output_shape = output_view.shape();

        // ループ変数
        let loop_var = "ridx0";

        // 最内側のループ本体: output[i] = (float)i
        let output_offset = self.compute_offset_from_view(node, &[0]);

        // ループ変数をfloatにキャスト
        let index_as_float = cast(var(loop_var), AstDType::F32);

        let body_statements = vec![store(var("output"), output_offset, index_as_float)];

        // ループを生成
        let shape_expr: AstNode = output_shape[0].clone().into();
        let loop_stmt = range(
            loop_var,
            const_int(0),
            const_int(1),
            shape_expr,
            block(body_statements, Scope::new()),
        );

        Ok(vec![loop_stmt])
    }
}
