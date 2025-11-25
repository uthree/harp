//! RandInit演算のlowering
//!
//! 一様乱数でテンソルを初期化する演算をカーネル関数に変換します。

use crate::ast::{AstNode, Scope, helper::*};
use crate::graph::GraphNode;

use super::Lowerer;

impl Lowerer {
    /// RandInit演算をカーネル関数に変換
    ///
    /// アルゴリズム:
    /// 出力バッファの各要素に0〜1の一様乱数を書き込む
    ///
    /// # 引数
    /// - `node`: RandInitノード
    /// - `node_id`: カーネル関数のID
    pub(super) fn lower_rand_init_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<AstNode, String> {
        let output_view = &node.view;
        let output_shape = output_view.shape();
        let ndim = output_view.ndim();

        // パラメータを生成（出力バッファとShape変数のみ）
        let mut params = Vec::new();
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(output_shape));

        // ループ本体の生成
        let body_statements = self.generate_rand_init_loops(node, ndim)?;

        // カーネル関数を作成
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// 乱数初期化ループを生成
    ///
    /// 出力バッファの各要素に乱数を書き込む
    fn generate_rand_init_loops(
        &mut self,
        node: &GraphNode,
        ndim: usize,
    ) -> Result<Vec<AstNode>, String> {
        let output_view = &node.view;
        let output_shape = output_view.shape();

        if ndim == 0 {
            // スカラーの場合
            let output_offset = self.compute_offset_from_view(node, &[]);
            return Ok(vec![store(var("output"), output_offset, rand())]);
        }

        let axes: Vec<usize> = (0..ndim).collect();

        // 最内側のループ本体: output[offset] = rand()
        let output_offset = self.compute_offset_from_view(node, &axes);
        let body_statements = vec![store(var("output"), output_offset, rand())];

        // ネストされたループを生成
        let (loop_statements, _) =
            self.generate_nested_loops(ndim, output_shape, "ridx", body_statements, Scope::new());

        Ok(loop_statements)
    }
}
