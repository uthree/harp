//! Slice演算のlowering
//!
//! テンソルの一部を切り出す演算をカーネル関数に変換します。

use crate::ast::{AstNode, Mutability, Scope, helper::*};
use crate::graph::GraphNode;

use super::Lowerer;

impl Lowerer {
    /// Slice演算をカーネル関数に変換
    ///
    /// アルゴリズム:
    /// 入力テンソルの指定範囲を出力テンソルにコピー
    ///
    /// # 引数
    /// - `node`: Sliceノード
    /// - `node_id`: カーネル関数のID
    /// - `ranges`: 各軸の(start, end)範囲
    pub(super) fn lower_slice_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        ranges: &[(usize, usize)],
    ) -> Result<AstNode, String> {
        let input_node = &node.src[0];
        let output_view = &node.view;
        let output_shape = output_view.shape();
        let ndim = output_view.ndim();

        assert_eq!(ranges.len(), ndim, "ranges length must match ndim");

        // パラメータを生成
        let mut params = Vec::new();
        params.push(self.create_input_param(0, &input_node.dtype)?);
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(output_shape));

        if ndim == 0 {
            // スカラーの場合（そのままコピー）
            let input_offset = self.compute_offset_from_view(input_node, &[]);
            let output_offset = self.compute_offset_from_view(node, &[]);
            let input_dtype = self.graph_dtype_to_ast(&input_node.dtype)?;

            let body = store(
                var("output"),
                output_offset,
                load(var("input0"), input_offset, input_dtype),
            );

            return Ok(self.create_kernel_function(node_id, params, vec![body], Scope::new()));
        }

        let mut scope = Scope::new();
        let axes: Vec<usize> = (0..ndim).collect();

        // 出力インデックスからロード
        let output_offset = self.compute_offset_from_view(node, &axes);

        // 入力インデックスを計算（出力インデックス + start）
        let input_axes: Vec<AstNode> = axes
            .iter()
            .zip(ranges.iter())
            .map(|(axis, (start, _))| {
                let ridx = var(format!("ridx{}", axis));
                if *start > 0 {
                    ridx + const_int(*start as isize)
                } else {
                    ridx
                }
            })
            .collect();

        let input_offset = self.compute_input_offset_from_indices(input_node, &input_axes);

        // データ読み込み
        let alu_var = self.fresh_alu();
        let input_dtype = self.graph_dtype_to_ast(&input_node.dtype)?;

        scope
            .declare(alu_var.clone(), input_dtype.clone(), Mutability::Mutable)
            .map_err(|e| format!("Failed to declare variable: {:?}", e))?;

        let body_statements = vec![
            assign(&alu_var, load(var("input0"), input_offset, input_dtype)),
            store(var("output"), output_offset, var(&alu_var)),
        ];

        // ネストされたループを生成（出力のshapeでループ）
        let (loop_statements, _) =
            self.generate_nested_loops(ndim, output_shape, "ridx", body_statements, scope);

        Ok(self.create_kernel_function(node_id, params, loop_statements, Scope::new()))
    }

    /// インデックス式からオフセットを計算（入力用）
    fn compute_input_offset_from_indices(&self, node: &GraphNode, indices: &[AstNode]) -> AstNode {
        use crate::graph::shape::View;

        match &node.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for (i, idx) in indices.iter().enumerate() {
                    let stride: AstNode = strides[i].clone().into();
                    result = result + idx.clone() * stride;
                }

                result
            }
        }
    }
}
