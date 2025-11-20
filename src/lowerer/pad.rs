//! Pad演算のlowering
//!
//! テンソルにパディングを追加する演算をカーネル関数に変換します。

use crate::ast::{AstNode, Mutability, Scope, helper::*};
use crate::graph::GraphNode;

use super::Lowerer;

impl Lowerer {
    /// Pad演算をカーネル関数に変換
    ///
    /// アルゴリズム:
    /// 1. 出力バッファ全体をpadding値で初期化
    /// 2. 入力データを適切な位置にコピー
    ///
    /// # 引数
    /// - `node`: Padノード
    /// - `node_id`: カーネル関数のID
    /// - `padding`: 各軸の(前, 後)パディング量
    /// - `value`: パディング値
    pub(super) fn lower_pad_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        padding: &[(usize, usize)],
        value: f32,
    ) -> Result<AstNode, String> {
        let input_node = &node.src[0];
        let output_view = &node.view;
        let output_shape = output_view.shape();

        let ndim = output_view.ndim();
        assert_eq!(padding.len(), ndim, "padding length must match ndim");

        // パラメータを生成
        let mut params = Vec::new();
        params.push(self.create_input_param(0, &input_node.dtype)?);
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(output_shape));

        let mut body_statements = Vec::new();

        // === ステップ1: 出力バッファを初期化 ===
        // 全要素をpadding値に設定
        let init_statements = self.generate_pad_init_loops(node, padding, value)?;
        body_statements.extend(init_statements);

        // === ステップ2: 入力データをコピー ===
        // 入力範囲をループして適切な位置にコピー
        let copy_statements = self.generate_pad_copy_loops(node, padding)?;
        body_statements.extend(copy_statements);

        // カーネル関数を作成
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// パディング初期化ループを生成
    ///
    /// 出力バッファ全体をpadding値で初期化
    fn generate_pad_init_loops(
        &mut self,
        node: &GraphNode,
        _padding: &[(usize, usize)],
        value: f32,
    ) -> Result<Vec<AstNode>, String> {
        let output_view = &node.view;
        let output_shape = output_view.shape();
        let ndim = output_view.ndim();

        if ndim == 0 {
            // スカラーの場合
            let output_offset = self.compute_offset_from_view(node, &[]);
            return Ok(vec![store(var("output"), output_offset, const_f32(value))]);
        }

        let axes: Vec<usize> = (0..ndim).collect();

        // 最内側のループ本体: output[offset] = value
        let output_offset = self.compute_offset_from_view(node, &axes);
        let body_statements = vec![store(var("output"), output_offset, const_f32(value))];

        // ネストされたループを生成
        let (loop_statements, _) =
            self.generate_nested_loops(ndim, output_shape, "ridx", body_statements, Scope::new());

        Ok(loop_statements)
    }

    /// パディングコピーループを生成
    ///
    /// 入力データを適切な位置にコピー
    fn generate_pad_copy_loops(
        &mut self,
        node: &GraphNode,
        padding: &[(usize, usize)],
    ) -> Result<Vec<AstNode>, String> {
        let input_node = &node.src[0];
        let input_view = &input_node.view;
        let input_shape = input_view.shape();
        let ndim = input_view.ndim();

        if ndim == 0 {
            // スカラーの場合（既に初期化で処理済み）
            return Ok(vec![]);
        }

        let mut scope = Scope::new();
        let axes: Vec<usize> = (0..ndim).collect();

        // 入力からロード
        let input_offset = self.compute_offset_from_view(input_node, &axes);
        let alu_var = self.fresh_alu();
        let input_dtype = self.graph_dtype_to_ast(&input_node.dtype)?;

        // 変数を宣言
        scope
            .declare(alu_var.clone(), input_dtype.clone(), Mutability::Mutable)
            .map_err(|e| format!("Failed to declare variable: {:?}", e))?;

        let mut body_statements = Vec::new();

        // 入力から読み込み
        body_statements.push(assign(
            &alu_var,
            load(var("input0"), input_offset, input_dtype),
        ));

        // 出力位置を計算（入力インデックス + padding_before）
        // 出力インデックスを計算するために、一時的に出力用のインデックスを使用
        // ridx0, ridx1, ... が入力インデックス
        // 出力インデックス = ridx_i + padding_before[i]
        let output_axes: Vec<AstNode> = axes
            .iter()
            .zip(padding.iter())
            .map(|(axis, (before, _))| {
                let ridx = var(format!("ridx{}", axis));
                if *before > 0 {
                    ridx + const_int(*before as isize)
                } else {
                    ridx
                }
            })
            .collect();

        // 出力のオフセットを手動で計算
        let output_offset = self.compute_output_offset_from_indices(node, &output_axes);

        // 出力に書き込み
        body_statements.push(store(var("output"), output_offset, var(&alu_var)));

        // ネストされたループを生成（入力のshapeでループ）
        let (loop_statements, _) =
            self.generate_nested_loops(ndim, input_shape, "ridx", body_statements, scope);

        Ok(loop_statements)
    }

    /// インデックス式からオフセットを計算
    fn compute_output_offset_from_indices(&self, node: &GraphNode, indices: &[AstNode]) -> AstNode {
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
