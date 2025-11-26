//! Concat演算のlowering
//!
//! 複数のテンソルを指定した軸で結合する演算をカーネル関数に変換します。

use crate::ast::{AstNode, Mutability, Scope, helper::*};
use crate::graph::GraphNode;

use super::Lowerer;

impl Lowerer {
    /// Concat演算をカーネル関数に変換
    ///
    /// アルゴリズム:
    /// 各入力テンソルを順番に出力テンソルにコピーする。
    /// 結合軸のオフセットを累積しながら各入力を処理。
    ///
    /// # 引数
    /// - `node`: Concatノード
    /// - `node_id`: カーネル関数のID
    /// - `axis`: 結合する軸
    pub(super) fn lower_concat_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        axis: usize,
    ) -> Result<AstNode, String> {
        let output_view = &node.view;
        let output_shape = output_view.shape();
        let ndim = output_view.ndim();

        if node.src.is_empty() {
            return Err("Concat requires at least one input".to_string());
        }

        // パラメータを生成
        let mut params = Vec::new();

        // 全ての入力バッファをパラメータとして追加
        for (i, input) in node.src.iter().enumerate() {
            params.push(self.create_input_param(i, &input.dtype)?);
        }

        // 出力バッファ
        params.push(self.create_output_param(&node.dtype)?);

        // Shape変数を収集（出力のshapeから）
        params.extend(self.extract_shape_params(output_shape));

        // 各入力のshapeからも変数を収集
        for input in &node.src {
            let input_shape = input.view.shape();
            for param in self.extract_shape_params(input_shape) {
                // 重複を避ける
                if !params.iter().any(|p| p.name == param.name) {
                    params.push(param);
                }
            }
        }

        if ndim == 0 {
            // スカラーの場合（0次元テンソル）
            // 最初の入力をそのままコピー
            let input_offset = self.compute_offset_from_view(&node.src[0], &[]);
            let output_offset = self.compute_offset_from_view(node, &[]);
            let input_dtype = self.graph_dtype_to_ast(&node.src[0].dtype)?;

            let body = store(
                var("output"),
                output_offset,
                load(var("input0"), input_offset, input_dtype),
            );

            return Ok(self.create_kernel_function(node_id, params, vec![body], Scope::new()));
        }

        // 各入力を順番に処理するための文を生成
        let mut all_statements: Vec<AstNode> = Vec::new();
        let mut outer_scope = Scope::new();

        // 結合軸のオフセットを追跡する変数を宣言
        let offset_var = "concat_offset";
        outer_scope
            .declare(
                offset_var.to_string(),
                crate::ast::DType::Int,
                Mutability::Mutable,
            )
            .map_err(|e| format!("Failed to declare offset variable: {:?}", e))?;

        // オフセットを0で初期化
        all_statements.push(assign(offset_var, const_int(0)));

        // 各入力テンソルを処理
        for (input_idx, input_node) in node.src.iter().enumerate() {
            let input_shape = input_node.view.shape();
            let input_name = format!("input{}", input_idx);

            // 入力テンソルのループを生成
            let axes: Vec<usize> = (0..ndim).collect();

            // 一時変数
            let alu_var = self.fresh_alu();
            let input_dtype = self.graph_dtype_to_ast(&input_node.dtype)?;

            let mut inner_scope = Scope::new();
            inner_scope
                .declare(alu_var.clone(), input_dtype.clone(), Mutability::Mutable)
                .map_err(|e| format!("Failed to declare variable: {:?}", e))?;

            // 入力オフセット計算
            let input_offset = self.compute_offset_from_view(input_node, &axes);

            // 出力オフセット計算
            // 結合軸のインデックスにオフセットを加算
            let output_indices: Vec<AstNode> = axes
                .iter()
                .map(|&ax| {
                    let ridx = var(format!("ridx{}", ax));
                    if ax == axis {
                        ridx + var(offset_var)
                    } else {
                        ridx
                    }
                })
                .collect();

            let output_offset = self.compute_output_offset_from_indices(node, &output_indices);

            // ループ本体：入力から読み込んで出力に書き込む
            let body_statements = vec![
                assign(
                    &alu_var,
                    load(var(&input_name), input_offset, input_dtype.clone()),
                ),
                store(var("output"), output_offset, var(&alu_var)),
            ];

            // ネストされたループを生成（入力のshapeでループ）
            let (loop_statements, _) =
                self.generate_nested_loops(ndim, input_shape, "ridx", body_statements, inner_scope);

            all_statements.extend(loop_statements);

            // オフセットを更新（この入力の結合軸サイズを加算）
            let axis_size: AstNode = input_shape[axis].clone().into();
            all_statements.push(assign(offset_var, var(offset_var) + axis_size));
        }

        Ok(self.create_kernel_function(node_id, params, all_statements, outer_scope))
    }
}
