use crate::ast::{AstNode, Mutability, Scope, helper::*};
use crate::graph::GraphNode;
use log::debug;

use super::Lowerer;

impl Lowerer {
    /// Contiguous演算をカーネル関数に変換
    pub(super) fn lower_contiguous_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<AstNode, String> {
        debug!("Lowering contiguous operation");
        debug!("Input view: {:?}", node.src[0].view);
        debug!("Output view: {:?}", node.view);

        if node.src.is_empty() {
            return Err("Contiguous operation requires at least one input".to_string());
        }

        let input = &node.src[0];
        let shape = node.view.shape();
        let ndim = shape.len();

        // パラメータを生成: 入力バッファー、出力バッファー、shape変数
        let mut params = Vec::new();
        params.push(self.create_input_param(0, &input.dtype)?);
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(shape));

        // ループ本体の生成
        let body_statements = self.generate_contiguous_loops(node, ndim)?;

        // カーネル関数を作成して返す
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// Contiguous演算のループを生成
    pub(super) fn generate_contiguous_loops(
        &mut self,
        node: &GraphNode,
        ndim: usize,
    ) -> Result<Vec<AstNode>, String> {
        let mut scope = Scope::new();
        let shape = node.view.shape();

        if ndim == 0 {
            // スカラーの場合（ループなし）
            return self.generate_contiguous_body(node, &[], &mut scope);
        }

        // 最内側のループ本体を生成
        let body_statements =
            self.generate_contiguous_body(node, &(0..ndim).collect::<Vec<_>>(), &mut scope)?;

        // ネストしたループを生成（共通関数を使用）
        let (loop_statements, _) =
            self.generate_nested_loops(ndim, shape, "ridx", body_statements, scope);

        Ok(loop_statements)
    }

    /// Contiguous演算の本体を生成（ループ内部の処理）
    pub(super) fn generate_contiguous_body(
        &mut self,
        node: &GraphNode,
        axes: &[usize],
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();

        let input = &node.src[0];

        // 入力からロード（入力のViewを考慮）
        let input_ptr = var("input0");
        let input_offset = self.compute_offset_from_view(input, axes);
        let alu_var = self.fresh_alu();
        let input_ptr_dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
        let input_dtype = input_ptr_dtype.deref_type().clone();

        // 変数を宣言
        scope.declare(alu_var.clone(), input_dtype.clone(), Mutability::Mutable)?;

        // 初期値を代入
        statements.push(assign(&alu_var, load(input_ptr, input_offset, input_dtype)));

        // 出力にストア（出力のViewを考慮）
        let output_ptr = var("output");
        let output_offset = self.compute_offset_from_view(node, axes);
        statements.push(store(output_ptr, output_offset, var(&alu_var)));

        Ok(statements)
    }
}
