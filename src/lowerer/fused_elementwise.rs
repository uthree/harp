use crate::ast::{AstNode, Literal, Mutability, Scope, helper::*};
use crate::graph::{
    GraphNode,
    ops::{FusedElementwiseOp, FusedInput},
};
use log::debug;

use super::Lowerer;

impl Lowerer {
    /// FusedElementwise演算をカーネル関数に変換
    pub(super) fn lower_fused_elementwise_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        ops: &[FusedElementwiseOp],
    ) -> Result<AstNode, String> {
        debug!(
            "Lowering fused elementwise operation with {} ops",
            ops.len()
        );
        debug!("View: {:?}", node.view);

        let shape = node.view.shape();
        let ndim = shape.len();

        // パラメータを生成: 入力バッファー、出力バッファー、shape変数
        let mut params = Vec::new();

        // 入力バッファー（srcノード）
        for (i, src) in node.src.iter().enumerate() {
            params.push(self.create_input_param(i, &src.dtype)?);
        }

        // 出力バッファーとShape変数
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(shape));

        // ループ本体の生成
        let mut scope = Scope::new();
        let body_statements = self.generate_fused_elementwise_loops(node, ops, ndim, &mut scope)?;

        // カーネル関数を作成して返す
        Ok(self.create_kernel_function(node_id, params, body_statements, scope))
    }

    /// FusedElementwise演算のループを生成
    fn generate_fused_elementwise_loops(
        &mut self,
        node: &GraphNode,
        ops: &[FusedElementwiseOp],
        ndim: usize,
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let shape = node.view.shape();

        if ndim == 0 {
            // スカラー演算（ループなし）
            return self.generate_fused_elementwise_body(node, ops, &[], scope);
        }

        // ネストしたループを生成（外側から内側へ）
        let mut body_statements =
            self.generate_fused_elementwise_body(node, ops, &(0..ndim).collect::<Vec<_>>(), scope)?;

        // ループを逆順に作成（内側から外側へ）
        for axis in (0..ndim).rev() {
            let loop_var = format!("ridx{}", axis);
            // shapeから直接AstNodeに変換
            let shape_expr: AstNode = shape[axis].clone().into();
            let elementwise_strategy = &node.elementwise_strategies[axis];
            let unroll_factor = elementwise_strategy.unroll_factor();

            if unroll_factor > 1 {
                // ループアンローリングを適用
                body_statements = self.generate_unrolled_loop_with_shape(
                    axis,
                    unroll_factor,
                    &shape_expr,
                    body_statements,
                )?;
            } else {
                // 通常のループ
                let loop_body = AstNode::Block {
                    statements: body_statements,
                    scope: Box::new(scope.clone()),
                };

                // 外側のループ用に新しいスコープを作成
                *scope = Scope::new();

                body_statements = vec![AstNode::Range {
                    var: loop_var.clone(),
                    start: Box::new(AstNode::Const(Literal::Int(0))),
                    step: Box::new(AstNode::Const(Literal::Int(1))),
                    stop: Box::new(shape_expr),
                    body: Box::new(loop_body),
                }];
            }
        }

        Ok(body_statements)
    }

    /// FusedElementwise演算の本体を生成（ループ内部の処理）
    fn generate_fused_elementwise_body(
        &mut self,
        node: &GraphNode,
        ops: &[FusedElementwiseOp],
        axes: &[usize],
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();

        // 最内側の軸のSIMD幅を取得（最後の軸を使用）
        let simd_width = if let Some(&last_axis) = axes.last() {
            node.elementwise_strategies[last_axis].simd_width()
        } else {
            1
        };

        // 全ての入力をロード
        let mut graph_inputs = Vec::new();
        for (i, src) in node.src.iter().enumerate() {
            let alu_var = self.fresh_alu();
            let input_ptr = var(format!("input{}", i));

            // 各srcノードのViewからオフセットを計算
            let offset = self.compute_offset_from_view(src, axes);
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();

            // SIMD化: simd_width > 1の場合はベクトルロード
            let (load_node, final_dtype) = if simd_width > 1 {
                let vec_dtype = src_dtype.to_vec(simd_width);
                (
                    load_vec(input_ptr, offset, simd_width, vec_dtype.clone()),
                    vec_dtype,
                )
            } else {
                (load(input_ptr, offset, src_dtype.clone()), src_dtype)
            };

            // 変数を宣言
            scope.declare(alu_var.clone(), final_dtype, Mutability::Mutable)?;

            // 初期値を代入
            statements.push(assign(&alu_var, load_node));
            graph_inputs.push(alu_var);
        }

        // 中間結果をAstNodeとして保存する配列（変数名ではなく式そのもの）
        let mut intermediate_results: Vec<AstNode> = Vec::new();

        // ops配列を順に評価して、最終的な式を構築
        for fused_op in ops {
            // この演算の入力を取得
            let mut operands = Vec::new();
            for input in &fused_op.inputs {
                let operand = match input {
                    FusedInput::GraphInput(idx) => {
                        // GraphNodeのsrc[i]から読み込んだ値
                        var(&graph_inputs[*idx])
                    }
                    FusedInput::IntermediateResult(idx) => {
                        // ops[i]の中間結果（式として保持）
                        intermediate_results[*idx].clone()
                    }
                    FusedInput::Const(lit) => {
                        // 定数値を直接埋め込む
                        AstNode::Const(lit.clone())
                    }
                };
                operands.push(operand);
            }

            // 演算を適用（結果は式として保存）
            let result = self.apply_elementwise_op(&fused_op.op, &operands)?;
            intermediate_results.push(result);
        }

        // 最後の演算結果を出力にストア（1つの式として）
        if let Some(final_result) = intermediate_results.last() {
            let output_ptr = var("output");
            let output_offset = self.compute_offset_from_view(node, axes);
            statements.push(store(output_ptr, output_offset, final_result.clone()));
        } else {
            return Err("FusedElementwise requires at least one operation".to_string());
        }

        Ok(statements)
    }
}
