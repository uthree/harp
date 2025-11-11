use crate::ast::{
    AccessRegion, AstNode, DType as AstDType, FunctionKind, Literal, Mutability, Scope, VarDecl,
    VarKind, helper::*,
};
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
            let dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            params.push(VarDecl {
                name: format!("input{}", i),
                dtype,
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
                initial_value: None,
            });
        }

        // 出力バッファー
        let output_dtype = self.graph_dtype_to_ast_ptr(&node.dtype)?;
        params.push(VarDecl {
            name: "output".to_string(),
            dtype: output_dtype,
            mutability: Mutability::Mutable,
            region: AccessRegion::Shared,
            kind: VarKind::Normal,
            initial_value: None,
        });

        // Shape変数（各軸のサイズ）
        for i in 0..ndim {
            params.push(VarDecl {
                name: format!("shape{}", i),
                dtype: AstDType::Usize,
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
                initial_value: None,
            });
        }

        // ループ本体の生成
        let mut scope = Scope::new();
        let body_statements = self.generate_fused_elementwise_loops(node, ops, ndim, &mut scope)?;

        // カーネル関数のbodyを作成（Blockノード）
        let body = AstNode::Block {
            statements: body_statements,
            scope: Box::new(scope),
        };

        // カーネル関数名
        let function_name = format!("kernel_{}", node_id);

        // AstNode::Functionとして返す
        Ok(function(
            Some(function_name),
            FunctionKind::Normal,
            params,
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// FusedElementwise演算のループを生成
    fn generate_fused_elementwise_loops(
        &mut self,
        node: &GraphNode,
        ops: &[FusedElementwiseOp],
        ndim: usize,
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
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
            let shape_var = var(format!("shape{}", axis));
            let elementwise_strategy = &node.elementwise_strategies[axis];
            let unroll_factor = elementwise_strategy.unroll_factor();

            if unroll_factor > 1 {
                // ループアンローリングを適用
                body_statements =
                    self.generate_unrolled_loop(axis, unroll_factor, body_statements)?;
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
                    start: Box::new(AstNode::Const(Literal::Usize(0))),
                    step: Box::new(AstNode::Const(Literal::Usize(1))),
                    stop: Box::new(shape_var),
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

        // 全ての入力をロード
        let mut graph_inputs = Vec::new();
        for (i, src) in node.src.iter().enumerate() {
            let alu_var = self.fresh_alu();
            let input_ptr = var(format!("input{}", i));

            // 各srcノードのViewからオフセットを計算
            let offset = self.compute_offset_from_view(src, axes);
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();
            let load_node = load(input_ptr, offset, src_dtype.clone());

            // 変数を宣言（初期値付き）
            scope.declare(
                alu_var.clone(),
                src_dtype,
                Mutability::Mutable,
                AccessRegion::ThreadLocal,
                Some(load_node),
            )?;
            graph_inputs.push(alu_var);
        }

        // 中間結果を保存する配列
        let mut intermediate_results = Vec::new();

        // ops配列を順に評価
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
                        // ops[i]の中間結果
                        var(&intermediate_results[*idx])
                    }
                };
                operands.push(operand);
            }

            // 演算を適用
            let result = self.apply_elementwise_op(&fused_op.op, &operands)?;
            let result_var = self.fresh_alu();

            // 結果の型を取得（最初のオペランドの型を使用）
            let result_dtype = if let Some(first_operand) = operands.first() {
                if let AstNode::Var(var_name) = first_operand {
                    scope
                        .get(var_name)
                        .ok_or_else(|| format!("Variable {} not found in scope", var_name))?
                        .dtype
                        .clone()
                } else {
                    return Err("Expected variable as operand".to_string());
                }
            } else {
                return Err("Operation requires at least one operand".to_string());
            };

            // 変数を宣言（初期値付き）
            scope.declare(
                result_var.clone(),
                result_dtype,
                Mutability::Mutable,
                AccessRegion::ThreadLocal,
                Some(result),
            )?;
            intermediate_results.push(result_var);
        }

        // 最後の演算結果を出力にストア
        if let Some(last_result) = intermediate_results.last() {
            let output_ptr = var("output");
            let output_offset = self.compute_offset_from_view(node, axes);
            statements.push(store(output_ptr, output_offset, var(last_result)));
        } else {
            return Err("FusedElementwise requires at least one operation".to_string());
        }

        Ok(statements)
    }
}
