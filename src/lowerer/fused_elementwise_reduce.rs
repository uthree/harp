use crate::ast::{
    AccessRegion, AstNode, DType as AstDType, FunctionKind, Literal, Mutability, Scope, VarDecl,
    VarKind, helper::*,
};
use crate::graph::{
    GraphNode,
    ops::{FusedElementwiseOp, FusedInput, ReduceOp},
};
use log::debug;

use super::Lowerer;

impl Lowerer {
    /// FusedElementwiseReduce演算をカーネル関数に変換
    pub(super) fn lower_fused_elementwise_reduce_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        elementwise_ops: &[FusedElementwiseOp],
        reduce_op: &ReduceOp,
        axis: usize,
    ) -> Result<AstNode, String> {
        debug!(
            "Lowering fused elementwise-reduce operation: {} elementwise ops, reduce: {:?} on axis {}",
            elementwise_ops.len(),
            reduce_op,
            axis
        );

        if node.src.is_empty() {
            return Err("FusedElementwiseReduce operation requires at least one input".to_string());
        }

        let input = &node.src[0];
        let input_shape = input.view.shape();
        let input_ndim = input_shape.len();

        if axis >= input_ndim {
            return Err(format!(
                "Reduce axis {} is out of bounds for shape with {} dimensions",
                axis, input_ndim
            ));
        }

        // パラメータを生成: 入力バッファー、出力バッファー、shape変数
        let mut params = Vec::new();

        // 入力バッファー
        for (i, src) in node.src.iter().enumerate() {
            let input_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            params.push(VarDecl {
                name: format!("input{}", i),
                dtype: input_dtype,
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
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
        });

        // Shape変数（入力のshape）
        for i in 0..input_ndim {
            params.push(VarDecl {
                name: format!("shape{}", i),
                dtype: AstDType::Usize,
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
            });
        }

        // ループ本体の生成
        let body_statements =
            self.generate_fused_elementwise_reduce_loops(node, elementwise_ops, reduce_op, axis)?;

        // カーネル関数のbodyを作成（Blockノード）
        let body = AstNode::Block {
            statements: body_statements,
            scope: Box::new(Scope::new()),
        };

        // カーネル関数名
        let function_name = format!("kernel_{}", node_id);

        debug!(
            "Generated fused elementwise-reduce function with {} parameters",
            params.len()
        );

        // TODO: Renderer更新後にデバッグ出力を復活させる
        // if log::log_enabled!(log::Level::Debug) {
        //     use crate::backend::metal::MetalRenderer;
        //     let mut renderer = MetalRenderer::new();
        //     let code = renderer.render_function(&function_name, &function);
        //     debug!("Generated code:\n{}", code);
        // }

        // AstNode::Functionとして返す
        Ok(function(
            Some(function_name),
            FunctionKind::Normal,
            params,
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// FusedElementwiseReduce演算のループを生成
    fn generate_fused_elementwise_reduce_loops(
        &mut self,
        node: &GraphNode,
        elementwise_ops: &[FusedElementwiseOp],
        reduce_op: &ReduceOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let output_ndim = node.view.shape().len();

        // 出力がスカラーの場合とテンソルの場合で処理を分ける
        if output_ndim == 0 {
            // 全縮約（スカラー出力）
            return self.generate_fused_er_to_scalar(node, elementwise_ops, reduce_op, axis);
        }

        // テンソル出力の場合
        let mut body_statements =
            self.generate_fused_er_body_with_axis(node, elementwise_ops, reduce_op, axis)?;

        // 出力の各軸についてループを生成（逆順に、内側から外側へ）
        for out_idx in (0..output_ndim).rev() {
            // 出力軸out_idxは入力軸in_idxに対応
            // 縮約軸より前ならそのまま、縮約軸以降なら+1
            let in_idx = if out_idx < axis { out_idx } else { out_idx + 1 };

            let loop_var = format!("oidx{}", out_idx);
            let shape_var = var(format!("shape{}", in_idx));

            let loop_body = AstNode::Block {
                statements: body_statements,
                scope: Box::new(Scope::new()),
            };

            body_statements = vec![AstNode::Range {
                var: loop_var,
                start: Box::new(AstNode::Const(Literal::Usize(0))),
                step: Box::new(AstNode::Const(Literal::Usize(1))),
                stop: Box::new(shape_var),
                body: Box::new(loop_body),
            }];
        }

        Ok(body_statements)
    }

    /// スカラー出力への融合全縮約を生成
    fn generate_fused_er_to_scalar(
        &mut self,
        node: &GraphNode,
        elementwise_ops: &[FusedElementwiseOp],
        reduce_op: &ReduceOp,
        _axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_ndim = input.view.shape().len();

        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_reduce_init_value(reduce_op, &node.dtype)?;
        statements.push(assign(&acc_var, init_value));

        // 全ての軸についてループしてアキュムレート
        let mut accumulate_statements = vec![self.generate_fused_er_accumulate_statement(
            &acc_var,
            elementwise_ops,
            reduce_op,
            &(0..input_ndim).collect::<Vec<_>>(),
            node,
        )?];

        // ループを逆順に作成（内側から外側へ）
        for i in (0..input_ndim).rev() {
            let loop_var = format!("ridx{}", i);
            let shape_var = var(format!("shape{}", i));

            let loop_body = AstNode::Block {
                statements: accumulate_statements,
                scope: Box::new(Scope::new()),
            };

            accumulate_statements = vec![AstNode::Range {
                var: loop_var,
                start: Box::new(AstNode::Const(Literal::Usize(0))),
                step: Box::new(AstNode::Const(Literal::Usize(1))),
                stop: Box::new(shape_var),
                body: Box::new(loop_body),
            }];
        }

        statements.extend(accumulate_statements);

        // 結果をoutput[0]に書き込み
        let output_ptr = var("output");
        let output_offset = AstNode::Const(Literal::Usize(0));
        statements.push(store(output_ptr, output_offset, var(&acc_var)));

        Ok(statements)
    }

    /// 指定軸での融合縮約を含む本体を生成（出力がテンソルの場合）
    fn generate_fused_er_body_with_axis(
        &mut self,
        node: &GraphNode,
        elementwise_ops: &[FusedElementwiseOp],
        reduce_op: &ReduceOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_reduce_init_value(reduce_op, &node.dtype)?;
        statements.push(assign(&acc_var, init_value));

        // 縮約軸についてループしてアキュムレート
        let loop_var = format!("ridx{}", axis);
        let shape_var = var(format!("shape{}", axis));

        // ループ内でアキュムレートする
        let output_ndim = node.view.shape().len();
        let mut input_axes = Vec::new();
        for out_idx in 0..output_ndim {
            let in_idx = if out_idx < axis { out_idx } else { out_idx + 1 };
            input_axes.push(in_idx);
        }

        let accumulate_stmt = self.generate_fused_er_accumulate_statement_with_reduce_axis(
            &acc_var,
            elementwise_ops,
            reduce_op,
            &input_axes,
            axis,
            node,
        )?;

        let reduce_loop = AstNode::Range {
            var: loop_var,
            start: Box::new(AstNode::Const(Literal::Usize(0))),
            step: Box::new(AstNode::Const(Literal::Usize(1))),
            stop: Box::new(shape_var),
            body: Box::new(AstNode::Block {
                statements: vec![accumulate_stmt],
                scope: Box::new(Scope::new()),
            }),
        };

        statements.push(reduce_loop);

        // 結果を出力に書き込み
        let output_ptr = var("output");
        let output_axes: Vec<usize> = (0..output_ndim).collect();
        let output_offset = self.compute_offset_for_output(&output_axes, node);
        statements.push(store(output_ptr, output_offset, var(&acc_var)));

        Ok(statements)
    }

    /// 融合elementwise-reduceのアキュムレート文を生成
    fn generate_fused_er_accumulate_statement(
        &mut self,
        acc_var: &str,
        elementwise_ops: &[FusedElementwiseOp],
        reduce_op: &ReduceOp,
        axes: &[usize],
        node: &GraphNode,
    ) -> Result<AstNode, String> {
        // elementwise演算チェーンを評価して、その結果をアキュムレート
        let elementwise_result =
            self.evaluate_fused_elementwise_chain(elementwise_ops, axes, node)?;

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(reduce_op, acc, elementwise_result)?;

        Ok(assign(acc_var, result))
    }

    /// 縮約軸を含む融合elementwise-reduceのアキュムレート文を生成
    fn generate_fused_er_accumulate_statement_with_reduce_axis(
        &mut self,
        acc_var: &str,
        elementwise_ops: &[FusedElementwiseOp],
        reduce_op: &ReduceOp,
        output_axes: &[usize],
        reduce_axis: usize,
        node: &GraphNode,
    ) -> Result<AstNode, String> {
        // インデックス変数の設定: oidx{i} と ridx{reduce_axis} を使う
        // これを実現するため、軸のリストを構築
        let input_ndim = node.src[0].view.shape().len();
        let mut axes = vec![0; input_ndim];

        // 出力軸に対応する位置にはoidxを使う（実際の軸番号を設定）
        for (out_idx, &in_axis) in output_axes.iter().enumerate() {
            axes[in_axis] = out_idx + 100; // 100をオフセットとして使用してoidxを区別
        }
        // reduce軸にはridxを使う
        axes[reduce_axis] = reduce_axis;

        // elementwise演算チェーンを評価
        let elementwise_result = self.evaluate_fused_elementwise_chain_with_reduce_axis(
            elementwise_ops,
            output_axes,
            reduce_axis,
            node,
        )?;

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(reduce_op, acc, elementwise_result)?;

        Ok(assign(acc_var, result))
    }

    /// elementwise演算チェーンを評価（ridx変数を使用）
    fn evaluate_fused_elementwise_chain(
        &mut self,
        ops: &[FusedElementwiseOp],
        axes: &[usize],
        node: &GraphNode,
    ) -> Result<AstNode, String> {
        // 全ての入力から値をロード
        let mut graph_inputs = Vec::new();
        for (i, src) in node.src.iter().enumerate() {
            let input_ptr = var(format!("input{}", i));
            let offset = self.compute_offset_from_view(src, axes);
            graph_inputs.push(load(input_ptr, offset));
        }

        // 中間結果を保存
        let mut intermediate_results: Vec<AstNode> = Vec::new();

        // ops配列を順に評価
        for fused_op in ops {
            let mut operands = Vec::new();
            for input in &fused_op.inputs {
                let operand = match input {
                    FusedInput::GraphInput(idx) => graph_inputs[*idx].clone(),
                    FusedInput::IntermediateResult(idx) => intermediate_results[*idx].clone(),
                };
                operands.push(operand);
            }

            let result = self.apply_elementwise_op(&fused_op.op, &operands)?;
            intermediate_results.push(result);
        }

        // 最後の演算結果を返す
        intermediate_results
            .last()
            .cloned()
            .ok_or_else(|| "FusedElementwise requires at least one operation".to_string())
    }

    /// elementwise演算チェーンを評価（oidx+ridx変数を使用）
    fn evaluate_fused_elementwise_chain_with_reduce_axis(
        &mut self,
        ops: &[FusedElementwiseOp],
        output_axes: &[usize],
        reduce_axis: usize,
        node: &GraphNode,
    ) -> Result<AstNode, String> {
        // 全ての入力から値をロード（oidx + ridxを使用）
        let mut graph_inputs = Vec::new();
        for (i, src) in node.src.iter().enumerate() {
            let input_ptr = var(format!("input{}", i));
            let offset =
                self.compute_offset_for_input_with_reduce_axis(output_axes, reduce_axis, src);
            graph_inputs.push(load(input_ptr, offset));
        }

        // 中間結果を保存
        let mut intermediate_results: Vec<AstNode> = Vec::new();

        // ops配列を順に評価
        for fused_op in ops {
            let mut operands = Vec::new();
            for input in &fused_op.inputs {
                let operand = match input {
                    FusedInput::GraphInput(idx) => graph_inputs[*idx].clone(),
                    FusedInput::IntermediateResult(idx) => intermediate_results[*idx].clone(),
                };
                operands.push(operand);
            }

            let result = self.apply_elementwise_op(&fused_op.op, &operands)?;
            intermediate_results.push(result);
        }

        // 最後の演算結果を返す
        intermediate_results
            .last()
            .cloned()
            .ok_or_else(|| "FusedElementwise requires at least one operation".to_string())
    }
}
