// FusedElementwiseCumulative演算のコード生成

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::{
    GraphNode,
    ops::{CumulativeOp, FusedElementwiseOp, FusedInput},
};
use log::debug;

use super::Lowerer;

impl Lowerer {
    /// FusedElementwiseCumulative演算をカーネル関数に変換
    pub(super) fn lower_fused_elementwise_cumulative_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        elementwise_ops: &[FusedElementwiseOp],
        cumulative_op: &CumulativeOp,
        axis: usize,
    ) -> Result<AstNode, String> {
        debug!(
            "Lowering fused elementwise-cumulative operation: {} elementwise ops, cumulative: {:?} on axis {}",
            elementwise_ops.len(),
            cumulative_op,
            axis
        );

        if node.src.is_empty() {
            return Err(
                "FusedElementwiseCumulative operation requires at least one input".to_string(),
            );
        }

        let input = &node.src[0];
        let input_shape = input.view.shape();
        let input_ndim = input_shape.len();

        if axis >= input_ndim {
            return Err(format!(
                "Cumulative axis {} is out of bounds for shape with {} dimensions",
                axis, input_ndim
            ));
        }

        // パラメータを生成: 入力バッファー、出力バッファー、shape変数
        let mut params = Vec::new();

        // 入力バッファー
        for (i, src) in node.src.iter().enumerate() {
            params.push(self.create_input_param(i, &src.dtype)?);
        }

        // 出力バッファーとShape変数
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(input_shape));

        // ループ本体の生成
        let body_statements = self.generate_fused_elementwise_cumulative_loops(
            node,
            elementwise_ops,
            cumulative_op,
            axis,
        )?;

        debug!(
            "Generated fused elementwise-cumulative function with {} parameters",
            params.len()
        );

        // カーネル関数を作成して返す
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// FusedElementwiseCumulative演算のループを生成
    fn generate_fused_elementwise_cumulative_loops(
        &mut self,
        node: &GraphNode,
        elementwise_ops: &[FusedElementwiseOp],
        cumulative_op: &CumulativeOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let scope = Scope::new();
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let ndim = input_shape.len();

        // 累積演算本体を生成（アキュムレータ初期化、累積ループ）
        let mut body_statements =
            self.generate_fused_ec_body(node, elementwise_ops, cumulative_op, axis)?;

        // 累積軸以外の各軸についてループを生成（逆順に、内側から外側へ）
        let mut current_scope = scope;
        for i in (0..ndim).rev() {
            if i == axis {
                // 累積軸は内部で処理されるのでスキップ
                continue;
            }

            let loop_var = format!("idx{}", i);
            let shape_expr: AstNode = input_shape[i].clone().into();

            let loop_body = block(body_statements, current_scope);
            current_scope = Scope::new();

            body_statements = vec![range(
                loop_var,
                const_int(0),
                const_int(1),
                shape_expr,
                loop_body,
            )];
        }

        Ok(body_statements)
    }

    /// FusedElementwiseCumulative演算の本体を生成（指定軸に沿った累積）
    fn generate_fused_ec_body(
        &mut self,
        node: &GraphNode,
        elementwise_ops: &[FusedElementwiseOp],
        cumulative_op: &CumulativeOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let ndim = input_shape.len();
        let mut scope = Scope::new();
        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_cumulative_init_value(cumulative_op, &node.dtype)?;
        let acc_dtype = match &node.dtype {
            crate::graph::DType::F32 => AstDType::F32,
            crate::graph::DType::Unknown => {
                return Err("Cannot determine dtype for Unknown".to_string());
            }
        };

        // scope.declareを使用して変数を宣言
        scope.declare(acc_var.clone(), acc_dtype.clone(), Mutability::Mutable)?;
        statements.push(assign(&acc_var, init_value));

        // 累積軸に沿ってループ
        let loop_var = format!("cumidx{}", axis);
        let shape_expr: AstNode = input_shape[axis].clone().into();

        // ループ内の処理:
        // 1. elementwise演算チェーンを評価
        // 2. アキュムレータを更新
        // 3. 結果を出力に書き込み
        let mut inner_statements = Vec::new();

        // elementwise演算チェーンを評価
        let elementwise_result = self.evaluate_fused_elementwise_chain_with_cumulative_axis(
            elementwise_ops,
            axis,
            &loop_var,
            node,
        )?;

        let alu_var = self.fresh_alu();
        scope.declare(alu_var.clone(), acc_dtype.clone(), Mutability::Mutable)?;
        inner_statements.push(assign(&alu_var, elementwise_result));

        // アキュムレータを更新
        let accumulate_expr = self.apply_cumulative_op(cumulative_op, var(&acc_var), var(&alu_var));
        inner_statements.push(assign(&acc_var, accumulate_expr));

        // 結果を出力に書き込み（出力のインデックスは入力と同じ）
        let output_ptr = var("output");
        let output_offset = self.compute_offset_for_cumulative_output(node, ndim, axis, &loop_var);
        inner_statements.push(store(output_ptr, output_offset, var(&acc_var)));

        // ループを作成
        let loop_body = block(inner_statements, Scope::new());
        let cumulative_loop = range(loop_var, const_int(0), const_int(1), shape_expr, loop_body);

        statements.push(cumulative_loop);

        // スコープをBlockにラップして返す
        Ok(vec![block(statements, scope)])
    }

    /// elementwise演算チェーンを評価（idx + cumidx変数を使用）
    fn evaluate_fused_elementwise_chain_with_cumulative_axis(
        &mut self,
        ops: &[FusedElementwiseOp],
        cumulative_axis: usize,
        cumulative_var: &str,
        node: &GraphNode,
    ) -> Result<AstNode, String> {
        // 全ての入力から値をロード（idx + cumidxを使用）
        let mut graph_inputs = Vec::new();
        for (i, src) in node.src.iter().enumerate() {
            let input_ptr = var(format!("input{}", i));
            let ndim = src.view.shape().len();
            let offset = self.compute_offset_for_cumulative_input(
                src,
                ndim,
                cumulative_axis,
                cumulative_var,
            );
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();
            graph_inputs.push(load(input_ptr, offset, src_dtype));
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
                    FusedInput::Const(lit) => AstNode::Const(lit.clone()),
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
            .ok_or_else(|| "FusedElementwiseCumulative requires at least one operation".to_string())
    }

    /// Cumulative入力用のオフセット計算（idx + cumidx変数を使用）
    fn compute_offset_for_cumulative_input(
        &self,
        node: &GraphNode,
        ndim: usize,
        cumulative_axis: usize,
        cumulative_var: &str,
    ) -> AstNode {
        use crate::graph::shape::View;

        match &node.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for (i, stride_expr) in strides.iter().take(ndim).enumerate() {
                    let idx_var = if i == cumulative_axis {
                        var(cumulative_var)
                    } else {
                        var(format!("idx{}", i))
                    };
                    let stride: AstNode = stride_expr.clone().into();
                    result = result + idx_var * stride;
                }

                result
            }
        }
    }

    /// Cumulative出力用のオフセット計算
    fn compute_offset_for_cumulative_output(
        &self,
        node: &GraphNode,
        ndim: usize,
        cumulative_axis: usize,
        cumulative_var: &str,
    ) -> AstNode {
        use crate::graph::shape::View;

        match &node.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for (i, stride_expr) in strides.iter().take(ndim).enumerate() {
                    let idx_var = if i == cumulative_axis {
                        var(cumulative_var)
                    } else {
                        var(format!("idx{}", i))
                    };
                    let stride: AstNode = stride_expr.clone().into();
                    result = result + idx_var * stride;
                }

                result
            }
        }
    }
}
