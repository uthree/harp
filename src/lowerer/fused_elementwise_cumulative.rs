// FusedElementwiseCumulative演算のコード生成

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::{GraphNode, ops::CumulativeOp};
use log::debug;
use std::collections::HashMap;

use super::Lowerer;

impl Lowerer {
    /// FusedElementwiseCumulative演算をカーネル関数に変換
    pub(super) fn lower_fused_elementwise_cumulative_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        expr: &AstNode,
        cumulative_op: &CumulativeOp,
        axis: usize,
    ) -> Result<AstNode, String> {
        debug!(
            "Lowering fused elementwise-cumulative operation: cumulative {:?} on axis {}",
            cumulative_op, axis
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
        let body_statements =
            self.generate_fused_elementwise_cumulative_loops(node, expr, cumulative_op, axis)?;

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
        expr: &AstNode,
        cumulative_op: &CumulativeOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let scope = Scope::new();
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let ndim = input_shape.len();

        // 累積演算本体を生成（アキュムレータ初期化、累積ループ）
        let mut body_statements = self.generate_fused_ec_body(node, expr, cumulative_op, axis)?;

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
        expr: &AstNode,
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
            crate::graph::DType::Bool => AstDType::Bool,
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

        // 入力をロードしてマッピングを作成（直接load式をWildcardに対応）
        let mut mappings = HashMap::new();
        for (i, src) in node.src.iter().enumerate() {
            let input_ptr = var(format!("input{}", i));
            let src_ndim = src.view.shape().len();
            let offset = self.compute_offset_for_cumulative(src, src_ndim, axis, &loop_var, "idx");
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();
            let loaded = load(input_ptr, offset, src_dtype);
            mappings.insert(i.to_string(), loaded);
        }

        // exprのWildcardを置き換えてelementwise演算結果を生成
        let elementwise_result = expr.substitute(&mappings);

        // アキュムレータを更新（中間変数を排除）
        let accumulate_expr =
            self.apply_cumulative_op(cumulative_op, var(&acc_var), elementwise_result);
        inner_statements.push(assign(&acc_var, accumulate_expr));

        // 結果を出力に書き込み（出力のインデックスは入力と同じ）
        let output_ptr = var("output");
        let output_offset = self.compute_offset_for_cumulative(node, ndim, axis, &loop_var, "idx");
        inner_statements.push(store(output_ptr, output_offset, var(&acc_var)));

        // ループを作成
        let loop_body = block(inner_statements, Scope::new());
        let cumulative_loop = range(loop_var, const_int(0), const_int(1), shape_expr, loop_body);

        statements.push(cumulative_loop);

        // スコープをBlockにラップして返す
        Ok(vec![block(statements, scope)])
    }
}
