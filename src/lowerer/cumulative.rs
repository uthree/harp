// Cumulative演算のコード生成

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::{GraphNode, ops::CumulativeOp};
use log::debug;

use super::Lowerer;

impl Lowerer {
    /// Cumulative演算をカーネル関数に変換
    pub(super) fn lower_cumulative_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        op: &CumulativeOp,
        axis: usize,
    ) -> Result<AstNode, String> {
        debug!("Lowering cumulative operation: {:?} on axis {}", op, axis);

        if node.src.is_empty() {
            return Err("Cumulative operation requires at least one input".to_string());
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
        params.push(self.create_input_param(0, &input.dtype)?);
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(input_shape));

        // ループ本体の生成
        let body_statements = self.generate_cumulative_loops(node, op, axis)?;

        debug!(
            "Generated cumulative function with {} parameters",
            params.len()
        );

        // カーネル関数を作成して返す
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// Cumulative演算のループを生成
    pub(super) fn generate_cumulative_loops(
        &mut self,
        node: &GraphNode,
        op: &CumulativeOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let scope = Scope::new();
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let ndim = input_shape.len();

        // 累積演算本体を生成（アキュムレータ初期化、累積ループ）
        let mut body_statements = self.generate_cumulative_body(node, op, axis)?;

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

    /// Cumulative演算の本体を生成（指定軸に沿った累積）
    pub(super) fn generate_cumulative_body(
        &mut self,
        node: &GraphNode,
        op: &CumulativeOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let ndim = input_shape.len();
        let mut scope = Scope::new();
        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_cumulative_init_value(op, &node.dtype)?;
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
        // 1. 入力値を読み込み
        // 2. アキュムレータを更新
        // 3. 結果を出力に書き込み
        let mut inner_statements = Vec::new();

        // 入力値を読み込み（Viewのストライドを使ったオフセット計算）
        let input_ptr = var("input0");
        let input_offset = self.compute_offset_for_cumulative(input, ndim, axis, &loop_var, "idx");
        let input_ptr_dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
        let input_dtype = input_ptr_dtype.deref_type().clone();
        let input_value = load(input_ptr.clone(), input_offset.clone(), input_dtype.clone());

        let alu_var = self.fresh_alu();
        scope.declare(alu_var.clone(), acc_dtype.clone(), Mutability::Mutable)?;
        inner_statements.push(assign(&alu_var, input_value));

        // アキュムレータを更新
        let accumulate_expr = self.apply_cumulative_op(op, var(&acc_var), var(&alu_var));
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

    /// Cumulative演算用のオフセット計算
    fn compute_offset_for_cumulative(
        &self,
        node: &GraphNode,
        ndim: usize,
        cumulative_axis: usize,
        cumulative_var: &str,
        index_prefix: &str,
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
                        var(format!("{}{}", index_prefix, i))
                    };
                    let stride: AstNode = stride_expr.clone().into();
                    result = result + idx_var * stride;
                }

                result
            }
        }
    }

    /// Cumulative演算の初期値を取得
    fn get_cumulative_init_value(
        &self,
        op: &CumulativeOp,
        dtype: &crate::graph::DType,
    ) -> Result<AstNode, String> {
        match dtype {
            crate::graph::DType::F32 => match op {
                CumulativeOp::Sum => Ok(const_f32(0.0)),
                CumulativeOp::Prod => Ok(const_f32(1.0)),
            },
            crate::graph::DType::Unknown => {
                Err("Cannot determine init value for Unknown dtype".to_string())
            }
        }
    }

    /// Cumulative演算を適用
    fn apply_cumulative_op(&self, op: &CumulativeOp, acc: AstNode, value: AstNode) -> AstNode {
        match op {
            CumulativeOp::Sum => acc + value,
            CumulativeOp::Prod => acc * value,
        }
    }
}
