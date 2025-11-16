use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::{DType as GraphDType, GraphNode, ops::ReduceOp};
use log::debug;

use super::Lowerer;

impl Lowerer {
    /// Reduce演算をカーネル関数に変換
    pub(super) fn lower_reduce_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        op: &ReduceOp,
        axis: usize,
    ) -> Result<AstNode, String> {
        debug!("Lowering reduce operation: {:?} on axis {}", op, axis);

        if node.src.is_empty() {
            return Err("Reduce operation requires at least one input".to_string());
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
        params.push(self.create_input_param(0, &input.dtype)?);
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(input_shape));

        // ループ本体の生成
        let body_statements = self.generate_reduce_loops(node, op, axis)?;

        debug!("Generated reduce function with {} parameters", params.len());

        // カーネル関数を作成して返す
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// Reduce演算のループを生成
    pub(super) fn generate_reduce_loops(
        &mut self,
        node: &GraphNode,
        op: &ReduceOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let mut scope = Scope::new();
        let output_ndim = node.view.shape().len();

        // 入力shapeを取得（後でループ生成に使用）
        let input = &node.src[0];
        let input_shape = input.view.shape();

        // 出力がスカラーの場合とテンソルの場合で処理を分ける
        if output_ndim == 0 {
            // 全縮約（スカラー出力）
            return self.generate_reduce_to_scalar(node, op, axis, &mut scope);
        }

        // テンソル出力の場合
        // アキュムレータ初期化、縮約ループ、書き込みを含む本体を生成
        let mut body_statements =
            self.generate_reduce_body_with_axis(node, op, axis, &mut scope)?;

        // 出力の各軸についてループを生成（逆順に、内側から外側へ）
        for out_idx in (0..output_ndim).rev() {
            // 出力軸out_idxは入力軸in_idxに対応
            // 縮約軸より前ならそのまま、縮約軸以降なら+1
            let in_idx = if out_idx < axis { out_idx } else { out_idx + 1 };

            let loop_var = format!("oidx{}", out_idx);
            // 入力shapeから直接AstNodeに変換
            let shape_expr: AstNode = input_shape[in_idx].clone().into();

            let loop_body = block(body_statements, scope.clone());

            // 外側のループ用に新しいスコープを作成
            scope = Scope::new();

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

    /// スカラー出力への全縮約を生成
    pub(super) fn generate_reduce_to_scalar(
        &mut self,
        node: &GraphNode,
        op: &ReduceOp,
        _axis: usize,
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let input_ndim = input_shape.len();

        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_reduce_init_value(op, &node.dtype)?;
        let acc_dtype = match &node.dtype {
            crate::graph::DType::F32 => AstDType::F32,
            crate::graph::DType::Unknown => {
                return Err("Cannot determine dtype for Unknown".to_string());
            }
        };

        // scope.declareを使用して変数を宣言
        scope.declare(acc_var.clone(), acc_dtype, Mutability::Mutable)?;

        // 初期値を代入
        statements.push(assign(&acc_var, init_value));

        // 全ての軸についてループしてアキュムレート
        let mut accumulate_statements = vec![self.generate_accumulate_statement(
            &acc_var,
            op,
            &(0..input_ndim).collect::<Vec<_>>(),
            input,
        )?];

        // ループを逆順に作成（内側から外側へ）
        for i in (0..input_ndim).rev() {
            let loop_var = format!("ridx{}", i);
            // 入力shapeから直接AstNodeに変換
            let shape_expr: AstNode = input_shape[i].clone().into();

            let loop_body = block(accumulate_statements, Scope::new());

            accumulate_statements = vec![range(
                loop_var,
                const_int(0),
                const_int(1),
                shape_expr,
                loop_body,
            )];
        }

        statements.extend(accumulate_statements);

        // 結果をoutput[0]に書き込み
        let output_ptr = var("output");
        let output_offset = const_int(0);
        statements.push(store(output_ptr, output_offset, var(&acc_var)));

        Ok(statements)
    }

    /// 指定軸での縮約を含む本体を生成（出力がテンソルの場合）
    pub(super) fn generate_reduce_body_with_axis(
        &mut self,
        node: &GraphNode,
        op: &ReduceOp,
        axis: usize,
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_reduce_init_value(op, &node.dtype)?;
        let acc_dtype = match &node.dtype {
            crate::graph::DType::F32 => AstDType::F32,
            crate::graph::DType::Unknown => {
                return Err("Cannot determine dtype for Unknown".to_string());
            }
        };

        // scope.declareを使用して変数を宣言
        scope.declare(acc_var.clone(), acc_dtype, Mutability::Mutable)?;

        // 初期値を代入
        statements.push(assign(&acc_var, init_value));

        // 縮約軸についてループしてアキュムレート
        let loop_var = format!("ridx{}", axis);
        // 入力shapeから直接AstNodeに変換
        let shape_expr: AstNode = input_shape[axis].clone().into();

        // ループ内でアキュムレートする
        // 入力のインデックスを構築: 出力インデックス + 縮約軸インデックス
        let output_ndim = node.view.shape().len();
        let mut input_axes = Vec::new();
        for out_idx in 0..output_ndim {
            let in_idx = if out_idx < axis { out_idx } else { out_idx + 1 };
            input_axes.push(in_idx);
        }

        let accumulate_stmt = self.generate_accumulate_statement_with_reduce_axis(
            &acc_var,
            op,
            &input_axes,
            axis,
            input,
        )?;

        let reduce_loop = range(
            loop_var,
            const_int(0),
            const_int(1),
            shape_expr,
            block(vec![accumulate_stmt], Scope::new()),
        );

        statements.push(reduce_loop);

        // 結果を出力に書き込み
        let output_ptr = var("output");
        let output_axes: Vec<usize> = (0..output_ndim).collect();
        let output_offset = self.compute_offset_for_output(&output_axes, node);
        statements.push(store(output_ptr, output_offset, var(&acc_var)));

        Ok(statements)
    }

    /// ReduceOpに応じた初期値を取得
    pub(super) fn get_reduce_init_value(
        &self,
        op: &ReduceOp,
        dtype: &GraphDType,
    ) -> Result<AstNode, String> {
        match op {
            ReduceOp::Sum => match dtype {
                GraphDType::F32 => Ok(const_f32(0.0)),
                GraphDType::Unknown => {
                    Err("Cannot determine init value for Unknown dtype".to_string())
                }
            },
            ReduceOp::Prod => match dtype {
                GraphDType::F32 => Ok(const_f32(1.0)),
                GraphDType::Unknown => {
                    Err("Cannot determine init value for Unknown dtype".to_string())
                }
            },
            ReduceOp::Max => match dtype {
                GraphDType::F32 => Ok(const_f32(f32::NEG_INFINITY)),
                GraphDType::Unknown => {
                    Err("Cannot determine init value for Unknown dtype".to_string())
                }
            },
        }
    }

    /// アキュムレート文を生成
    pub(super) fn generate_accumulate_statement(
        &mut self,
        acc_var: &str,
        op: &ReduceOp,
        axes: &[usize],
        input: &GraphNode,
    ) -> Result<AstNode, String> {
        // 入力から値をロード
        let input_ptr = var("input0");
        let offset = self.compute_offset_for_input(axes, input);
        let input_ptr_dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
        let input_dtype = input_ptr_dtype.deref_type().clone();
        let loaded_value = load(input_ptr, offset, input_dtype);

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(op, acc, loaded_value)?;

        Ok(assign(acc_var, result))
    }

    /// 縮約軸を含むアキュムレート文を生成
    pub(super) fn generate_accumulate_statement_with_reduce_axis(
        &mut self,
        acc_var: &str,
        op: &ReduceOp,
        output_axes: &[usize],
        reduce_axis: usize,
        input: &GraphNode,
    ) -> Result<AstNode, String> {
        // 入力のインデックスを構築
        // output_axes[i]の位置にoidx{i}を、reduce_axisの位置にridx{reduce_axis}を配置
        let input_ptr = var("input0");
        let offset =
            self.compute_offset_for_input_with_reduce_axis(output_axes, reduce_axis, input);
        let input_ptr_dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
        let input_dtype = input_ptr_dtype.deref_type().clone();
        let loaded_value = load(input_ptr, offset, input_dtype);

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(op, acc, loaded_value)?;

        Ok(assign(acc_var, result))
    }

    /// Reduce演算をASTノードに変換
    pub(super) fn apply_reduce_op(
        &self,
        op: &ReduceOp,
        acc: AstNode,
        value: AstNode,
    ) -> Result<AstNode, String> {
        match op {
            ReduceOp::Sum => Ok(acc + value),
            ReduceOp::Prod => Ok(acc * value),
            ReduceOp::Max => Ok(max(acc, value)),
        }
    }
}
