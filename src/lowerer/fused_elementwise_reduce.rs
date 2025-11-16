use crate::ast::{AstNode, Mutability, Scope, helper::*};
use crate::graph::{GraphNode, ops::ReduceOp};
use log::debug;
use std::collections::HashMap;

use super::Lowerer;

impl Lowerer {
    /// FusedElementwiseReduce演算をカーネル関数に変換
    pub(super) fn lower_fused_elementwise_reduce_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        axis: usize,
    ) -> Result<AstNode, String> {
        debug!(
            "Lowering fused elementwise-reduce operation: reduce {:?} on axis {}",
            reduce_op, axis
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
            params.push(self.create_input_param(i, &src.dtype)?);
        }

        // 出力バッファーとShape変数
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(input_shape));

        // ループ本体の生成
        let body_statements =
            self.generate_fused_elementwise_reduce_loops(node, expr, reduce_op, axis)?;

        debug!(
            "Generated fused elementwise-reduce function with {} parameters",
            params.len()
        );

        // カーネル関数を作成して返す
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// FusedElementwiseReduce演算のループを生成
    fn generate_fused_elementwise_reduce_loops(
        &mut self,
        node: &GraphNode,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let output_ndim = node.view.shape().len();
        let input = &node.src[0];
        let input_shape = input.view.shape();

        // 出力がスカラーの場合とテンソルの場合で処理を分ける
        if output_ndim == 0 {
            // 全縮約（スカラー出力）
            let mut scope = Scope::new();
            return self.generate_fused_er_to_scalar(node, expr, reduce_op, axis, &mut scope);
        }

        // テンソル出力の場合
        let mut scope = Scope::new();
        let mut body_statements =
            self.generate_fused_er_body_with_axis(node, expr, reduce_op, axis, &mut scope)?;

        // 出力の各軸についてループを生成（逆順に、内側から外側へ）
        for out_idx in (0..output_ndim).rev() {
            // 出力軸out_idxは入力軸in_idxに対応
            // 縮約軸より前ならそのまま、縮約軸以降なら+1
            let in_idx = if out_idx < axis { out_idx } else { out_idx + 1 };

            let loop_var = format!("oidx{}", out_idx);
            let shape_expr: AstNode = input_shape[in_idx].clone().into();

            let loop_body = block(body_statements, scope.clone());
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

    /// スカラー出力への融合全縮約を生成
    fn generate_fused_er_to_scalar(
        &mut self,
        node: &GraphNode,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        _axis: usize,
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let input_ndim = input_shape.len();

        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_reduce_init_value(reduce_op, &node.dtype)?;
        let acc_ptr_dtype = self.graph_dtype_to_ast_ptr(&node.dtype)?;
        let acc_dtype = acc_ptr_dtype.deref_type().clone();
        scope.declare(acc_var.clone(), acc_dtype, Mutability::Mutable)?;
        statements.push(assign(&acc_var, init_value));

        // 全ての軸についてネストしたループを生成
        let axes: Vec<usize> = (0..input_ndim).collect();
        let inner_scope = Scope::new();
        let accumulate_stmt =
            self.generate_fused_er_accumulate_statement(&acc_var, expr, reduce_op, &axes, node)?;

        let mut body = vec![accumulate_stmt];
        let mut current_scope = inner_scope.clone();

        // ループを逆順に作成（内側から外側へ）
        for &axis in axes.iter().rev() {
            let loop_var = format!("ridx{}", axis);
            let shape_expr: AstNode = input_shape[axis].clone().into();
            let loop_body = block(body, current_scope);
            current_scope = Scope::new();
            body = vec![range(
                loop_var,
                const_int(0),
                const_int(1),
                shape_expr,
                loop_body,
            )];
        }

        statements.extend(body);

        // 結果を出力に書き込み
        let output_ptr = var("output");
        statements.push(store(output_ptr, const_int(0), var(&acc_var)));

        Ok(statements)
    }

    /// 軸を指定した縮約の本体を生成
    fn generate_fused_er_body_with_axis(
        &mut self,
        node: &GraphNode,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        axis: usize,
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_shape = input.view.shape();
        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_acc();
        let init_value = self.get_reduce_init_value(reduce_op, &node.dtype)?;
        let acc_ptr_dtype = self.graph_dtype_to_ast_ptr(&node.dtype)?;
        let acc_dtype = acc_ptr_dtype.deref_type().clone();
        scope.declare(acc_var.clone(), acc_dtype, Mutability::Mutable)?;
        statements.push(assign(&acc_var, init_value));

        // 縮約軸についてループ
        let loop_var = format!("ridx{}", axis);
        let shape_expr: AstNode = input_shape[axis].clone().into();

        // インデックス変数の設定: oidx{i} と ridx{axis} を使う
        let output_ndim = node.view.shape().len();
        let mut output_axes = Vec::new();
        for out_idx in 0..output_ndim {
            let in_idx = if out_idx < axis { out_idx } else { out_idx + 1 };
            output_axes.push(in_idx);
        }

        let inner_scope = Scope::new();
        let accumulate_stmt = self.generate_fused_er_accumulate_statement_with_reduce_axis(
            &acc_var,
            expr,
            reduce_op,
            &output_axes,
            axis,
            node,
        )?;

        let reduce_loop = range(
            loop_var,
            const_int(0),
            const_int(1),
            shape_expr,
            block(vec![accumulate_stmt], inner_scope),
        );

        statements.push(reduce_loop);

        // 結果を出力に書き込み
        let output_ptr = var("output");
        let output_axes_for_offset: Vec<usize> = (0..output_ndim).collect();
        let output_offset = self.compute_offset_for_output(&output_axes_for_offset, node);
        statements.push(store(output_ptr, output_offset, var(&acc_var)));

        Ok(statements)
    }

    /// 融合elementwise-reduceのアキュムレート文を生成
    fn generate_fused_er_accumulate_statement(
        &mut self,
        acc_var: &str,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        axes: &[usize],
        node: &GraphNode,
    ) -> Result<AstNode, String> {
        // 入力をロードしてマッピングを作成
        let mut mappings = HashMap::new();
        for (i, src) in node.src.iter().enumerate() {
            let input_ptr = var(format!("input{}", i));
            let offset = self.compute_offset_from_view(src, axes);
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();
            let loaded = load(input_ptr, offset, src_dtype);
            mappings.insert(i.to_string(), loaded);
        }

        // exprのWildcardを置き換え
        let elementwise_result = expr.substitute(&mappings);

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(reduce_op, acc, elementwise_result)?;

        Ok(assign(acc_var, result))
    }

    /// 縮約軸を含む融合elementwise-reduceのアキュムレート文を生成
    #[allow(clippy::too_many_arguments)]
    fn generate_fused_er_accumulate_statement_with_reduce_axis(
        &mut self,
        acc_var: &str,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        output_axes: &[usize],
        reduce_axis: usize,
        node: &GraphNode,
    ) -> Result<AstNode, String> {
        // 入力をロードしてマッピングを作成（oidx + ridxを使用）
        let mut mappings = HashMap::new();
        for (i, src) in node.src.iter().enumerate() {
            let input_ptr = var(format!("input{}", i));
            let offset =
                self.compute_offset_for_input_with_reduce_axis(output_axes, reduce_axis, src);
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();
            let loaded = load(input_ptr, offset, src_dtype);
            mappings.insert(i.to_string(), loaded);
        }

        // exprのWildcardを置き換え
        let elementwise_result = expr.substitute(&mappings);

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(reduce_op, acc, elementwise_result)?;

        Ok(assign(acc_var, result))
    }
}
