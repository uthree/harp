use crate::ast::{AstNode, Literal, Scope, helper::*};
use crate::graph::GraphNode;
use log::debug;
use std::collections::HashMap;

use super::Lowerer;

impl Lowerer {
    /// FusedElementwise演算をカーネル関数に変換
    pub(super) fn lower_fused_elementwise_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        expr: &AstNode,
    ) -> Result<AstNode, String> {
        debug!("Lowering fused elementwise operation with expr");
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
        let body_statements =
            self.generate_fused_elementwise_loops(node, expr, ndim, &mut scope)?;

        // カーネル関数を作成して返す
        Ok(self.create_kernel_function(node_id, params, body_statements, scope))
    }

    /// FusedElementwise演算のループを生成
    fn generate_fused_elementwise_loops(
        &mut self,
        node: &GraphNode,
        expr: &AstNode,
        ndim: usize,
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        let shape = node.view.shape();

        if ndim == 0 {
            // スカラー演算（ループなし）
            return self.generate_fused_elementwise_body(node, expr, &[], scope);
        }

        // ネストしたループを生成（外側から内側へ）
        let mut body_statements = self.generate_fused_elementwise_body(
            node,
            expr,
            &(0..ndim).collect::<Vec<_>>(),
            scope,
        )?;

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
        expr: &AstNode,
        axes: &[usize],
        _scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        // 最内側の軸のSIMD幅を取得（最後の軸を使用）
        let simd_width = if let Some(&last_axis) = axes.last() {
            node.elementwise_strategies[last_axis].simd_width()
        } else {
            1
        };

        // Wildcardを直接load式に対応させる（中間変数を排除）
        let mut mappings: HashMap<String, AstNode> = HashMap::new();
        for (i, src) in node.src.iter().enumerate() {
            let input_ptr = var(format!("input{}", i));

            // 各srcノードのViewからオフセットを計算
            let offset = self.compute_offset_from_view(src, axes);
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();

            // SIMD化: simd_width > 1の場合はベクトルロード
            let load_node = if simd_width > 1 {
                let vec_dtype = src_dtype.to_vec(simd_width);
                load_vec(input_ptr, offset, simd_width, vec_dtype)
            } else {
                load(input_ptr, offset, src_dtype)
            };

            // Wildcard("0"), Wildcard("1") 等に対応するload式をマッピング
            mappings.insert(i.to_string(), load_node);
        }

        // exprのWildcardを置き換えて最終的な式を生成
        let final_result = expr.substitute(&mappings);

        // 最終結果を直接出力にストア
        let output_ptr = var("output");
        let output_offset = self.compute_offset_from_view(node, axes);
        let store_stmt = store(output_ptr, output_offset, final_result);

        Ok(vec![store_stmt])
    }
}
