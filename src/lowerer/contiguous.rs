use crate::ast::{
    AccessRegion, AstNode, DType as AstDType, FunctionKind, Literal, Mutability, Scope, VarDecl,
    VarKind, helper::*,
};
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

        // 入力バッファー
        let input_dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
        params.push(VarDecl {
            name: "input0".to_string(),
            dtype: input_dtype,
            mutability: Mutability::Immutable,
            region: AccessRegion::Shared,
            kind: VarKind::Normal,
            initial_value: None,
        });

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
        let body_statements = self.generate_contiguous_loops(node, ndim)?;

        // カーネル関数のbodyを作成（Blockノード）
        let body = AstNode::Block {
            statements: body_statements,
            scope: Box::new(Scope::new()),
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

    /// Contiguous演算のループを生成
    pub(super) fn generate_contiguous_loops(
        &mut self,
        node: &GraphNode,
        ndim: usize,
    ) -> Result<Vec<AstNode>, String> {
        let mut scope = Scope::new();

        if ndim == 0 {
            // スカラーの場合（ループなし）
            return self.generate_contiguous_body(node, &[], &mut scope);
        }

        // ネストしたループを生成（外側から内側へ）
        let mut body_statements =
            self.generate_contiguous_body(node, &(0..ndim).collect::<Vec<_>>(), &mut scope)?;

        // ループを逆順に作成（内側から外側へ）
        for axis in (0..ndim).rev() {
            let loop_var = format!("ridx{}", axis);
            let shape_var = var(format!("shape{}", axis));

            let loop_body = AstNode::Block {
                statements: body_statements,
                scope: Box::new(scope.clone()),
            };

            // 外側のループ用に新しいスコープを作成
            scope = Scope::new();

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

        // 変数を宣言（初期値付き）
        scope.declare(
            alu_var.clone(),
            input_dtype.clone(),
            Mutability::Mutable,
            AccessRegion::ThreadLocal,
            Some(load(input_ptr, input_offset, input_dtype)),
        )?;

        // 出力にストア（出力のViewを考慮）
        let output_ptr = var("output");
        let output_offset = self.compute_offset_from_view(node, axes);
        statements.push(store(output_ptr, output_offset, var(&alu_var)));

        Ok(statements)
    }
}
