use crate::ast::{
    AccessRegion, AstNode, DType as AstDType, FunctionKind, Literal, Mutability, Scope, VarDecl,
    VarKind, helper::*,
};
use crate::graph::{GraphNode, ops::ElementwiseOp};
use log::debug;

use super::Lowerer;

impl Lowerer {
    /// Elementwise演算をカーネル関数に変換
    pub(super) fn lower_elementwise_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        op: &ElementwiseOp,
    ) -> Result<AstNode, String> {
        debug!("Lowering elementwise operation: {:?}", op);
        debug!("View: {:?}", node.view);
        debug!("Is contiguous: {}", node.view.is_contiguous());

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

        // Shape変数（各軸のサイズ）
        for i in 0..ndim {
            params.push(VarDecl {
                name: format!("shape{}", i),
                dtype: AstDType::Usize,
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
            });
        }

        // ループ本体の生成
        let body_statements = self.generate_elementwise_loops(node, op, ndim)?;

        // カーネル関数のbodyを作成（Blockノード）
        let body = AstNode::Block {
            statements: body_statements,
            scope: Box::new(Scope::new()),
        };

        // カーネル関数名
        let function_name = format!("kernel_{}", node_id);

        // 生成されたコードをログ出力
        debug!("Generated function with {} parameters", params.len());

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
            FunctionKind::Normal, // まずは通常の関数として（並列化は後で）
            params,
            AstDType::Tuple(vec![]), // unit型
            body,
        ))
    }

    /// Elementwise演算のループを生成
    pub(super) fn generate_elementwise_loops(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        ndim: usize,
    ) -> Result<Vec<AstNode>, String> {
        if ndim == 0 {
            // スカラー演算（ループなし）
            return self.generate_elementwise_body(node, op, &[]);
        }

        // ネストしたループを生成（外側から内側へ）
        let mut body_statements =
            self.generate_elementwise_body(node, op, &(0..ndim).collect::<Vec<_>>())?;

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
                    scope: Box::new(Scope::new()),
                };

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

    /// ループアンローリングを適用したループを生成
    pub(super) fn generate_unrolled_loop(
        &mut self,
        axis: usize,
        unroll_factor: usize,
        body_statements: Vec<AstNode>,
    ) -> Result<Vec<AstNode>, String> {
        let loop_var = format!("ridx{}", axis);
        let shape_var = var(format!("shape{}", axis));

        // メインループ: shape / unroll_factor 回のイテレーション
        let mut unrolled_body = vec![];

        for i in 0..unroll_factor {
            // ridx{axis} = ridx{axis}_base * unroll_factor + i
            let offset = if i == 0 {
                var(format!("{}_base", loop_var))
            } else {
                var(format!("{}_base", loop_var)) + AstNode::Const(Literal::Usize(i))
            };

            // ループ変数を置き換えた本体を生成
            let mut iter_body = body_statements.clone();
            for stmt in &mut iter_body {
                self.substitute_loop_var(stmt, &loop_var, &offset);
            }

            unrolled_body.extend(iter_body);
        }

        let unrolled_loop_body = AstNode::Block {
            statements: unrolled_body,
            scope: Box::new(Scope::new()),
        };

        // メインループ: for ridx{axis}_base in 0..(shape{axis}/unroll_factor)
        let main_loop_stop = idiv(
            shape_var.clone(),
            AstNode::Const(Literal::Usize(unroll_factor)),
        );

        let main_loop = AstNode::Range {
            var: format!("{}_base", loop_var),
            start: Box::new(AstNode::Const(Literal::Usize(0))),
            step: Box::new(AstNode::Const(Literal::Usize(1))),
            stop: Box::new(main_loop_stop),
            body: Box::new(unrolled_loop_body),
        };

        // 残り処理: for ridx{axis} in (shape{axis}/unroll_factor)*unroll_factor..shape{axis}
        let remainder_start = idiv(
            shape_var.clone(),
            AstNode::Const(Literal::Usize(unroll_factor)),
        ) * AstNode::Const(Literal::Usize(unroll_factor));

        let remainder_loop_body = AstNode::Block {
            statements: body_statements,
            scope: Box::new(Scope::new()),
        };

        let remainder_loop = AstNode::Range {
            var: loop_var,
            start: Box::new(remainder_start),
            step: Box::new(AstNode::Const(Literal::Usize(1))),
            stop: Box::new(shape_var),
            body: Box::new(remainder_loop_body),
        };

        Ok(vec![main_loop, remainder_loop])
    }

    /// ASTノード内のループ変数を置換
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn substitute_loop_var(
        &self,
        node: &mut AstNode,
        var_name: &str,
        replacement: &AstNode,
    ) {
        match node {
            AstNode::Var(name) if name == var_name => {
                *node = replacement.clone();
            }
            AstNode::Add(lhs, rhs)
            | AstNode::Mul(lhs, rhs)
            | AstNode::Max(lhs, rhs)
            | AstNode::Rem(lhs, rhs)
            | AstNode::Idiv(lhs, rhs) => {
                self.substitute_loop_var(lhs, var_name, replacement);
                self.substitute_loop_var(rhs, var_name, replacement);
            }
            AstNode::Recip(inner)
            | AstNode::Sqrt(inner)
            | AstNode::Log2(inner)
            | AstNode::Exp2(inner)
            | AstNode::Sin(inner) => {
                self.substitute_loop_var(inner, var_name, replacement);
            }
            AstNode::Cast(inner, _) => {
                self.substitute_loop_var(inner, var_name, replacement);
            }
            AstNode::Load { ptr, offset, .. } => {
                self.substitute_loop_var(ptr, var_name, replacement);
                self.substitute_loop_var(offset, var_name, replacement);
            }
            AstNode::Store {
                ptr, offset, value, ..
            } => {
                self.substitute_loop_var(ptr, var_name, replacement);
                self.substitute_loop_var(offset, var_name, replacement);
                self.substitute_loop_var(value, var_name, replacement);
            }
            AstNode::Assign { value, .. } => {
                self.substitute_loop_var(value, var_name, replacement);
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    self.substitute_loop_var(stmt, var_name, replacement);
                }
            }
            _ => {}
        }
    }

    /// Elementwise演算の本体を生成（ループ内部の処理）
    pub(super) fn generate_elementwise_body(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        axes: &[usize],
    ) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();

        // 入力をロード（各入力のViewを考慮）
        let mut loaded_values = Vec::new();
        for (i, src) in node.src.iter().enumerate() {
            let alu_var = self.fresh_alu();
            let input_ptr = var(format!("input{}", i));

            // 各srcノードのViewからオフセットを計算
            let offset = self.compute_offset_from_view(src, axes);
            let load_node = load(input_ptr, offset);

            statements.push(assign(&alu_var, load_node));
            loaded_values.push(var(&alu_var));
        }

        // 演算を適用
        let result = self.apply_elementwise_op(op, &loaded_values)?;
        let result_var = self.fresh_alu();
        statements.push(assign(&result_var, result));

        // 結果をストア（出力のViewを考慮）
        let output_ptr = var("output");
        let output_offset = self.compute_offset_from_view(node, axes);
        statements.push(store(output_ptr, output_offset, var(&result_var)));

        Ok(statements)
    }

    /// Elementwise演算をASTノードに変換
    pub(super) fn apply_elementwise_op(
        &self,
        op: &ElementwiseOp,
        operands: &[AstNode],
    ) -> Result<AstNode, String> {
        match op {
            ElementwiseOp::Add => {
                if operands.len() != 2 {
                    return Err("Add requires 2 operands".to_string());
                }
                Ok(operands[0].clone() + operands[1].clone())
            }
            ElementwiseOp::Mul => {
                if operands.len() != 2 {
                    return Err("Mul requires 2 operands".to_string());
                }
                Ok(operands[0].clone() * operands[1].clone())
            }
            ElementwiseOp::Neg => {
                if operands.len() != 1 {
                    return Err("Neg requires 1 operand".to_string());
                }
                // -x = -1 * x
                Ok(AstNode::Const(Literal::F32(-1.0)) * operands[0].clone())
            }
            ElementwiseOp::Max => {
                if operands.len() != 2 {
                    return Err("Max requires 2 operands".to_string());
                }
                Ok(max(operands[0].clone(), operands[1].clone()))
            }
            ElementwiseOp::Rem => {
                if operands.len() != 2 {
                    return Err("Rem requires 2 operands".to_string());
                }
                Ok(operands[0].clone() % operands[1].clone())
            }
            ElementwiseOp::Idiv => {
                if operands.len() != 2 {
                    return Err("Idiv requires 2 operands".to_string());
                }
                Ok(idiv(operands[0].clone(), operands[1].clone()))
            }
            ElementwiseOp::Recip => {
                if operands.len() != 1 {
                    return Err("Recip requires 1 operand".to_string());
                }
                Ok(recip(operands[0].clone()))
            }
        }
    }
}
