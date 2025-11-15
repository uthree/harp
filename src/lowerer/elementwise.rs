use crate::ast::{
    AstNode, DType as AstDType, FunctionKind, Literal, Mutability, Scope, VarDecl, VarKind,
    helper::*,
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

        // 入力バッファー（srcノード、Constノードを除く）
        let mut input_idx = 0;
        for src in node.src.iter() {
            // Constノードはパラメータとして必要ない（コード内に直接埋め込まれる）
            if matches!(src.op, crate::graph::GraphOp::Const(_)) {
                continue;
            }
            let dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            params.push(VarDecl {
                name: format!("input{}", input_idx),
                dtype,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
            input_idx += 1;
        }

        // 出力バッファー
        let output_dtype = self.graph_dtype_to_ast_ptr(&node.dtype)?;
        params.push(VarDecl {
            name: "output".to_string(),
            dtype: output_dtype,
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

        // Shape変数（必要な変数のみをパラメータとして追加）
        let shape_params = self.extract_shape_params(shape);
        params.extend(shape_params);

        // ループ本体の生成（scopeも返す）
        let (body_statements, body_scope) = self.generate_elementwise_loops(node, op, ndim)?;

        // カーネル関数のbodyを作成（Blockノード）
        let body = AstNode::Block {
            statements: body_statements,
            scope: Box::new(body_scope),
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

    /// Elementwise演算のループを生成（statements と scope を返す）
    pub(super) fn generate_elementwise_loops(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        ndim: usize,
    ) -> Result<(Vec<AstNode>, Scope), String> {
        let mut scope = Scope::new();

        if ndim == 0 {
            // スカラー演算（ループなし）
            let statements = self.generate_elementwise_body(node, op, &[], &mut scope)?;
            return Ok((statements, scope));
        }

        // ネストしたループを生成（外側から内側へ）
        let mut body_statements =
            self.generate_elementwise_body(node, op, &(0..ndim).collect::<Vec<_>>(), &mut scope)?;

        // ループを逆順に作成（内側から外側へ）
        for axis in (0..ndim).rev() {
            let loop_var = format!("ridx{}", axis);
            // shapeから直接AstNodeに変換
            let shape_expr: AstNode = node.view.shape()[axis].clone().into();
            let elementwise_strategy = &node.elementwise_strategies[axis];
            let unroll_factor = elementwise_strategy.unroll_factor();
            let simd_width = elementwise_strategy.simd_width();

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
                scope = Scope::new();

                // ステップをSIMD幅に合わせる
                if simd_width > 1 {
                    // SIMD化されたループ
                    let simd_step = AstNode::Const(Literal::Int(simd_width as isize));

                    // メインループ: SIMD幅の倍数まで処理
                    // stop = (shape / simd_width) * simd_width
                    let simd_stop = idiv(
                        shape_expr.clone(),
                        AstNode::Const(Literal::Int(simd_width as isize)),
                    ) * AstNode::Const(Literal::Int(simd_width as isize));

                    let simd_loop = AstNode::Range {
                        var: loop_var.clone(),
                        start: Box::new(AstNode::Const(Literal::Int(0))),
                        step: Box::new(simd_step),
                        stop: Box::new(simd_stop.clone()),
                        body: Box::new(loop_body.clone()),
                    };

                    // 残りループ: スカラー処理（SIMD幅の倍数から最後まで）
                    // スカラー用のbodyを生成（simd_width = 1として再生成）
                    let mut scalar_scope = Scope::new();
                    let scalar_body_statements = self.generate_elementwise_body_with_simd(
                        node,
                        op,
                        &(0..ndim).collect::<Vec<_>>(),
                        &mut scalar_scope,
                        Some(1), // スカラー処理
                    )?;

                    let scalar_loop_body = AstNode::Block {
                        statements: scalar_body_statements,
                        scope: Box::new(scalar_scope),
                    };

                    let remainder_loop = AstNode::Range {
                        var: loop_var.clone(),
                        start: Box::new(simd_stop),
                        step: Box::new(AstNode::Const(Literal::Int(1))),
                        stop: Box::new(shape_expr),
                        body: Box::new(scalar_loop_body),
                    };

                    body_statements = vec![simd_loop, remainder_loop];
                } else {
                    // 通常のスカラーループ
                    body_statements = vec![AstNode::Range {
                        var: loop_var.clone(),
                        start: Box::new(AstNode::Const(Literal::Int(0))),
                        step: Box::new(AstNode::Const(Literal::Int(1))),
                        stop: Box::new(shape_expr),
                        body: Box::new(loop_body),
                    }];
                }
            }
        }

        Ok((body_statements, scope))
    }

    /// ループアンローリングを適用したループを生成
    pub(super) fn generate_unrolled_loop_with_shape(
        &mut self,
        axis: usize,
        unroll_factor: usize,
        shape_expr: &AstNode,
        body_statements: Vec<AstNode>,
    ) -> Result<Vec<AstNode>, String> {
        let loop_var = format!("ridx{}", axis);

        // メインループ: step数をunroll_factorに変更
        let mut unrolled_body = vec![];

        for i in 0..unroll_factor {
            // ridx{axis} + i でオフセット
            let offset = if i == 0 {
                var(loop_var.clone())
            } else {
                var(loop_var.clone()) + AstNode::Const(Literal::Int(i as isize))
            };

            // ループ変数を置き換えた本体を生成
            let mut iter_body = body_statements.clone();
            for stmt in &mut iter_body {
                self.substitute_loop_var(stmt, &loop_var, &offset);
            }

            unrolled_body.extend(iter_body);
        }

        // メインループのスコープ
        let main_scope = Scope::new();
        let unrolled_loop_body = AstNode::Block {
            statements: unrolled_body,
            scope: Box::new(main_scope),
        };

        // メインループ: for ridx{axis} in 0..(shape{axis}/unroll_factor)*unroll_factor step unroll_factor
        // stop値を切り下げて、unroll_factorの倍数にする
        let aligned_stop = idiv(
            shape_expr.clone(),
            AstNode::Const(Literal::Int(unroll_factor as isize)),
        ) * AstNode::Const(Literal::Int(unroll_factor as isize));

        let main_loop = AstNode::Range {
            var: loop_var.clone(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(unroll_factor as isize))),
            stop: Box::new(aligned_stop.clone()),
            body: Box::new(unrolled_loop_body),
        };

        // 残り処理: for ridx{axis} in aligned_stop..shape{axis} step 1
        let remainder_scope = Scope::new();
        let remainder_loop_body = AstNode::Block {
            statements: body_statements,
            scope: Box::new(remainder_scope),
        };

        let remainder_loop = AstNode::Range {
            var: loop_var,
            start: Box::new(aligned_stop),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(shape_expr.clone()),
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
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                // ループ変数がシャドウイングされている場合は本体を置換しない
                if var != var_name {
                    self.substitute_loop_var(start, var_name, replacement);
                    self.substitute_loop_var(step, var_name, replacement);
                    self.substitute_loop_var(stop, var_name, replacement);
                    self.substitute_loop_var(body, var_name, replacement);
                }
            }
            AstNode::BitwiseAnd(lhs, rhs)
            | AstNode::BitwiseOr(lhs, rhs)
            | AstNode::BitwiseXor(lhs, rhs)
            | AstNode::LeftShift(lhs, rhs)
            | AstNode::RightShift(lhs, rhs) => {
                self.substitute_loop_var(lhs, var_name, replacement);
                self.substitute_loop_var(rhs, var_name, replacement);
            }
            AstNode::BitwiseNot(inner) => {
                self.substitute_loop_var(inner, var_name, replacement);
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
        scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        self.generate_elementwise_body_with_simd(node, op, axes, scope, None)
    }

    /// Elementwise演算の本体を生成（SIMD幅をオーバーライド可能）
    fn generate_elementwise_body_with_simd(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        axes: &[usize],
        scope: &mut Scope,
        override_simd_width: Option<usize>,
    ) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();

        // 最内側の軸のSIMD幅を取得（最後の軸を使用）またはオーバーライド値を使用
        let simd_width = if let Some(width) = override_simd_width {
            width
        } else if let Some(&last_axis) = axes.last() {
            node.elementwise_strategies[last_axis].simd_width()
        } else {
            1
        };

        // 入力をロード（各入力のViewを考慮）
        let mut loaded_values = Vec::new();
        let mut input_idx = 0; // Constノードをスキップした後のパラメータインデックス
        for src in node.src.iter() {
            // Constノードの場合は直接定数として使用
            if let crate::graph::ops::GraphOp::Const(lit) = &src.op {
                // 定数値を直接AstNodeとして使用
                loaded_values.push(AstNode::Const(lit.clone()));
                continue;
            }

            let alu_var = self.fresh_alu();
            let input_ptr = var(format!("input{}", input_idx));
            input_idx += 1;

            // 各srcノードのViewからオフセットを計算
            let offset = self.compute_offset_from_view(src, axes);
            let src_ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = src_ptr_dtype.deref_type().clone();

            // SIMD化: simd_width > 1の場合はベクトルロード
            let (load_node, final_dtype) = if simd_width > 1 {
                let vec_dtype = src_dtype.to_vec(simd_width);
                (
                    load_vec(input_ptr, offset, simd_width, vec_dtype.clone()),
                    vec_dtype,
                )
            } else {
                (load(input_ptr, offset, src_dtype.clone()), src_dtype)
            };

            // 変数を宣言
            scope.declare(alu_var.clone(), final_dtype, Mutability::Mutable)?;

            // 初期値を代入
            statements.push(assign(&alu_var, load_node));

            loaded_values.push(var(&alu_var));
        }

        // 演算を適用
        let result = self.apply_elementwise_op(op, &loaded_values)?;
        let result_var = self.fresh_alu();

        // 結果の型を推定（SIMD化を考慮）
        let result_dtype = if let Some(src) = node.src.first() {
            let ptr_dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            let src_dtype = ptr_dtype.deref_type().clone();
            if simd_width > 1 {
                src_dtype.to_vec(simd_width)
            } else {
                src_dtype
            }
        } else {
            return Err("No input found for elementwise operation".to_string());
        };

        // 変数を宣言
        scope.declare(result_var.clone(), result_dtype, Mutability::Mutable)?;

        // 結果を変数に代入
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
            ElementwiseOp::Log2 => {
                if operands.len() != 1 {
                    return Err("Log2 requires 1 operand".to_string());
                }
                Ok(crate::ast::helper::log2(operands[0].clone()))
            }
            ElementwiseOp::Exp2 => {
                if operands.len() != 1 {
                    return Err("Exp2 requires 1 operand".to_string());
                }
                Ok(crate::ast::helper::exp2(operands[0].clone()))
            }
            ElementwiseOp::Sin => {
                if operands.len() != 1 {
                    return Err("Sin requires 1 operand".to_string());
                }
                Ok(crate::ast::helper::sin(operands[0].clone()))
            }
            ElementwiseOp::Sqrt => {
                if operands.len() != 1 {
                    return Err("Sqrt requires 1 operand".to_string());
                }
                Ok(crate::ast::helper::sqrt(operands[0].clone()))
            }
        }
    }
}
