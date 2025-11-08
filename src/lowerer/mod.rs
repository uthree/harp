use crate::ast::{
    AccessRegion, AstNode, DType as AstDType, Function, FunctionKind, Literal, Mutability, Scope,
    VarDecl, VarKind, helper::*,
};
use crate::backend::KernelSignature;
use crate::graph::{DType as GraphDType, Graph, GraphNode, ops::ElementwiseOp, ops::GraphOp};
use log::debug;
use std::collections::{HashMap, HashSet, VecDeque};

pub struct Lowerer {
    alu_counter: usize, // 一時変数のカウンター
}

/// トポロジカルソートの結果。各世代（Generation）は並列実行可能なノード群。
pub type TopologicalOrder = Vec<Vec<GraphNode>>;

/// GraphNodeから内部のポインタを取得するヘルパー関数
fn node_ptr(node: &GraphNode) -> *const () {
    node.as_ptr() as *const ()
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

/// GraphをProgramに変換する公開関数
///
/// Graphの全ノードをカーネル関数に変換し、Programとして返します。
/// 現時点では各ノードを個別のカーネル関数として生成し、
/// kernel_main関数による統合は未実装です。
pub(crate) fn lower(graph: Graph) -> crate::ast::Program {
    let mut lowerer = Lowerer::new();

    // トポロジカルソートでノードを取得
    let generations = Lowerer::topological_sort(&graph);

    // Programを作成（entry_pointはとりあえず"main"）
    let mut program = crate::ast::Program::new("main".to_string());

    // 各世代の各ノードをカーネル関数に変換
    let mut kernel_id = 0;
    let mut first_kernel_name = String::new();
    for generation in generations {
        for node in generation {
            // Input ノードはスキップ
            if matches!(node.op, GraphOp::Input) {
                continue;
            }

            // カーネル関数を生成
            if let Ok(function) = lowerer.lower_node_to_kernel(&node, kernel_id) {
                let kernel_name = format!("kernel_{}", kernel_id);
                if kernel_id == 0 {
                    first_kernel_name = kernel_name.clone();
                }
                let _ = program.add_function(kernel_name, function);
                kernel_id += 1;
            }
        }
    }

    // entry_pointを最初のカーネルに設定（もしあれば）
    if !first_kernel_name.is_empty() {
        program.entry_point = first_kernel_name;
    }

    program
}

impl Lowerer {
    pub fn new() -> Self {
        Self { alu_counter: 0 }
    }

    /// GraphからKernelSignatureを生成
    pub fn create_signature(graph: &Graph) -> KernelSignature {
        use crate::backend::{BufferSignature, KernelSignature};
        use std::collections::HashSet;

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut shape_vars = HashSet::new();

        // 入力バッファのシグネチャを生成
        for (name, weak_node) in graph.inputs() {
            if let Some(node_rc) = weak_node.upgrade() {
                let shape: Vec<_> = node_rc.view.shape().to_vec();

                // shape内の変数名を収集
                for expr in &shape {
                    Self::collect_shape_vars(expr, &mut shape_vars);
                }

                inputs.push(BufferSignature::new(name.clone(), shape));
            }
        }

        // 出力バッファのシグネチャを生成
        for (name, node) in graph.outputs() {
            let shape: Vec<_> = node.view.shape().to_vec();

            // shape内の変数名を収集
            for expr in &shape {
                Self::collect_shape_vars(expr, &mut shape_vars);
            }

            outputs.push(BufferSignature::new(name.clone(), shape));
        }

        // shape_varsをソートしてVecに変換
        let mut shape_vars_vec: Vec<_> = shape_vars.into_iter().collect();
        shape_vars_vec.sort();

        KernelSignature::new(inputs, outputs, shape_vars_vec)
    }

    /// Exprから変数名を再帰的に収集
    fn collect_shape_vars(expr: &crate::graph::shape::Expr, vars: &mut HashSet<String>) {
        use crate::graph::shape::Expr;

        match expr {
            Expr::Var(name) => {
                vars.insert(name.clone());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Rem(a, b) => {
                Self::collect_shape_vars(a, vars);
                Self::collect_shape_vars(b, vars);
            }
            Expr::Const(_) => {}
        }
    }

    /// 新しい一時変数名を生成
    fn fresh_alu(&mut self) -> String {
        let name = format!("alu{}", self.alu_counter);
        self.alu_counter += 1;
        name
    }

    /// GraphNodeを一つのカーネル関数に変換（最も単純なケース）
    /// 前提：contiguous, 全軸Sequential, SIMD未使用
    pub fn lower_node_to_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<Function, String> {
        match &node.op {
            GraphOp::Elementwise { op, .. } => self.lower_elementwise_kernel(node, node_id, op),
            GraphOp::Reduce { op, axis, .. } => self.lower_reduce_kernel(node, node_id, op, *axis),
            GraphOp::Contiguous { .. } => self.lower_contiguous_kernel(node, node_id),
            _ => Err(format!("Unsupported operation: {:?}", node.op)),
        }
    }

    /// Reduce演算をカーネル関数に変換
    fn lower_reduce_kernel(
        &mut self,
        node: &GraphNode,
        _node_id: usize,
        op: &crate::graph::ops::ReduceOp,
        axis: usize,
    ) -> Result<Function, String> {
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

        // 入力バッファー
        let input_dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
        params.push(VarDecl {
            name: "input0".to_string(),
            dtype: input_dtype,
            mutability: Mutability::Immutable,
            region: AccessRegion::Shared,
            kind: VarKind::Normal,
        });

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
        let body_statements = self.generate_reduce_loops(node, op, axis)?;

        // カーネル関数を作成
        let function = Function::new(
            FunctionKind::Normal,
            params,
            AstDType::Tuple(vec![]),
            body_statements,
        )?;

        // 生成されたコードをログ出力
        debug!(
            "Generated reduce function with {} parameters",
            function.params.len()
        );
        if log::log_enabled!(log::Level::Debug) {
            use crate::backend::metal::MetalRenderer;
            let mut renderer = MetalRenderer::new();
            let code = renderer.render_function("reduce_kernel_fn", &function);
            debug!("Generated code:\n{}", code);
        }

        Ok(function)
    }

    /// Reduce演算のループを生成
    fn generate_reduce_loops(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::ReduceOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let output_ndim = node.view.shape().len();

        // 出力がスカラーの場合とテンソルの場合で処理を分ける
        if output_ndim == 0 {
            // 全縮約（スカラー出力）
            return self.generate_reduce_to_scalar(node, op, axis);
        }

        // テンソル出力の場合
        // アキュムレータ初期化、縮約ループ、書き込みを含む本体を生成
        let mut body_statements = self.generate_reduce_body_with_axis(node, op, axis)?;

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

    /// スカラー出力への全縮約を生成
    fn generate_reduce_to_scalar(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::ReduceOp,
        _axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let input_ndim = input.view.shape().len();

        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_alu();
        let init_value = self.get_reduce_init_value(op, &node.dtype)?;
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

    /// 指定軸での縮約を含む本体を生成（出力がテンソルの場合）
    fn generate_reduce_body_with_axis(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::ReduceOp,
        axis: usize,
    ) -> Result<Vec<AstNode>, String> {
        let input = &node.src[0];
        let mut statements = Vec::new();

        // アキュムレータを初期化
        let acc_var = self.fresh_alu();
        let init_value = self.get_reduce_init_value(op, &node.dtype)?;
        statements.push(assign(&acc_var, init_value));

        // 縮約軸についてループしてアキュムレート
        let loop_var = format!("ridx{}", axis);
        let shape_var = var(format!("shape{}", axis));

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

    /// ReduceOpに応じた初期値を取得
    fn get_reduce_init_value(
        &self,
        op: &crate::graph::ops::ReduceOp,
        dtype: &GraphDType,
    ) -> Result<AstNode, String> {
        use crate::graph::ops::ReduceOp;

        match op {
            ReduceOp::Add => match dtype {
                GraphDType::F32 => Ok(AstNode::Const(Literal::F32(0.0))),
                GraphDType::Unknown => {
                    Err("Cannot determine init value for Unknown dtype".to_string())
                }
            },
            ReduceOp::Mul => match dtype {
                GraphDType::F32 => Ok(AstNode::Const(Literal::F32(1.0))),
                GraphDType::Unknown => {
                    Err("Cannot determine init value for Unknown dtype".to_string())
                }
            },
            ReduceOp::Max => match dtype {
                GraphDType::F32 => Ok(AstNode::Const(Literal::F32(f32::NEG_INFINITY))),
                GraphDType::Unknown => {
                    Err("Cannot determine init value for Unknown dtype".to_string())
                }
            },
        }
    }

    /// アキュムレート文を生成
    fn generate_accumulate_statement(
        &mut self,
        acc_var: &str,
        op: &crate::graph::ops::ReduceOp,
        axes: &[usize],
        input: &GraphNode,
    ) -> Result<AstNode, String> {
        // 入力から値をロード
        let input_ptr = var("input0");
        let offset = self.compute_offset_for_input(axes, input);
        let loaded_value = load(input_ptr, offset);

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(op, acc, loaded_value)?;

        Ok(assign(acc_var, result))
    }

    /// 縮約軸を含むアキュムレート文を生成
    fn generate_accumulate_statement_with_reduce_axis(
        &mut self,
        acc_var: &str,
        op: &crate::graph::ops::ReduceOp,
        output_axes: &[usize],
        reduce_axis: usize,
        input: &GraphNode,
    ) -> Result<AstNode, String> {
        // 入力のインデックスを構築
        // output_axes[i]の位置にoidx{i}を、reduce_axisの位置にridx{reduce_axis}を配置
        let input_ptr = var("input0");
        let offset =
            self.compute_offset_for_input_with_reduce_axis(output_axes, reduce_axis, input);
        let loaded_value = load(input_ptr, offset);

        // アキュムレート演算を適用
        let acc = var(acc_var);
        let result = self.apply_reduce_op(op, acc, loaded_value)?;

        Ok(assign(acc_var, result))
    }

    /// Reduce演算をASTノードに変換
    fn apply_reduce_op(
        &self,
        op: &crate::graph::ops::ReduceOp,
        acc: AstNode,
        value: AstNode,
    ) -> Result<AstNode, String> {
        use crate::graph::ops::ReduceOp;

        match op {
            ReduceOp::Add => Ok(acc + value),
            ReduceOp::Mul => Ok(acc * value),
            ReduceOp::Max => Ok(max(acc, value)),
        }
    }

    /// 入力のオフセット計算（ridx変数を使用）
    fn compute_offset_for_input(&self, axes: &[usize], input: &GraphNode) -> AstNode {
        use crate::graph::shape::View;

        match &input.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for &axis in axes {
                    let ridx = var(format!("ridx{}", axis));
                    let stride: AstNode = strides[axis].clone().into();
                    result = result + ridx * stride;
                }

                result
            }
        }
    }

    /// 出力のオフセット計算（oidx変数を使用）
    fn compute_offset_for_output(&self, axes: &[usize], output: &GraphNode) -> AstNode {
        use crate::graph::shape::View;

        match &output.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for &axis in axes {
                    let oidx = var(format!("oidx{}", axis));
                    let stride: AstNode = strides[axis].clone().into();
                    result = result + oidx * stride;
                }

                result
            }
        }
    }

    /// 入力のオフセット計算（縮約軸を含む、oidxとridxを組み合わせ）
    fn compute_offset_for_input_with_reduce_axis(
        &self,
        output_axes: &[usize],
        reduce_axis: usize,
        input: &GraphNode,
    ) -> AstNode {
        use crate::graph::shape::View;

        match &input.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                // 出力軸に対応する入力軸
                for (out_idx, &in_axis) in output_axes.iter().enumerate() {
                    let oidx = var(format!("oidx{}", out_idx));
                    let stride: AstNode = strides[in_axis].clone().into();
                    result = result + oidx * stride;
                }

                // 縮約軸
                let ridx = var(format!("ridx{}", reduce_axis));
                let stride: AstNode = strides[reduce_axis].clone().into();
                result = result + ridx * stride;

                result
            }
        }
    }

    /// Elementwise演算をカーネル関数に変換
    fn lower_elementwise_kernel(
        &mut self,
        node: &GraphNode,
        _node_id: usize,
        op: &ElementwiseOp,
    ) -> Result<Function, String> {
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

        // カーネル関数を作成
        let function = Function::new(
            FunctionKind::Normal, // まずは通常の関数として（並列化は後で）
            params,
            AstDType::Tuple(vec![]), // unit型
            body_statements,
        )?;

        // 生成されたコードをログ出力
        debug!(
            "Generated function with {} parameters",
            function.params.len()
        );
        if log::log_enabled!(log::Level::Debug) {
            use crate::backend::metal::MetalRenderer;
            let mut renderer = MetalRenderer::new();
            let code = renderer.render_function("kernel_fn", &function);
            debug!("Generated code:\n{}", code);
        }

        Ok(function)
    }

    /// Elementwise演算のループを生成
    fn generate_elementwise_loops(
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
    fn generate_unrolled_loop(
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
    fn substitute_loop_var(&self, node: &mut AstNode, var_name: &str, replacement: &AstNode) {
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
    fn generate_elementwise_body(
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

    /// Viewを考慮したオフセット計算
    fn compute_offset_from_view(&self, node: &GraphNode, axes: &[usize]) -> AstNode {
        use crate::graph::shape::View;

        if axes.is_empty() {
            // スカラーの場合
            match &node.view {
                View::Linear { offset, .. } => {
                    // Expr::intoでAstNodeに変換
                    offset.clone().into()
                }
            }
        } else {
            // テンソルの場合：offset + sum(ridx[i] * stride[i])
            match &node.view {
                View::Linear {
                    strides, offset, ..
                } => {
                    let mut result: AstNode = offset.clone().into();

                    for &axis in axes {
                        let ridx = var(format!("ridx{}", axis));
                        let stride: AstNode = strides[axis].clone().into();
                        result = result + ridx * stride;
                    }

                    result
                }
            }
        }
    }

    /// Elementwise演算をASTノードに変換
    fn apply_elementwise_op(
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

    /// Contiguous演算をカーネル関数に変換
    fn lower_contiguous_kernel(
        &mut self,
        node: &GraphNode,
        _node_id: usize,
    ) -> Result<Function, String> {
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
        });

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
        let body_statements = self.generate_contiguous_loops(node, ndim)?;

        // カーネル関数を作成
        let function = Function::new(
            FunctionKind::Normal,
            params,
            AstDType::Tuple(vec![]),
            body_statements,
        )?;

        // 生成されたコードをログ出力
        debug!(
            "Generated contiguous function with {} parameters",
            function.params.len()
        );
        if log::log_enabled!(log::Level::Debug) {
            use crate::backend::metal::MetalRenderer;
            let mut renderer = MetalRenderer::new();
            let code = renderer.render_function("contiguous_kernel_fn", &function);
            debug!("Generated code:\n{}", code);
        }

        Ok(function)
    }

    /// Contiguous演算のループを生成
    fn generate_contiguous_loops(
        &mut self,
        node: &GraphNode,
        ndim: usize,
    ) -> Result<Vec<AstNode>, String> {
        if ndim == 0 {
            // スカラーの場合（ループなし）
            return self.generate_contiguous_body(node, &[]);
        }

        // ネストしたループを生成（外側から内側へ）
        let mut body_statements =
            self.generate_contiguous_body(node, &(0..ndim).collect::<Vec<_>>())?;

        // ループを逆順に作成（内側から外側へ）
        for axis in (0..ndim).rev() {
            let loop_var = format!("ridx{}", axis);
            let shape_var = var(format!("shape{}", axis));

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

    /// Contiguous演算の本体を生成（ループ内部の処理）
    fn generate_contiguous_body(
        &mut self,
        node: &GraphNode,
        axes: &[usize],
    ) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();

        let input = &node.src[0];

        // 入力からロード（入力のViewを考慮）
        let input_ptr = var("input0");
        let input_offset = self.compute_offset_from_view(input, axes);
        let alu_var = self.fresh_alu();
        statements.push(assign(&alu_var, load(input_ptr, input_offset)));

        // 出力にストア（出力のViewを考慮）
        let output_ptr = var("output");
        let output_offset = self.compute_offset_from_view(node, axes);
        statements.push(store(output_ptr, output_offset, var(&alu_var)));

        Ok(statements)
    }

    /// GraphのDTypeをASTのPtr<DType>に変換
    fn graph_dtype_to_ast_ptr(&self, dtype: &GraphDType) -> Result<AstDType, String> {
        let element_dtype = match dtype {
            GraphDType::F32 => AstDType::F32,
            GraphDType::Unknown => return Err("Cannot convert Unknown dtype".to_string()),
        };
        Ok(AstDType::Ptr(Box::new(element_dtype)))
    }

    // === トポロジカルソート関連 ===

    /// Kahnのアルゴリズムを使用してグラフをトポロジカルソートし、世代別にグループ化する。
    /// 各世代のノードは同時に計算可能。
    pub fn topological_sort(graph: &Graph) -> TopologicalOrder {
        // 1. すべてのノードを収集（出力ノードから再帰的に辿る）
        let all_nodes = Self::collect_all_nodes(graph);

        // 2. 各ノードの入次数を計算（何個のノードから参照されているか）
        let mut in_degree: HashMap<*const (), usize> = HashMap::new();
        for node in &all_nodes {
            let ptr = node_ptr(node);
            in_degree.entry(ptr).or_insert(0);

            // このノードが参照する各srcノードの入次数を増やす
            for src in &node.src {
                let src_ptr = node_ptr(src);
                *in_degree.entry(src_ptr).or_insert(0) += 1;
            }
        }

        // 3. Kahnのアルゴリズムで世代別にグループ化
        let mut result: TopologicalOrder = Vec::new();
        let mut queue: VecDeque<GraphNode> = VecDeque::new();

        // 入次数が0のノード（誰からも参照されていない=出力ノード）をキューに追加
        for node in &all_nodes {
            let ptr = node_ptr(node);
            if in_degree[&ptr] == 0 {
                queue.push_back(node.clone());
            }
        }

        // 世代ごとに処理
        while !queue.is_empty() {
            let generation_size = queue.len();
            let mut current_generation = Vec::new();

            // 現在の世代のノードをすべて処理
            for _ in 0..generation_size {
                let node = queue.pop_front().unwrap();
                current_generation.push(node.clone());

                // このノードが参照するsrcノードの入次数を減らす
                for src in &node.src {
                    let src_ptr = node_ptr(src);
                    let degree = in_degree.get_mut(&src_ptr).unwrap();
                    *degree -= 1;

                    // 入次数が0になったらキューに追加
                    if *degree == 0 {
                        queue.push_back(src.clone());
                    }
                }
            }

            result.push(current_generation);
        }

        result
    }

    /// グラフの出力ノードから再帰的にすべてのノードを収集する
    fn collect_all_nodes(graph: &Graph) -> Vec<GraphNode> {
        let mut visited: HashSet<*const ()> = HashSet::new();
        let mut nodes: Vec<GraphNode> = Vec::new();

        for output_node in graph.outputs().values() {
            Self::collect_nodes_recursive(output_node, &mut visited, &mut nodes);
        }

        nodes
    }

    /// 再帰的にノードを収集する（深さ優先探索）
    fn collect_nodes_recursive(
        node: &GraphNode,
        visited: &mut HashSet<*const ()>,
        nodes: &mut Vec<GraphNode>,
    ) {
        let ptr = node_ptr(node);

        if visited.contains(&ptr) {
            return;
        }

        visited.insert(ptr);

        // 先にsrcノードを訪問（依存関係の順序）
        for src in &node.src {
            Self::collect_nodes_recursive(src, visited, nodes);
        }

        nodes.push(node.clone());
    }
}

#[cfg(test)]
mod tests;
