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
        // 現時点では、Elementwise演算のみをサポート
        match &node.op {
            GraphOp::Elementwise { op, .. } => self.lower_elementwise_kernel(node, node_id, op),
            _ => Err(format!("Unsupported operation: {:?}", node.op)),
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

        Ok(body_statements)
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
