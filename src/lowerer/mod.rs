use crate::graph::{Graph, GraphNode, ops::GraphOp};
use std::collections::{HashMap, HashSet, VecDeque};

// モジュール宣言
mod contiguous;
mod cumulative;
mod elementwise;
mod fold;
mod fused_elementwise;
mod fused_elementwise_reduce;
mod fused_reduce;
mod reduce;
mod utils;

pub struct Lowerer {
    alu_counter: usize, // 一時変数のカウンター
    acc_counter: usize, // アキュムレータのカウンター
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
/// Graphの全ノードをカーネル関数に変換し、AstNode::Programとして返します。
/// 現時点では各ノードを個別のカーネル関数として生成し、
/// kernel_main関数による統合は未実装です。
pub(crate) fn lower(graph: Graph) -> crate::ast::AstNode {
    let mut lowerer = Lowerer::new();

    // トポロジカルソートでノードを取得
    let generations = Lowerer::topological_sort(&graph);

    // 関数リストを作成
    let mut functions = Vec::new();

    // 各世代の各ノードをカーネル関数に変換
    let mut kernel_id = 0;
    for generation in generations {
        for node in generation {
            // Input と Const ノードはスキップ（Constは使用先で直接埋め込まれる）
            if matches!(node.op, GraphOp::Input | GraphOp::Const(_)) {
                continue;
            }

            // 各カーネル関数は独立しているので、変数カウンターをリセット
            lowerer.alu_counter = 0;
            lowerer.acc_counter = 0;

            // カーネル関数を生成（AstNode::Functionとして）
            if let Ok(function) = lowerer.lower_node_to_kernel(&node, kernel_id) {
                functions.push(function);
                kernel_id += 1;
            }
        }
    }

    // main関数を生成してすべてのカーネルを呼び出す
    if kernel_id > 0 {
        // main関数を生成
        let main_function = generate_main_function(&functions, kernel_id);
        functions.push(main_function);
    }

    // AstNode::Programを作成
    crate::ast::helper::program(functions, "main")
}

/// すべてのカーネルを呼び出すmain関数を生成
fn generate_main_function(
    kernel_functions: &[crate::ast::AstNode],
    kernel_count: usize,
) -> crate::ast::AstNode {
    use crate::ast::{AstNode, DType, FunctionKind, Scope, VarDecl};

    // すべてのカーネルのパラメータを収集して統合
    let mut all_params: Vec<VarDecl> = Vec::new();
    let mut seen_params: std::collections::HashSet<String> = std::collections::HashSet::new();

    for kernel_func in kernel_functions {
        if let AstNode::Function { params, .. } = kernel_func {
            for param in params {
                if !seen_params.contains(&param.name) {
                    all_params.push(param.clone());
                    seen_params.insert(param.name.clone());
                }
            }
        }
    }

    // 各カーネルを呼び出す文を生成
    let mut statements = Vec::new();
    for i in 0..kernel_count {
        let kernel_name = format!("kernel_{}", i);
        if let Some(AstNode::Function { params, .. }) = kernel_functions.get(i) {
            // カーネル関数の引数を収集
            let args: Vec<AstNode> = params
                .iter()
                .map(|p| AstNode::Var(p.name.clone()))
                .collect();

            // Call文を追加
            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });
        }
    }

    // main関数のbodyを作成（Blockノード）
    let body = AstNode::Block {
        statements,
        scope: Box::new(Scope::new()),
    };

    // AstNode::Functionとして返す
    crate::ast::helper::function(
        Some("main"),
        FunctionKind::Normal,
        all_params,
        DType::Tuple(vec![]), // void
        body,
    )
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            alu_counter: 0,
            acc_counter: 0,
        }
    }

    /// 新しい一時変数名を生成
    pub(super) fn fresh_alu(&mut self) -> String {
        let name = format!("alu{}", self.alu_counter);
        self.alu_counter += 1;
        name
    }

    /// 新しいアキュムレータ変数名を生成
    pub(super) fn fresh_acc(&mut self) -> String {
        let name = format!("acc{}", self.acc_counter);
        self.acc_counter += 1;
        name
    }

    /// shapeの式から必要なパラメータを抽出
    ///
    /// shapeの各軸の式から変数を収集し、それらをパラメータとして返す。
    /// 定数のみの式の場合は空のVecを返す。
    pub(super) fn extract_shape_params(
        &self,
        shape: &[crate::graph::shape::Expr],
    ) -> Vec<crate::ast::VarDecl> {
        use crate::ast::{DType as AstDType, Mutability, VarDecl, VarKind};
        use std::collections::BTreeSet;

        // 全ての式から変数を収集
        let mut all_vars = BTreeSet::new();
        for expr in shape {
            all_vars.extend(expr.collect_vars());
        }

        // 変数をパラメータに変換（BTreeSetを使っているので自動的にソートされる）
        all_vars
            .into_iter()
            .map(|var_name| VarDecl {
                name: var_name,
                dtype: AstDType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            })
            .collect()
    }

    /// ネストしたループを生成する汎用関数
    ///
    /// # Arguments
    /// * `ndim` - ループのネストレベル（次元数）
    /// * `shape` - 各軸のサイズを表す式
    /// * `var_prefix` - ループ変数のプレフィックス（例: "ridx"）
    /// * `inner_statements` - 最内側のループ本体の文
    /// * `inner_scope` - 最内側のスコープ
    ///
    /// # Returns
    /// ネストされたループを含む文のベクトルとスコープのタプル
    pub(super) fn generate_nested_loops(
        &self,
        ndim: usize,
        shape: &[crate::graph::shape::Expr],
        var_prefix: &str,
        inner_statements: Vec<crate::ast::AstNode>,
        inner_scope: crate::ast::Scope,
    ) -> (Vec<crate::ast::AstNode>, crate::ast::Scope) {
        use crate::ast::Scope;
        use crate::ast::helper::{block, const_int, range};

        let mut body_statements = inner_statements;
        let mut scope = inner_scope;

        // ループを逆順に作成（内側から外側へ）
        for axis in (0..ndim).rev() {
            let loop_var = format!("{}{}", var_prefix, axis);
            let shape_expr: crate::ast::AstNode = shape[axis].clone().into();

            let loop_body = block(body_statements, scope);

            scope = Scope::new();

            body_statements = vec![range(
                loop_var,
                const_int(0),
                const_int(1),
                shape_expr,
                loop_body,
            )];
        }

        (body_statements, scope)
    }

    /// カーネル関数の標準パラメータを生成
    ///
    /// # Arguments
    /// * `inputs` - 入力ノードのスライス
    /// * `output` - 出力ノード
    /// * `shape` - 出力の形状
    ///
    /// # Returns
    /// パラメータ宣言のベクトル（入力バッファ、出力バッファ、形状パラメータの順）
    #[allow(dead_code)]
    pub(super) fn generate_kernel_params(
        &mut self,
        inputs: &[&GraphNode],
        output: &GraphNode,
        shape: &[crate::graph::shape::Expr],
    ) -> Result<Vec<crate::ast::VarDecl>, String> {
        use crate::ast::{Mutability, VarDecl, VarKind};

        let mut params = Vec::new();

        // 入力バッファー
        for (i, input) in inputs.iter().enumerate() {
            let dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
            params.push(VarDecl {
                name: format!("input{}", i),
                dtype,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
        }

        // 出力バッファー
        let output_dtype = self.graph_dtype_to_ast_ptr(&output.dtype)?;
        params.push(VarDecl {
            name: "output".to_string(),
            dtype: output_dtype,
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

        // Shape変数
        let shape_params = self.extract_shape_params(shape);
        params.extend(shape_params);

        Ok(params)
    }

    /// カーネル関数ノードを作成
    ///
    /// # Arguments
    /// * `node_id` - ノードID（カーネル名の生成に使用）
    /// * `params` - パラメータ宣言
    /// * `body_statements` - 本体の文
    /// * `body_scope` - 本体のスコープ
    ///
    /// # Returns
    /// AstNode::Function
    #[allow(dead_code)]
    pub(super) fn wrap_as_kernel(
        &self,
        node_id: usize,
        params: Vec<crate::ast::VarDecl>,
        body_statements: Vec<crate::ast::AstNode>,
        body_scope: crate::ast::Scope,
    ) -> crate::ast::AstNode {
        use crate::ast::helper::{block, function};
        use crate::ast::{DType, FunctionKind};

        let body = block(body_statements, body_scope);

        function(
            Some(format!("kernel_{}", node_id)),
            FunctionKind::Normal,
            params,
            DType::Tuple(vec![]),
            body,
        )
    }

    /// GraphNodeを一つのカーネル関数に変換（最も単純なケース）
    /// 前提：contiguous, 全軸Sequential, SIMD未使用
    pub fn lower_node_to_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<crate::ast::AstNode, String> {
        match &node.op {
            GraphOp::Elementwise { op, .. } => self.lower_elementwise_kernel(node, node_id, op),
            GraphOp::Reduce { op, axis, .. } => self.lower_reduce_kernel(node, node_id, op, *axis),
            GraphOp::Contiguous { .. } => self.lower_contiguous_kernel(node, node_id),
            GraphOp::FusedElementwise { ops, .. } => {
                self.lower_fused_elementwise_kernel(node, node_id, ops)
            }
            GraphOp::FusedElementwiseReduce {
                elementwise_ops,
                reduce_op,
                axis,
                ..
            } => self.lower_fused_elementwise_reduce_kernel(
                node,
                node_id,
                elementwise_ops,
                reduce_op,
                *axis,
            ),
            GraphOp::FusedReduce { .. } => {
                Err("FusedReduce is not yet supported (requires tuple output)".to_string())
            }
            _ => Err(format!("Unsupported operation: {:?}", node.op)),
        }
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
