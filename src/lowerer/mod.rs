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

/// カーネル関数の情報（main関数生成用）
struct KernelInfo {
    /// カーネル関数のAST
    function: crate::ast::AstNode,
    /// 入力バッファー名のリスト（input0, input1, tmp0 など）
    input_buffers: Vec<String>,
    /// 出力バッファー名（output, tmp0 など）
    output_buffer: String,
    /// 出力の型
    output_dtype: crate::ast::DType,
    /// 出力のサイズ（要素数）を計算する式
    output_size: crate::ast::AstNode,
}

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
/// 中間バッファーは自動的に確保・管理されます。
pub(crate) fn lower(graph: Graph) -> crate::ast::AstNode {
    let mut lowerer = Lowerer::new();

    // トポロジカルソートでノードを取得
    let generations = Lowerer::topological_sort(&graph);

    // 各ノードの出力バッファー名を追跡（ノードポインタ → バッファー名）
    let mut node_buffer_map: HashMap<*const (), String> = HashMap::new();

    // 入力ノードのバッファー名を設定
    // 注: グラフ最適化後はgraph.inputs()のWeakリファレンスと実際のノードのポインタが
    // 異なる場合があるため、トポロジカルソートから収集されたノードを使用する
    let mut input_counter = 0;
    for generation in &generations {
        for node in generation {
            if matches!(node.op, GraphOp::Input) {
                let buffer_name = format!("input{}", input_counter);
                let ptr = node_ptr(node);
                log::debug!("Registering input node as {}", buffer_name);
                node_buffer_map.insert(ptr, buffer_name);
                input_counter += 1;
            }
        }
    }

    // グラフの出力ノードを収集（最終出力として扱う）
    // トポロジカルソートは出力→入力の順序なので、最初の世代が出力ノード
    let final_output_ptrs: HashSet<*const ()> = if !generations.is_empty() {
        generations[0].iter().map(node_ptr).collect()
    } else {
        graph.outputs().values().map(node_ptr).collect()
    };

    // カーネル情報を収集
    let mut kernel_infos: Vec<KernelInfo> = Vec::new();
    let mut tmp_counter = 0;

    // 各世代の各ノードをカーネル関数に変換
    // トポロジカルソートは出力→入力の順序なので、逆順にする（入力→出力）
    let mut kernel_id = 0;
    for generation in generations.into_iter().rev() {
        for node in generation {
            // Input と Const ノードはスキップ
            if matches!(node.op, GraphOp::Input | GraphOp::Const(_)) {
                continue;
            }

            // 各カーネル関数は独立しているので、変数カウンターをリセット
            lowerer.alu_counter = 0;
            lowerer.acc_counter = 0;

            // 入力バッファー名を収集
            let input_buffers: Vec<String> = node
                .src
                .iter()
                .filter_map(|src| {
                    // Constノードはバッファーを持たない
                    if matches!(src.op, GraphOp::Const(_)) {
                        None
                    } else {
                        let src_ptr = node_ptr(src);
                        let buf = node_buffer_map.get(&src_ptr).cloned();
                        log::debug!(
                            "Kernel {}: src node {:?} -> buffer {:?}",
                            kernel_id,
                            src.op,
                            buf
                        );
                        buf
                    }
                })
                .collect();

            // 出力バッファー名を決定
            let output_buffer = if final_output_ptrs.contains(&node_ptr(&node)) {
                "output".to_string()
            } else {
                let name = format!("tmp{}", tmp_counter);
                tmp_counter += 1;
                name
            };

            log::debug!(
                "Kernel {}: input_buffers = {:?}, output_buffer = {}",
                kernel_id,
                input_buffers,
                output_buffer
            );

            // このノードの出力バッファー名を記録
            node_buffer_map.insert(node_ptr(&node), output_buffer.clone());

            // 出力サイズを計算
            let output_size = compute_buffer_size(&node);
            let output_dtype = lowerer
                .graph_dtype_to_ast(&node.dtype)
                .unwrap_or(crate::ast::DType::Ptr(Box::new(crate::ast::DType::F32)));

            // カーネル関数を生成
            if let Ok(function) = lowerer.lower_node_to_kernel(&node, kernel_id) {
                kernel_infos.push(KernelInfo {
                    function,
                    input_buffers,
                    output_buffer,
                    output_dtype,
                    output_size,
                });
                kernel_id += 1;
            }
        }
    }

    // main関数を生成
    let functions: Vec<crate::ast::AstNode> = kernel_infos
        .iter()
        .map(|info| info.function.clone())
        .collect();

    let main_function = generate_main_function_with_intermediates(&kernel_infos, input_counter);

    let mut all_functions = functions;
    all_functions.push(main_function);

    // AstNode::Programを作成
    crate::ast::helper::program(all_functions, "main")
}

/// ノードの出力サイズを計算する式を生成
fn compute_buffer_size(node: &GraphNode) -> crate::ast::AstNode {
    use crate::ast::helper::const_int;

    let shape = node.view.shape();
    if shape.is_empty() {
        return const_int(1);
    }

    // 全要素数 = product of all dimensions
    let mut size: crate::ast::AstNode = shape[0].clone().into();
    for dim in &shape[1..] {
        let dim_ast: crate::ast::AstNode = dim.clone().into();
        size = size * dim_ast;
    }
    size
}

/// 中間バッファーを含むmain関数を生成
fn generate_main_function_with_intermediates(
    kernel_infos: &[KernelInfo],
    input_count: usize,
) -> crate::ast::AstNode {
    use crate::ast::helper::{assign, var};
    use crate::ast::{AstNode, DType, FunctionKind, Mutability, Scope, VarDecl, VarKind};

    // main関数のパラメータを収集
    let mut params: Vec<VarDecl> = Vec::new();
    let mut param_names: HashSet<String> = HashSet::new();

    // 1. 入力バッファーをパラメータとして追加
    for i in 0..input_count {
        let input_name = format!("input{}", i);
        if !param_names.contains(&input_name) {
            // 型はカーネル情報から推測（最初に使われているカーネルから）
            let dtype = kernel_infos
                .iter()
                .find(|info| info.input_buffers.contains(&input_name))
                .and_then(|info| {
                    info.input_buffers
                        .iter()
                        .position(|b| b == &input_name)
                        .and_then(|_| {
                            // カーネル関数のパラメータから型を取得
                            if let AstNode::Function { params, .. } = &info.function {
                                params
                                    .iter()
                                    .find(|p| p.name.starts_with("input"))
                                    .map(|p| p.dtype.clone())
                            } else {
                                None
                            }
                        })
                })
                .unwrap_or(DType::Ptr(Box::new(DType::F32)));

            params.push(VarDecl {
                name: input_name.clone(),
                dtype,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
            param_names.insert(input_name);
        }
    }

    // 2. 出力バッファーをパラメータとして追加
    let output_dtype = kernel_infos
        .iter()
        .find(|info| info.output_buffer == "output")
        .map(|info| DType::Ptr(Box::new(info.output_dtype.clone())))
        .unwrap_or(DType::Ptr(Box::new(DType::F32)));

    params.push(VarDecl {
        name: "output".to_string(),
        dtype: output_dtype,
        mutability: Mutability::Mutable,
        kind: VarKind::Normal,
    });

    // 3. Shape変数をパラメータとして追加（全カーネルから収集）
    for info in kernel_infos {
        if let AstNode::Function {
            params: kernel_params,
            ..
        } = &info.function
        {
            for param in kernel_params {
                // shape変数（Int型）のみ追加
                if param.dtype == DType::Int && !param_names.contains(&param.name) {
                    params.push(param.clone());
                    param_names.insert(param.name.clone());
                }
            }
        }
    }

    // main関数のbody文を生成
    let mut statements: Vec<AstNode> = Vec::new();
    let mut scope = Scope::new();

    // 4. 中間バッファーを宣言・確保
    let mut allocated_buffers: HashSet<String> = HashSet::new();
    for info in kernel_infos {
        if info.output_buffer.starts_with("tmp") && !allocated_buffers.contains(&info.output_buffer)
        {
            let ptr_dtype = DType::Ptr(Box::new(info.output_dtype.clone()));

            // 変数宣言をスコープに追加
            if scope
                .declare(
                    info.output_buffer.clone(),
                    ptr_dtype.clone(),
                    Mutability::Mutable,
                )
                .is_ok()
            {
                // malloc相当の処理（Allocate文として表現）
                // 注: ASTにAllocate文がない場合は、Cast(malloc(...))として表現
                let alloc_expr = AstNode::Allocate {
                    dtype: Box::new(info.output_dtype.clone()),
                    size: Box::new(info.output_size.clone()),
                };
                statements.push(assign(&info.output_buffer, alloc_expr));
                allocated_buffers.insert(info.output_buffer.clone());
            }
        }
    }

    // 5. 各カーネルを呼び出す
    for (i, info) in kernel_infos.iter().enumerate() {
        let kernel_name = format!("kernel_{}", i);

        // カーネル関数のパラメータに対応する引数を構築
        if let AstNode::Function {
            params: kernel_params,
            ..
        } = &info.function
        {
            let mut args: Vec<AstNode> = Vec::new();
            let mut input_idx = 0;

            for param in kernel_params {
                if param.name.starts_with("input") {
                    // 入力バッファー → 実際のバッファー名に置換
                    if input_idx < info.input_buffers.len() {
                        args.push(var(&info.input_buffers[input_idx]));
                        input_idx += 1;
                    }
                } else if param.name == "output" {
                    // 出力バッファー → 実際のバッファー名に置換
                    args.push(var(&info.output_buffer));
                } else {
                    // その他のパラメータ（shape変数など）はそのまま
                    args.push(var(&param.name));
                }
            }

            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });
        }
    }

    // 6. 中間バッファーを解放（Deallocate文）
    for buffer_name in &allocated_buffers {
        statements.push(AstNode::Deallocate {
            ptr: Box::new(var(buffer_name)),
        });
    }

    // main関数のbodyを作成
    let body = AstNode::Block {
        statements,
        scope: Box::new(scope),
    };

    // AstNode::Functionとして返す
    crate::ast::helper::function(
        Some("main"),
        FunctionKind::Normal,
        params,
        DType::Tuple(vec![]), // void
        body,
    )
}

/// すべてのカーネルを呼び出すmain関数を生成（旧バージョン、互換性のため残す）
#[allow(dead_code)]
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
