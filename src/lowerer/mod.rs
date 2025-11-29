use crate::graph::{Graph, GraphNode, ops::GraphOp};
use crate::opt::graph::SimpleCostEstimator;
use std::collections::{HashMap, HashSet, VecDeque};

// モジュール宣言
// グラフ最適化が必須のため、LoweringSuggesterが対応していないノードのみを処理
mod custom; // Customノード（LoweringSuggesterで生成）
mod fold; // Foldノード（LoweringSuggesterで未対応）
mod utils; // 共通ユーティリティ

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

/// グラフ内にCustom(Program)ノードがあれば、そのProgramを返す
/// KernelMergeSuggesterの出力を検出するために使用
fn find_custom_program(graph: &Graph) -> Option<crate::ast::AstNode> {
    for output in graph.outputs().values() {
        if let GraphOp::Custom { ast } = &output.op
            && matches!(ast, crate::ast::AstNode::Program { .. })
        {
            return Some(ast.clone());
        }
    }
    None
}

/// Viewノードの場合、実際のストレージノードまでトレースバックする
/// Viewノードはメモリアクセスパターンを記述するだけで、バッファーは持たない
fn trace_to_storage_node(node: &GraphNode) -> &GraphNode {
    match &node.op {
        GraphOp::View(_) => {
            if let Some(src) = node.src.first() {
                trace_to_storage_node(src)
            } else {
                node
            }
        }
        _ => node,
    }
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

/// グラフ最適化を実行する
///
/// LoweringSuggesterにより、ほとんどのGraphOpがCustomノードに変換されます。
/// KernelMergeSuggesterは無効にして、Custom(Function)をそのまま使用します。
fn optimize_graph_for_lowering(graph: Graph) -> Graph {
    use crate::backend::pipeline::{SuggesterFlags, optimize_graph_with_history};

    let cost_estimator = SimpleCostEstimator::new();

    // KernelMergeSuggesterを無効にした最適化を実行
    // lowering, fusionのみで、カーネルマージは行わない
    let flags = SuggesterFlags::new(); // include_kernel_merge = false
    let (optimized_graph, _history) = optimize_graph_with_history(
        graph,
        flags,
        cost_estimator,
        10,    // beam_width
        100,   // max_steps
        false, // show_progress
    );

    optimized_graph
}

/// GraphをProgramに変換する公開関数
///
/// グラフ最適化を自動的に実行し、全ノードをカーネル関数に変換します。
/// 中間バッファーは自動的に確保・管理されます。
pub(crate) fn lower(graph: Graph) -> crate::ast::AstNode {
    // グラフ最適化を実行（LoweringSuggesterでCustomノードに変換）
    let optimized_graph = optimize_graph_for_lowering(graph);

    // Custom(Program)ノードがあればそれを直接返す（KernelMergeSuggesterの出力）
    if let Some(program) = find_custom_program(&optimized_graph) {
        log::debug!("Found Custom(Program) node, returning directly");
        return program;
    }

    let mut lowerer = Lowerer::new();

    // トポロジカルソートでノードを取得
    let generations = Lowerer::topological_sort(&optimized_graph);

    // 各ノードの出力バッファー名を追跡（ノードポインタ → バッファー名）
    let mut node_buffer_map: HashMap<*const (), String> = HashMap::new();

    // 入力ノードのバッファー名を設定
    // 決定論的な順序にするため、入力名をアルファベット順にソートする
    let mut sorted_input_names: Vec<_> = optimized_graph.inputs().keys().cloned().collect();
    sorted_input_names.sort();

    // 入力ノードのポインタを名前にマッピング
    let mut input_node_by_name: HashMap<String, *const ()> = HashMap::new();
    for generation in &generations {
        for node in generation {
            if matches!(node.op, GraphOp::Buffer { .. }) {
                // 入力ノードの名前を見つける
                for (name, weak_node) in optimized_graph.inputs().iter() {
                    if let Some(rc_node) = weak_node.upgrade() {
                        let input_node = GraphNode::from_rc(rc_node);
                        if node_ptr(&input_node) == node_ptr(node) {
                            input_node_by_name.insert(name.clone(), node_ptr(node));
                            break;
                        }
                    }
                }
            }
        }
    }

    // ソートされた順序でバッファー名を割り当て
    for (input_counter, name) in sorted_input_names.iter().enumerate() {
        if let Some(&ptr) = input_node_by_name.get(name) {
            let buffer_name = format!("input{}", input_counter);
            log::debug!("Registering input '{}' as {}", name, buffer_name);
            node_buffer_map.insert(ptr, buffer_name);
        }
    }

    // グラフの出力ノードを収集（最終出力として扱う）
    // トポロジカルソートは出力→入力の順序なので、最初の世代が出力ノード
    let final_output_ptrs: HashSet<*const ()> = if !generations.is_empty() {
        generations[0].iter().map(node_ptr).collect()
    } else {
        optimized_graph.outputs().values().map(node_ptr).collect()
    };

    // カーネル情報を収集
    let mut kernel_infos: Vec<KernelInfo> = Vec::new();
    let mut tmp_counter = 0;

    // 各世代の各ノードをカーネル関数に変換
    // トポロジカルソートは出力→入力の順序なので、逆順にする（入力→出力）
    let mut kernel_id = 0;
    for generation in generations.into_iter().rev() {
        for node in generation {
            // Buffer, Const, ComplexConst, View ノードはスキップ
            // Viewノードはメモリアクセスパターンを記述するだけで、バッファーを生成しない
            if matches!(
                node.op,
                GraphOp::Buffer { .. }
                    | GraphOp::Const(_)
                    | GraphOp::ComplexConst { .. }
                    | GraphOp::View(_)
            ) {
                continue;
            }

            // 各カーネル関数は独立しているので、変数カウンターをリセット
            lowerer.alu_counter = 0;
            lowerer.acc_counter = 0;

            // 入力バッファー名を収集
            // Viewノードの場合は実際のストレージノードまでトレースバック
            // 出力Buffer（名前が "output" で始まる）はスキップ
            let input_buffers: Vec<String> = node
                .src
                .iter()
                .filter_map(|src| {
                    // Constノード、ComplexConstノードはバッファーを持たない
                    if matches!(src.op, GraphOp::Const(_) | GraphOp::ComplexConst { .. }) {
                        return None;
                    }
                    // 出力Bufferはスキップ（入力として扱わない）
                    if matches!(&src.op, GraphOp::Buffer { name } if name.starts_with("output")) {
                        return None;
                    }
                    // Viewノードは実際のストレージノードまでトレース
                    let storage_node = trace_to_storage_node(src);
                    let src_ptr = node_ptr(storage_node);
                    let buf = node_buffer_map.get(&src_ptr).cloned();
                    log::debug!(
                        "Kernel {}: src node {:?} -> storage {:?} -> buffer {:?}",
                        kernel_id,
                        src.op,
                        storage_node.op,
                        buf
                    );
                    buf
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

    let main_function =
        generate_main_function_with_intermediates(&kernel_infos, sorted_input_names.len());

    let mut all_functions = functions;
    all_functions.push(main_function);

    // AstNode::Programを作成
    // エントリーポイント名を "harp_main" にしてC言語のmain関数との衝突を避ける
    crate::ast::helper::program(all_functions, "harp_main")
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
    use crate::ast::helper::{assign, block, function, var};
    use crate::ast::{AstNode, DType, Mutability, Scope, VarDecl, VarKind};

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
    let body = block(statements, scope);

    // AstNode::Functionとして返す
    function(Some("harp_main"), params, DType::Tuple(vec![]), body)
}

/// すべてのカーネルを呼び出すmain関数を生成（旧バージョン、互換性のため残す）
#[allow(dead_code)]
fn generate_main_function(
    kernel_functions: &[crate::ast::AstNode],
    kernel_count: usize,
) -> crate::ast::AstNode {
    use crate::ast::helper::{block, function, var};
    use crate::ast::{AstNode, DType, Scope, VarDecl};

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
            let args: Vec<AstNode> = params.iter().map(|p| var(p.name.clone())).collect();

            // Call文を追加
            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });
        }
    }

    // main関数のbodyを作成（Blockノード）
    let body = block(statements, Scope::new());

    // AstNode::Functionとして返す
    function(
        Some("harp_main"),
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
        use crate::ast::DType;
        use crate::ast::helper::{block, function};

        let body = block(body_statements, body_scope);

        function(
            Some(format!("kernel_{}", node_id)),
            params,
            DType::Tuple(vec![]),
            body,
        )
    }

    /// GraphNodeを一つのカーネル関数に変換
    ///
    /// ## アーキテクチャ
    /// グラフ最適化は必須であり、LoweringSuggesterがほとんどのGraphOpを
    /// Customノードに変換します。最適化後のグラフでは、
    /// 以下のノードタイプのみが残ります：
    ///
    /// - **Custom**: LoweringSuggesterによって生成（`lower_custom_function`で処理）
    /// - **Fold**: 複雑なため、LoweringSuggesterでは未対応（`lower_fold_kernel`で処理）
    /// - **FusedReduce**: タプル出力が必要なため、未対応（エラー）
    ///
    /// それ以外のノードタイプは、グラフ最適化によってCustomノードに変換されているべきです。
    pub fn lower_node_to_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<crate::ast::AstNode, String> {
        match &node.op {
            GraphOp::Custom { ast } => {
                // Custom AST（Function または Program）をloweringする
                // プレースホルダー変数を実際のパラメータに置換
                self.lower_custom_ast(node, node_id, ast)
            }
            GraphOp::Fold {
                output_size,
                kernel_size,
                stride,
                dilation,
                groups,
            } => self.lower_fold_kernel(
                node,
                node_id,
                output_size,
                kernel_size,
                stride,
                dilation,
                *groups,
            ),
            GraphOp::FusedReduce { .. } => {
                Err("FusedReduce is not yet supported (requires tuple output)".to_string())
            }
            _ => Err(format!(
                "Unsupported operation: {:?}. Graph optimization should have converted this to Custom node.",
                node.op
            )),
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
