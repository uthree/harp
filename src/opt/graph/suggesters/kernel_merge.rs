//! Kernel Merge Suggester
//!
//! 複数のKernelノード（Function/Program）をペアワイズでマージします。
//! ビームサーチによって最適なマージ順序を探索できます。
//!
//! # サポートするマージパターン
//! - Kernel(Function) + Kernel(Function) → Kernel(Program)
//! - Kernel(Program) + Kernel(Function) → Kernel(Program) (増分マージ)
//! - Kernel(Program) + Kernel(Program) → Kernel(Program) (Program融合)
//!
//! # 実行順序
//! カーネルの実行順序はホスト側（CompiledProgram）で管理されます。
//! AST内にはエントリーポイント関数を生成しません。

use crate::ast::helper::const_int;
use crate::ast::{
    AstKernelCallInfo, AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind,
};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::shape::Expr;
use crate::graph::{DType as GraphDType, Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::{GraphSuggester, SuggestResult};
use std::collections::{HashMap, HashSet};

/// Box<AstNode>からExprへの変換（変換失敗時はConst(1)を返す）
fn ast_to_expr(node: &AstNode) -> Expr {
    Expr::try_from(node).unwrap_or(Expr::Const(1))
}

/// 複数のKernelノードをペアワイズでマージするSuggester
pub struct KernelMergeSuggester;

impl KernelMergeSuggester {
    pub fn new() -> Self {
        Self
    }

    /// ノードが出力 Buffer かどうかを判定
    /// 出力 Buffer は名前が "output" で始まる GraphOp::Buffer
    fn is_output_buffer(node: &GraphNode) -> bool {
        matches!(&node.op, GraphOp::Buffer { name } if name.starts_with("output"))
    }

    /// src から入力ノードのみを取得（出力 Buffer を除外）
    fn get_input_nodes(src: &[GraphNode]) -> Vec<&GraphNode> {
        src.iter().filter(|n| !Self::is_output_buffer(n)).collect()
    }

    /// グラフ内の全ノードを収集（トポロジカル順）
    fn collect_all_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const GraphNodeData>,
            nodes: &mut Vec<GraphNode>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in &node.src {
                visit(src, visited, nodes);
            }

            nodes.push(node.clone());
        }

        for output in graph.outputs().values() {
            visit(output, &mut visited, &mut nodes);
        }

        nodes
    }

    /// グラフ内のKernelノードを収集
    fn collect_custom_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        self.collect_all_nodes(graph)
            .into_iter()
            .filter(|node| matches!(&node.op, GraphOp::Kernel { .. }))
            .collect()
    }

    /// Viewノードをトレースバックして、実際のストレージノードを取得
    ///
    /// Viewノードはメモリアクセスパターンを記述するだけで、
    /// バッファーを持たないため、実際のストレージノードまでトレースバックする
    fn trace_to_storage_node(node: &GraphNode) -> &GraphNode {
        match &node.op {
            GraphOp::View(_) => {
                if let Some(src) = node.src.first() {
                    Self::trace_to_storage_node(src)
                } else {
                    node
                }
            }
            _ => node,
        }
    }

    /// マージ可能なKernelノードのペアを検出
    ///
    /// 親ノード（consumer）と子ノード（producer）の関係にあるペアを返す
    /// Viewノードが間に挟まっている場合もトレースバックして検出する
    fn find_mergeable_pairs(&self, graph: &Graph) -> Vec<(GraphNode, GraphNode)> {
        let custom_nodes = self.collect_custom_nodes(graph);
        let mut pairs = Vec::new();

        // 各Kernelノードの参照カウントを計算
        let ref_counts = self.count_node_references(graph);

        for consumer in &custom_nodes {
            // 出力 Buffer を除外した入力ノードのみを処理
            let input_nodes = Self::get_input_nodes(&consumer.src);
            for producer_or_view in input_nodes {
                // Viewノードをトレースバックして実際のストレージノードを取得
                let producer = Self::trace_to_storage_node(producer_or_view);

                // producerがKernelノードかチェック
                if !matches!(&producer.op, GraphOp::Kernel { .. }) {
                    continue;
                }

                // producerが他のノードからも参照されている場合はマージしない
                // （出力が複数箇所で使われる場合、マージすると計算が重複する）
                let producer_ptr = producer.as_ptr();
                let ref_count = ref_counts.get(&producer_ptr).copied().unwrap_or(0);
                if ref_count > 1 {
                    continue;
                }

                pairs.push((consumer.clone(), producer.clone()));
            }
        }

        pairs
    }

    /// グラフ内の各ノードの被参照数をカウント
    ///
    /// Viewノードを透過的に扱い、実際のストレージノード（Kernel等）の
    /// 被参照数をカウントします。
    fn count_node_references(&self, graph: &Graph) -> HashMap<*const GraphNodeData, usize> {
        let mut ref_counts: HashMap<*const GraphNodeData, usize> = HashMap::new();
        let mut visited = HashSet::new();

        fn visit(
            node: &GraphNode,
            ref_counts: &mut HashMap<*const GraphNodeData, usize>,
            visited: &mut HashSet<*const GraphNodeData>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in &node.src {
                // Viewノードをトレースバックして実際のストレージノードの参照をカウント
                let storage_node = KernelMergeSuggester::trace_to_storage_node(src);
                let storage_ptr = storage_node.as_ptr();
                *ref_counts.entry(storage_ptr).or_insert(0) += 1;

                // 再帰的に訪問
                visit(src, ref_counts, visited);
            }
        }

        for output in graph.outputs().values() {
            visit(output, &mut ref_counts, &mut visited);
        }

        ref_counts
    }

    /// 2つのKernelノードを1つのKernel(Program)にマージ
    fn merge_pair(
        &self,
        graph: &Graph,
        consumer: &GraphNode,
        producer: &GraphNode,
    ) -> Option<Graph> {
        // producer と consumer のAST情報を取得
        let (producer_ast, consumer_ast) = match (&producer.op, &consumer.op) {
            (GraphOp::Kernel { ast: p_ast, .. }, GraphOp::Kernel { ast: c_ast, .. }) => {
                (p_ast, c_ast)
            }
            _ => return None,
        };

        // 各ASTからカーネル関数を抽出/作成
        let mut kernels: Vec<AstNode> = Vec::new();
        let mut used_names: HashSet<String> = HashSet::new();

        // Producer側のカーネルを追加
        let producer_kernels =
            self.extract_or_create_kernels(producer, producer_ast, &mut used_names);
        kernels.extend(producer_kernels);

        // Consumer側のカーネルを追加
        let consumer_kernels =
            self.extract_or_create_kernels(consumer, consumer_ast, &mut used_names);
        kernels.extend(consumer_kernels);

        // 実行波情報を生成
        // Producer側のカーネルをまず追加
        let mut execution_waves: Vec<Vec<AstKernelCallInfo>> =
            Self::extract_execution_waves(producer_ast);

        // Consumer側のカーネルを追加（新しいwaveとして）
        let consumer_waves = Self::extract_execution_waves(consumer_ast);
        execution_waves.extend(consumer_waves);

        // カーネル名の重複を修正（used_namesに基づいて更新）
        // 注意: extract_or_create_kernelsで名前が変更されている可能性があるため、
        // 実際のカーネル名と一致させる
        Self::update_execution_waves_names(&mut execution_waves, &kernels);

        // Programを作成（harp_mainは生成しない - 実行順序はCompiledProgramで管理）
        let program = AstNode::Program {
            functions: kernels,
            execution_waves,
        };

        // 新しいKernelノードの入力を構築
        // producer の入力 + consumer の入力（producerと出力Bufferを除く）
        // 出力 Buffer は最後に追加する
        let producer_inputs = Self::get_input_nodes(&producer.src);
        let mut new_inputs: Vec<GraphNode> = producer_inputs.iter().map(|n| (*n).clone()).collect();
        for src in &consumer.src {
            // producer と出力 Buffer は除外
            if src.as_ptr() != producer.as_ptr() && !Self::is_output_buffer(src) {
                new_inputs.push(src.clone());
            }
        }
        // 出力 Buffer を追加（consumer の出力 Buffer を使用）
        for src in &consumer.src {
            if Self::is_output_buffer(src) {
                new_inputs.push(src.clone());
                break;
            }
        }

        // 新しいKernelノードを作成
        let merged_node = GraphNode::new(
            consumer.dtype.clone(),
            GraphOp::Kernel {
                ast: program,
                input_buffers: None,
            },
            new_inputs,
            consumer.view.clone(),
        );

        // グラフを再構築（consumerをmerged_nodeに置き換え）
        let new_graph = self.replace_node_in_graph(graph, consumer, merged_node);

        Some(new_graph)
    }

    /// ASTからカーネル関数を抽出、または新規作成
    ///
    /// 元のFunction/Kernel名を保持し、重複時は`__n`を追加
    fn extract_or_create_kernels(
        &self,
        node: &GraphNode,
        ast: &AstNode,
        used_names: &mut HashSet<String>,
    ) -> Vec<AstNode> {
        match ast {
            AstNode::Function { .. } => {
                // Kernel(Function) → Kernelに変換、元の名前を保持
                let kernel = self.create_kernel_from_function(node, ast, used_names);
                if let AstNode::Kernel { name: Some(n), .. } = &kernel {
                    used_names.insert(n.clone());
                }
                vec![kernel]
            }
            AstNode::Program { functions, .. } => {
                // Kernel(Program) → 既存のKernelを抽出（main以外）
                let mut kernels = Vec::new();
                for func in functions {
                    match func {
                        AstNode::Kernel {
                            name,
                            params,
                            return_type,
                            body,
                            default_grid_size,
                            default_thread_group_size,
                        } => {
                            // 元の名前を保持、重複時は__nを追加
                            let base_name = name.clone().unwrap_or_else(|| "kernel".to_string());
                            let new_name = Self::make_unique_name(&base_name, used_names);
                            used_names.insert(new_name.clone());
                            kernels.push(AstNode::Kernel {
                                name: Some(new_name),
                                params: params.clone(),
                                return_type: return_type.clone(),
                                body: body.clone(),
                                default_grid_size: default_grid_size.clone(),
                                default_thread_group_size: default_thread_group_size.clone(),
                            });
                        }
                        AstNode::Function {
                            name,
                            params,
                            return_type,
                            body,
                        } => {
                            // FunctionをKernelに変換
                            let base_name = name.clone().unwrap_or_else(|| "kernel".to_string());
                            let new_name = Self::make_unique_name(&base_name, used_names);
                            used_names.insert(new_name.clone());
                            let one = const_int(1);
                            kernels.push(AstNode::Kernel {
                                name: Some(new_name),
                                params: params.clone(),
                                return_type: return_type.clone(),
                                body: body.clone(),
                                default_grid_size: [
                                    Box::new(one.clone()),
                                    Box::new(one.clone()),
                                    Box::new(one.clone()),
                                ],
                                default_thread_group_size: [
                                    Box::new(const_int(64)),
                                    Box::new(one.clone()),
                                    Box::new(one),
                                ],
                            });
                        }
                        _ => {}
                    }
                }
                kernels
            }
            _ => vec![],
        }
    }

    /// 重複を避けてユニークな名前を生成
    ///
    /// base_nameが使用済みの場合、`__1`, `__2`, ... を追加
    fn make_unique_name(base_name: &str, used_names: &HashSet<String>) -> String {
        if !used_names.contains(base_name) {
            return base_name.to_string();
        }

        let mut counter = 1;
        loop {
            let candidate = format!("{}__{}", base_name, counter);
            if !used_names.contains(&candidate) {
                return candidate;
            }
            counter += 1;
        }
    }

    /// FunctionからKernelを作成
    ///
    /// 元のFunction名を保持し、重複時は`__n`を追加
    fn create_kernel_from_function(
        &self,
        node: &GraphNode,
        func_ast: &AstNode,
        used_names: &HashSet<String>,
    ) -> AstNode {
        // 入力ノードのみを取得（出力 Buffer を除外）
        let input_nodes = Self::get_input_nodes(&node.src);

        let input_shape = if !input_nodes.is_empty() {
            input_nodes[0].view.shape().to_vec()
        } else {
            node.view.shape().to_vec()
        };

        // パラメータを生成
        let mut params = Vec::new();

        // 入力バッファー（出力 Buffer を除外）
        for (i, src) in input_nodes.iter().enumerate() {
            params.push(VarDecl {
                name: ph::input(i),
                dtype: Self::graph_dtype_to_ast_ptr(&src.dtype),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
        }

        // 出力バッファー
        params.push(VarDecl {
            name: ph::OUTPUT.to_string(),
            dtype: Self::graph_dtype_to_ast_ptr(&node.dtype),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

        // Shape変数
        for expr in input_shape.iter() {
            if let crate::graph::shape::Expr::Var(var_name) = expr {
                params.push(VarDecl {
                    name: var_name.clone(),
                    dtype: AstDType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                });
            }
        }

        // ボディを取得
        let body = if let AstNode::Function { body, .. } = func_ast {
            let mut shape_substitutions: HashMap<String, AstNode> = HashMap::new();
            for (axis, expr) in input_shape.iter().enumerate() {
                let placeholder_name = ph::shape(axis);
                let ast_expr: AstNode = expr.clone().into();
                shape_substitutions.insert(placeholder_name, ast_expr);
            }
            body.substitute_vars(&shape_substitutions)
        } else {
            AstNode::Block {
                statements: vec![],
                scope: Box::new(Scope::new()),
            }
        };

        // 元のFunction名を取得し、重複を避けてユニークな名前を生成
        let base_name = if let AstNode::Function { name: Some(n), .. } = func_ast {
            n.clone()
        } else {
            "kernel".to_string()
        };
        let kernel_name = Self::make_unique_name(&base_name, used_names);

        // デフォルト1D dispatch設定
        let one = const_int(1);
        AstNode::Kernel {
            name: Some(kernel_name),
            params,
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(body),
            default_grid_size: [
                Box::new(one.clone()),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(64)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        }
    }

    /// グラフ内の特定ノードを置き換えた新しいグラフを作成
    fn replace_node_in_graph(
        &self,
        graph: &Graph,
        old_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
        node_map.insert(old_node.as_ptr(), new_node.clone());

        let mut cache: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            cache: &mut HashMap<*const GraphNodeData, GraphNode>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // Inputノードは常に元のノードをそのまま返す
            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            // キャッシュを確認（再構築済みノードを返す）
            if let Some(cached) = cache.get(&ptr) {
                return cached.clone();
            }

            let new_src: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, cache))
                .collect();

            let src_changed = new_src
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            let result = if !src_changed {
                node.clone()
            } else {
                GraphNode::new(
                    node.dtype.clone(),
                    node.op.clone(),
                    new_src,
                    node.view.clone(),
                )
            };

            cache.insert(ptr, result.clone());
            result
        }

        let mut new_graph = Graph::new();

        // 入力・出力メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // 全ての出力ノードを再構築
        for (name, output_node) in graph.outputs() {
            let rebuilt = rebuild_node(output_node, &node_map, &mut cache);
            new_graph.set_output_node(name.clone(), rebuilt);
        }

        // shape変数のデフォルト値をコピー
        for (name, value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(name.clone(), *value);
        }

        new_graph
    }

    fn graph_dtype_to_ast(dtype: &GraphDType) -> AstDType {
        match dtype {
            GraphDType::Bool => AstDType::Bool,
            GraphDType::I32 => AstDType::Int,
            GraphDType::F32 => AstDType::F32,
            GraphDType::Unknown => AstDType::F32,
        }
    }

    fn graph_dtype_to_ast_ptr(dtype: &GraphDType) -> AstDType {
        AstDType::Ptr(Box::new(Self::graph_dtype_to_ast(dtype)))
    }

    /// ASTから実行wave情報を抽出
    ///
    /// KernelノードまたはProgram内のKernelから、
    /// 入出力バッファ情報を持つAstKernelCallInfoのwave構造を生成します。
    fn extract_execution_waves(ast: &AstNode) -> Vec<Vec<AstKernelCallInfo>> {
        match ast {
            AstNode::Kernel {
                name,
                params,
                default_grid_size,
                default_thread_group_size,
                ..
            } => {
                let kernel_name = name.clone().unwrap_or_else(|| "unnamed_kernel".to_string());
                let (inputs, outputs) = Self::extract_io_from_params(params);
                let call_info = AstKernelCallInfo::new(
                    kernel_name,
                    inputs,
                    outputs,
                    [
                        ast_to_expr(&default_grid_size[0]),
                        ast_to_expr(&default_grid_size[1]),
                        ast_to_expr(&default_grid_size[2]),
                    ],
                    [
                        ast_to_expr(&default_thread_group_size[0]),
                        ast_to_expr(&default_thread_group_size[1]),
                        ast_to_expr(&default_thread_group_size[2]),
                    ],
                );
                vec![vec![call_info]]
            }
            AstNode::Function { name, params, .. } => {
                // FunctionはKernelに変換される前提（デフォルトdispatchサイズを使用）
                let kernel_name = name.clone().unwrap_or_else(|| "unnamed_kernel".to_string());
                let (inputs, outputs) = Self::extract_io_from_params(params);
                let call_info =
                    AstKernelCallInfo::with_default_dispatch(kernel_name, inputs, outputs);
                vec![vec![call_info]]
            }
            AstNode::Program {
                functions,
                execution_waves,
            } => {
                // 既存のexecution_wavesがあればそのまま返す
                if !execution_waves.is_empty() {
                    execution_waves.clone()
                } else {
                    // execution_wavesがない場合は、各Kernelを順番に個別のwaveとして処理
                    let mut waves = Vec::new();
                    for func in functions {
                        match func {
                            AstNode::Kernel {
                                name,
                                params,
                                default_grid_size,
                                default_thread_group_size,
                                ..
                            } => {
                                let kernel_name =
                                    name.clone().unwrap_or_else(|| "unnamed_kernel".to_string());
                                let (inputs, outputs) = Self::extract_io_from_params(params);
                                let call_info = AstKernelCallInfo::new(
                                    kernel_name,
                                    inputs,
                                    outputs,
                                    [
                                        ast_to_expr(&default_grid_size[0]),
                                        ast_to_expr(&default_grid_size[1]),
                                        ast_to_expr(&default_grid_size[2]),
                                    ],
                                    [
                                        ast_to_expr(&default_thread_group_size[0]),
                                        ast_to_expr(&default_thread_group_size[1]),
                                        ast_to_expr(&default_thread_group_size[2]),
                                    ],
                                );
                                waves.push(vec![call_info]);
                            }
                            AstNode::Function { name, params, .. } => {
                                let kernel_name =
                                    name.clone().unwrap_or_else(|| "unnamed_kernel".to_string());
                                let (inputs, outputs) = Self::extract_io_from_params(params);
                                let call_info = AstKernelCallInfo::with_default_dispatch(
                                    kernel_name,
                                    inputs,
                                    outputs,
                                );
                                waves.push(vec![call_info]);
                            }
                            _ => {}
                        }
                    }
                    waves
                }
            }
            _ => vec![],
        }
    }

    /// パラメータリストから入力・出力バッファ名を抽出
    fn extract_io_from_params(params: &[VarDecl]) -> (Vec<String>, Vec<String>) {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for param in params {
            // Ptr型のパラメータのみをバッファとして扱う
            if matches!(param.dtype, AstDType::Ptr(_)) {
                if param.mutability == Mutability::Mutable {
                    outputs.push(param.name.clone());
                } else {
                    inputs.push(param.name.clone());
                }
            }
        }

        (inputs, outputs)
    }

    /// execution_wavesのカーネル名を実際のカーネル名と一致させる
    ///
    /// extract_or_create_kernelsで重複を避けるために名前が変更されている可能性があるため、
    /// 実際のカーネル名と順番で対応付けて更新します。
    fn update_execution_waves_names(
        execution_waves: &mut [Vec<AstKernelCallInfo>],
        kernels: &[AstNode],
    ) {
        // カーネルリストから名前を抽出
        let kernel_names: Vec<String> = kernels
            .iter()
            .filter_map(|k| match k {
                AstNode::Kernel { name, .. } => name.clone(),
                AstNode::Function { name, .. } => name.clone(),
                _ => None,
            })
            .collect();

        // execution_waves内の名前を更新（フラット化して順番で対応付け）
        let mut name_idx = 0;
        for wave in execution_waves.iter_mut() {
            for call in wave.iter_mut() {
                if name_idx < kernel_names.len() {
                    call.kernel_name = kernel_names[name_idx].clone();
                    name_idx += 1;
                }
            }
        }
    }

    /// グラフ内のKernelノードの統計を取得（テスト用）
    #[cfg(test)]
    pub fn count_custom_nodes(&self, graph: &Graph) -> (usize, usize) {
        let nodes = self.collect_all_nodes(graph);
        let mut function_count = 0;
        let mut program_count = 0;

        for node in &nodes {
            if let GraphOp::Kernel { ast, .. } = &node.op {
                match ast {
                    AstNode::Function { .. } => function_count += 1,
                    AstNode::Program { .. } => program_count += 1,
                    _ => {}
                }
            }
        }

        (function_count, program_count)
    }
}

impl Default for KernelMergeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for KernelMergeSuggester {
    fn name(&self) -> &'static str {
        "KernelMerge"
    }

    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult> {
        let pairs = self.find_mergeable_pairs(graph);

        log::debug!(
            "KernelMergeSuggester: found {} mergeable pairs",
            pairs.len()
        );

        let mut suggestions = Vec::new();

        for (consumer, producer) in pairs {
            if let Some(merged_graph) = self.merge_pair(graph, &consumer, &producer) {
                log::debug!("KernelMergeSuggester: merged pair (consumer -> producer)");
                suggestions.push(SuggestResult::new(merged_graph, self.name()));
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    #[test]
    fn test_kernel_merge_pairwise_basic() {
        use crate::ast::helper::wildcard;

        // a + b -> custom1, custom1 * c -> custom2
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        let custom1 = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let custom2 = custom1.custom_elementwise_binary(c, wildcard("0") * wildcard("1"));
        graph.output("result", custom2);

        let suggester = KernelMergeSuggester::new();

        // マージ可能なペアを検出
        let pairs = suggester.find_mergeable_pairs(&graph);
        assert_eq!(pairs.len(), 1, "Should find exactly 1 mergeable pair");

        // マージを実行
        let suggestions = suggester.suggest(&graph);
        assert_eq!(suggestions.len(), 1, "Should produce 1 suggestion");

        // マージ後のグラフを確認
        let merged = &suggestions[0].graph;
        let (fn_count, prog_count) = suggester.count_custom_nodes(merged);

        assert_eq!(prog_count, 1, "Should have 1 Kernel(Program)");
        assert_eq!(fn_count, 0, "Should have 0 Kernel(Function)");
    }

    #[test]
    fn test_kernel_merge_pairwise_chain() {
        use crate::ast::helper::wildcard;

        // 3つのKernelノードのチェーン: a -> custom1 -> custom2 -> custom3
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);
        let d = graph.input("d", DType::F32, vec![10]);

        let custom1 = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let custom2 = custom1.custom_elementwise_binary(c, wildcard("0") * wildcard("1"));
        let custom3 = custom2.custom_elementwise_binary(d, wildcard("0") + wildcard("1"));
        graph.output("result", custom3);

        let suggester = KernelMergeSuggester::new();

        // 最初のsuggestで2つのペアが見つかるはず
        let pairs = suggester.find_mergeable_pairs(&graph);
        assert_eq!(
            pairs.len(),
            2,
            "Should find 2 mergeable pairs (custom3-custom2, custom2-custom1)"
        );

        // 1回目のマージ
        let suggestions = suggester.suggest(&graph);
        assert_eq!(suggestions.len(), 2, "Should produce 2 suggestions");

        // 1つ目の提案でマージ後、さらにマージ可能
        let merged_once = &suggestions[0].graph;
        let pairs_after = suggester.find_mergeable_pairs(merged_once);

        // マージ後も1つのペアが残っているはず
        println!("After first merge: {} pairs remaining", pairs_after.len());
        assert!(
            pairs_after.len() <= 1,
            "Should have at most 1 pair after first merge"
        );
    }

    // Note: test_kernel_merge_no_merge_multiple_refs は複数出力が
    // 現在サポートされていないため削除されました。

    #[test]
    fn test_kernel_merge_creates_program_with_kernels() {
        use crate::ast::helper::wildcard;

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        let custom1 = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let custom2 = custom1.custom_elementwise_binary(c, wildcard("0") * wildcard("1"));
        graph.output("result", custom2);

        let suggester = KernelMergeSuggester::new();
        let suggestions = suggester.suggest(&graph);

        assert!(!suggestions.is_empty());

        // Programを検査して複数のKernelが含まれることを確認
        let merged = &suggestions[0].graph;
        if let Some(output) = merged.outputs().values().next()
            && let GraphOp::Kernel {
                ast: AstNode::Program { functions, .. },
                ..
            } = &output.op
        {
            // 全ての関数がKernelであることを確認（harp_mainは存在しない）
            let kernel_count = functions
                .iter()
                .filter(|f| matches!(f, AstNode::Kernel { .. }))
                .count();

            assert!(
                kernel_count >= 2,
                "Should have at least 2 kernels, got {}",
                kernel_count
            );

            // harp_mainが存在しないことを確認
            let has_main = functions
                .iter()
                .any(|f| matches!(f, AstNode::Function { name: Some(n), .. } if n == "harp_main"));
            assert!(!has_main, "Should not have harp_main function");
        }
    }

    #[test]
    fn test_kernel_merge_with_optimizer() {
        use crate::backend::pipeline::{MultiPhaseConfig, create_multi_phase_optimizer};
        use crate::opt::graph::GraphOptimizer;

        let _ = env_logger::try_init();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        let sum = &a + &b;
        let prod = &sum * &c;
        let result = &prod + 1.0f32;
        graph.output("result", result);

        let config = MultiPhaseConfig::new()
            .with_beam_width(4)
            .with_max_steps(100)
            .with_progress(false)
            .with_collect_logs(false);
        let optimizer = create_multi_phase_optimizer(config);

        let optimized = optimizer.optimize(graph);

        let mut custom_function_count = 0;
        let mut custom_program_count = 0;

        fn count_customs(
            node: &GraphNode,
            visited: &mut HashSet<*const GraphNodeData>,
            fn_count: &mut usize,
            prog_count: &mut usize,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in &node.src {
                count_customs(src, visited, fn_count, prog_count);
            }

            if let GraphOp::Kernel { ast, .. } = &node.op {
                match ast {
                    AstNode::Function { .. } => *fn_count += 1,
                    AstNode::Program { .. } => *prog_count += 1,
                    _ => {}
                }
            }
        }

        let mut visited = HashSet::new();
        for output in optimized.outputs().values() {
            count_customs(
                output,
                &mut visited,
                &mut custom_function_count,
                &mut custom_program_count,
            );
        }

        println!("Kernel(Function): {}", custom_function_count);
        println!("Kernel(Program): {}", custom_program_count);

        assert!(
            custom_function_count == 1 || custom_program_count >= 1,
            "Expected either 1 Kernel(Function) after fusion or 1 Kernel(Program) after merge"
        );
    }

    #[test]
    fn test_kernel_merge_generates_execution_waves() {
        use crate::ast::helper::wildcard;

        // 2つのKernelをマージして、execution_wavesが正しく生成されることを確認
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        let custom1 = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let custom2 = custom1.custom_elementwise_binary(c, wildcard("0") * wildcard("1"));
        graph.output("result", custom2);

        let suggester = KernelMergeSuggester::new();
        let suggestions = suggester.suggest(&graph);

        assert!(!suggestions.is_empty());

        // execution_wavesが生成されていることを確認
        let merged = &suggestions[0].graph;
        if let Some(output) = merged.outputs().values().next()
            && let GraphOp::Kernel {
                ast:
                    AstNode::Program {
                        functions,
                        execution_waves,
                    },
                ..
            } = &output.op
        {
            // execution_wavesが生成されていることを確認
            assert!(
                !execution_waves.is_empty(),
                "execution_waves should be generated"
            );

            // カーネルと同じ数のwave（各waveに1つのカーネル）が存在することを確認
            let total_calls: usize = execution_waves.iter().map(|w| w.len()).sum();
            assert_eq!(
                total_calls,
                functions.len(),
                "execution_waves should have same number of entries as kernels"
            );

            // Producer (wave 0) が Consumer (wave 1) より前にあることを確認
            assert!(execution_waves.len() >= 2, "Should have at least 2 waves");

            println!("execution_waves: {:?}", execution_waves);
        } else {
            panic!("Expected Kernel(Program) with execution_waves");
        }
    }
}
