//! Kernel Merge Suggester
//!
//! 複数のCustomノード（Function/Program）をペアワイズでマージします。
//! ビームサーチによって最適なマージ順序を探索できます。
//!
//! # サポートするマージパターン
//! - Custom(Function) + Custom(Function) → Custom(Program)
//! - Custom(Program) + Custom(Function) → Custom(Program) (増分マージ)
//! - Custom(Program) + Custom(Program) → Custom(Program) (Program融合)
//!
//! # バリア挿入
//! カーネル呼び出し間には `AstNode::Barrier` を挿入して、
//! メモリ書き込みの完了を保証します。

use crate::ast::helper::{assign, block, function, var};
use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{DType as GraphDType, Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// 複数のCustomノードをペアワイズでマージするSuggester
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

    /// グラフ内のCustomノードを収集
    fn collect_custom_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        self.collect_all_nodes(graph)
            .into_iter()
            .filter(|node| matches!(&node.op, GraphOp::Custom { .. }))
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

    /// マージ可能なCustomノードのペアを検出
    ///
    /// 親ノード（consumer）と子ノード（producer）の関係にあるペアを返す
    /// Viewノードが間に挟まっている場合もトレースバックして検出する
    fn find_mergeable_pairs(&self, graph: &Graph) -> Vec<(GraphNode, GraphNode)> {
        let custom_nodes = self.collect_custom_nodes(graph);
        let mut pairs = Vec::new();

        // 各Customノードの参照カウントを計算
        let ref_counts = self.count_node_references(graph);

        for consumer in &custom_nodes {
            // 出力 Buffer を除外した入力ノードのみを処理
            let input_nodes = Self::get_input_nodes(&consumer.src);
            for producer_or_view in input_nodes {
                // Viewノードをトレースバックして実際のストレージノードを取得
                let producer = Self::trace_to_storage_node(producer_or_view);

                // producerがCustomノードかチェック
                if !matches!(&producer.op, GraphOp::Custom { .. }) {
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
    /// Viewノードを透過的に扱い、実際のストレージノード（Custom等）の
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

    /// 2つのCustomノードを1つのCustom(Program)にマージ
    fn merge_pair(
        &self,
        graph: &Graph,
        consumer: &GraphNode,
        producer: &GraphNode,
    ) -> Option<Graph> {
        // producer と consumer のAST情報を取得
        let (producer_ast, consumer_ast) = match (&producer.op, &consumer.op) {
            (GraphOp::Custom { ast: p_ast }, GraphOp::Custom { ast: c_ast }) => (p_ast, c_ast),
            _ => return None,
        };

        // 各ASTからカーネル関数を抽出/作成
        let mut kernels: Vec<AstNode> = Vec::new();
        let mut used_names: HashSet<String> = HashSet::new();

        // Producer側のカーネルを追加
        let producer_kernels =
            self.extract_or_create_kernels(producer, producer_ast, &mut used_names);
        kernels.extend(producer_kernels);
        let producer_kernel_count = kernels.len();

        // Consumer側のカーネルを追加
        let consumer_kernels =
            self.extract_or_create_kernels(consumer, consumer_ast, &mut used_names);
        kernels.extend(consumer_kernels);

        // main関数を生成
        let main_fn =
            self.generate_main_function(producer, consumer, producer_kernel_count, &kernels);
        kernels.push(main_fn);

        // Programを作成
        let program = AstNode::Program {
            functions: kernels,
            entry_point: "harp_main".to_string(),
        };

        // 新しいCustomノードの入力を構築
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

        // 新しいCustomノードを作成
        let merged_node = GraphNode::new(
            consumer.dtype.clone(),
            GraphOp::Custom { ast: program },
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
                // Custom(Function) → Kernelに変換、元の名前を保持
                let kernel = self.create_kernel_from_function(node, ast, used_names);
                if let AstNode::Kernel { name: Some(n), .. } = &kernel {
                    used_names.insert(n.clone());
                }
                vec![kernel]
            }
            AstNode::Program { functions, .. } => {
                // Custom(Program) → 既存のKernelを抽出（main以外）
                let mut kernels = Vec::new();
                for func in functions {
                    match func {
                        AstNode::Kernel {
                            name,
                            params,
                            return_type,
                            body,
                            thread_group_size,
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
                                thread_group_size: *thread_group_size,
                            });
                        }
                        AstNode::Function {
                            name,
                            params,
                            return_type,
                            body,
                        } => {
                            // main関数以外をKernelに変換
                            if name.as_deref() != Some("harp_main") {
                                let base_name =
                                    name.clone().unwrap_or_else(|| "kernel".to_string());
                                let new_name = Self::make_unique_name(&base_name, used_names);
                                used_names.insert(new_name.clone());
                                kernels.push(AstNode::Kernel {
                                    name: Some(new_name),
                                    params: params.clone(),
                                    return_type: return_type.clone(),
                                    body: body.clone(),
                                    thread_group_size: 64,
                                });
                            }
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

        AstNode::Kernel {
            name: Some(kernel_name),
            params,
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(body),
            thread_group_size: 64,
        }
    }

    /// main関数を生成
    fn generate_main_function(
        &self,
        producer: &GraphNode,
        consumer: &GraphNode,
        producer_kernel_count: usize,
        all_kernels: &[AstNode],
    ) -> AstNode {
        let mut params: Vec<VarDecl> = Vec::new();
        let mut param_names: HashSet<String> = HashSet::new();

        // Producer の入力バッファー（出力 Buffer を除外）
        let producer_inputs = Self::get_input_nodes(&producer.src);
        for (i, src) in producer_inputs.iter().enumerate() {
            let name = format!("input{}", i);
            if !param_names.contains(&name) {
                params.push(VarDecl {
                    name: name.clone(),
                    dtype: Self::graph_dtype_to_ast_ptr(&src.dtype),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                });
                param_names.insert(name);
            }
        }

        // Consumer の追加入力バッファー（producerと出力Bufferを除く）
        let mut consumer_input_offset = producer_inputs.len();
        for src in &consumer.src {
            // producer と出力 Buffer は除外
            if src.as_ptr() != producer.as_ptr() && !Self::is_output_buffer(src) {
                let name = format!("input{}", consumer_input_offset);
                if !param_names.contains(&name) {
                    params.push(VarDecl {
                        name: name.clone(),
                        dtype: Self::graph_dtype_to_ast_ptr(&src.dtype),
                        mutability: Mutability::Immutable,
                        kind: VarKind::Normal,
                    });
                    param_names.insert(name);
                }
                consumer_input_offset += 1;
            }
        }

        // 出力バッファー
        params.push(VarDecl {
            name: "output".to_string(),
            dtype: Self::graph_dtype_to_ast_ptr(&consumer.dtype),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

        // main関数のbody
        let mut statements: Vec<AstNode> = Vec::new();
        let mut scope = Scope::new();

        // 中間バッファーの確保（producer -> consumer間のデータ用）
        let tmp_buffer_name = "tmp0".to_string();
        let producer_output_dtype = Self::graph_dtype_to_ast(&producer.dtype);
        let producer_output_size = Self::compute_buffer_size(producer);

        let ptr_dtype = AstDType::Ptr(Box::new(producer_output_dtype.clone()));
        if scope
            .declare(tmp_buffer_name.clone(), ptr_dtype, Mutability::Mutable)
            .is_ok()
        {
            let alloc_expr = AstNode::Allocate {
                dtype: Box::new(producer_output_dtype),
                size: Box::new(producer_output_size),
            };
            statements.push(assign(&tmp_buffer_name, alloc_expr));
        }

        // Producer カーネルの呼び出し
        for kernel in all_kernels.iter().take(producer_kernel_count) {
            let kernel_name = Self::get_kernel_name(kernel);
            let mut args: Vec<AstNode> = Vec::new();

            // 入力バッファー（出力 Buffer を除外した数だけ）
            for j in 0..producer_inputs.len() {
                args.push(var(format!("input{}", j)));
            }
            // 出力は中間バッファー
            args.push(var(&tmp_buffer_name));

            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });
        }

        // バリア挿入
        statements.push(AstNode::Barrier);

        // Consumer カーネルの呼び出し
        let consumer_inputs = Self::get_input_nodes(&consumer.src);
        for kernel in all_kernels.iter().skip(producer_kernel_count) {
            let kernel_name = Self::get_kernel_name(kernel);

            // main関数自体は除外
            if kernel_name == "harp_main" {
                continue;
            }

            let mut args: Vec<AstNode> = Vec::new();

            // Consumer の入力を構築（出力 Buffer を除外）
            // producer の位置には中間バッファーを使用
            let mut input_idx = producer_inputs.len();
            for src in consumer_inputs.iter() {
                if src.as_ptr() == producer.as_ptr() {
                    args.push(var(&tmp_buffer_name));
                } else {
                    args.push(var(format!("input{}", input_idx)));
                    input_idx += 1;
                }
            }
            // 出力バッファー
            args.push(var("output"));

            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });
        }

        // 中間バッファーの解放
        statements.push(AstNode::Deallocate {
            ptr: Box::new(var(&tmp_buffer_name)),
        });

        let body = block(statements, scope);

        function(Some("harp_main"), params, AstDType::Tuple(vec![]), body)
    }

    /// カーネル名を取得
    fn get_kernel_name(kernel: &AstNode) -> String {
        match kernel {
            AstNode::Kernel { name: Some(n), .. } => n.clone(),
            AstNode::Function { name: Some(n), .. } => n.clone(),
            _ => "unknown".to_string(),
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

        let mut visited = HashSet::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            visited: &mut HashSet<*const GraphNodeData>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // Inputノードは常に元のノードをそのまま返す
            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            if visited.contains(&ptr) {
                return node.clone();
            }
            visited.insert(ptr);

            let new_src: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, visited))
                .collect();

            let src_changed = new_src
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            if !src_changed {
                return node.clone();
            }

            GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
            )
        }

        let mut new_graph = Graph::new();

        // 入力ノードを保持
        let mut sorted_input_names: Vec<_> = graph.inputs().keys().cloned().collect();
        sorted_input_names.sort();

        for name in &sorted_input_names {
            if let Some(weak_input) = graph.inputs().get(name)
                && let Some(rc_node) = weak_input.upgrade()
            {
                let input_node = GraphNode::from_rc(rc_node);
                new_graph.register_input(name.clone(), input_node);
            }
        }

        // 出力ノードを再構築
        let mut outputs: Vec<_> = graph.outputs().iter().collect();
        outputs.sort_by_key(|(name, _)| name.as_str());

        for (name, output_node) in outputs {
            let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
            new_graph.output(name, rebuilt);
        }

        new_graph
    }

    fn graph_dtype_to_ast(dtype: &GraphDType) -> AstDType {
        match dtype {
            GraphDType::Bool => AstDType::Bool,
            GraphDType::I32 => AstDType::Int,
            GraphDType::F32 => AstDType::F32,
            GraphDType::Complex => AstDType::F32,
            GraphDType::Unknown => AstDType::F32,
        }
    }

    fn graph_dtype_to_ast_ptr(dtype: &GraphDType) -> AstDType {
        AstDType::Ptr(Box::new(Self::graph_dtype_to_ast(dtype)))
    }

    fn compute_buffer_size(node: &GraphNode) -> AstNode {
        use crate::ast::helper::const_int;

        let shape = node.view.shape();
        if shape.is_empty() {
            return const_int(1);
        }

        let mut size: AstNode = shape[0].clone().into();
        for dim in &shape[1..] {
            let dim_ast: AstNode = dim.clone().into();
            size = size * dim_ast;
        }
        size
    }

    /// グラフ内のCustomノードの統計を取得（テスト用）
    #[cfg(test)]
    pub fn count_custom_nodes(&self, graph: &Graph) -> (usize, usize) {
        let nodes = self.collect_all_nodes(graph);
        let mut function_count = 0;
        let mut program_count = 0;

        for node in &nodes {
            if let GraphOp::Custom { ast } = &node.op {
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

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let pairs = self.find_mergeable_pairs(graph);

        log::debug!(
            "KernelMergeSuggester: found {} mergeable pairs",
            pairs.len()
        );

        let mut suggestions = Vec::new();

        for (consumer, producer) in pairs {
            if let Some(merged_graph) = self.merge_pair(graph, &consumer, &producer) {
                log::debug!("KernelMergeSuggester: merged pair (consumer -> producer)");
                suggestions.push(merged_graph);
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
        let merged = &suggestions[0];
        let (fn_count, prog_count) = suggester.count_custom_nodes(merged);

        assert_eq!(prog_count, 1, "Should have 1 Custom(Program)");
        assert_eq!(fn_count, 0, "Should have 0 Custom(Function)");
    }

    #[test]
    fn test_kernel_merge_pairwise_chain() {
        use crate::ast::helper::wildcard;

        // 3つのCustomノードのチェーン: a -> custom1 -> custom2 -> custom3
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
        let merged_once = &suggestions[0];
        let pairs_after = suggester.find_mergeable_pairs(merged_once);

        // マージ後も1つのペアが残っているはず
        println!("After first merge: {} pairs remaining", pairs_after.len());
        assert!(
            pairs_after.len() <= 1,
            "Should have at most 1 pair after first merge"
        );
    }

    #[test]
    fn test_kernel_merge_no_merge_multiple_refs() {
        use crate::ast::helper::wildcard;

        // custom1が2箇所で使われる場合、マージしない
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        let custom1 = a.custom_elementwise_binary(b.clone(), wildcard("0") + wildcard("1"));
        let custom2 = custom1
            .clone()
            .custom_elementwise_binary(c, wildcard("0") * wildcard("1"));
        let custom3 = custom1.custom_elementwise_binary(b, wildcard("0") - wildcard("1"));

        // 両方を出力（custom1が複数回参照される）
        graph.output("result1", custom2);
        graph.output("result2", custom3);

        let suggester = KernelMergeSuggester::new();
        let pairs = suggester.find_mergeable_pairs(&graph);

        // custom1は2箇所で使われているのでマージ対象外
        // custom2とcustom3はcustom1に依存しているが、custom1をマージできない
        assert_eq!(
            pairs.len(),
            0,
            "Should not find mergeable pairs when producer has multiple references"
        );
    }

    #[test]
    fn test_kernel_merge_with_barriers() {
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

        // Programを検査してバリアが挿入されていることを確認
        let merged = &suggestions[0];
        if let Some(output) = merged.outputs().values().next() {
            if let GraphOp::Custom { ast } = &output.op {
                if let AstNode::Program { functions, .. } = ast {
                    let main_fn = functions.iter().find(|f| {
                        matches!(f, AstNode::Function { name: Some(n), .. } if n == "harp_main")
                    });

                    assert!(main_fn.is_some(), "Should have harp_main function");

                    if let Some(AstNode::Function { body, .. }) = main_fn {
                        let has_barrier = contains_barrier(body);
                        assert!(has_barrier, "Main function should contain barriers");
                    }
                }
            }
        }
    }

    fn contains_barrier(node: &AstNode) -> bool {
        match node {
            AstNode::Barrier => true,
            AstNode::Block { statements, .. } => statements.iter().any(contains_barrier),
            _ => node.children().iter().any(|child| contains_barrier(child)),
        }
    }

    #[test]
    fn test_kernel_merge_with_optimizer() {
        use crate::backend::pipeline::{SuggesterFlags, create_graph_suggester};
        use crate::opt::graph::{BeamSearchGraphOptimizer, GraphOptimizer, SimpleCostEstimator};

        let _ = env_logger::try_init();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        let sum = &a + &b;
        let prod = &sum * &c;
        let result = &prod + 1.0f32;
        graph.output("result", result);

        let suggester = create_graph_suggester(SuggesterFlags::new());
        let estimator = SimpleCostEstimator::new();
        let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(4)
            .with_max_steps(100);

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

            if let GraphOp::Custom { ast } = &node.op {
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

        println!("Custom(Function): {}", custom_function_count);
        println!("Custom(Program): {}", custom_program_count);

        assert!(
            custom_function_count == 1 || custom_program_count >= 1,
            "Expected either 1 Custom(Function) after fusion, or 1 Custom(Program) after merge"
        );
    }
}
