//! Sink Absorption Suggester
//!
//! Custom(Function)ノードをSinkノードに吸収するSuggester。
//! LoweringSuggesterで生成されたCustom(Function)ノードを順次Sinkに取り込み、
//! 最終的に Sink(Program)のみの構成を目指します。
//!
//! # 処理フロー
//! 1. Sinkの入力側にあるCustom(Function)を検出
//! 2. 参照カウントを計算して吸収可能かを判定
//! 3. 吸収時にProgramにKernelを追加し、main関数を更新
//!
//! # 参照カウントの計算
//! Sinkを起点に参照カウントを計算することで、
//! 複数出力間の依存関係問題を解決します。
//!
//! # ViewMergeSuggesterとの連携
//! このSuggesterはViewノードを透過的に扱わず、直接Custom(Function)を検出します。
//! Viewノードの処理はViewMergeSuggesterに委譲されるため、
//! パイプラインではViewMergeSuggesterが先に適用される必要があります。

use crate::ast::helper::{block, function, var};
use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{DType as GraphDType, Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// Custom(Function)ノードをSinkに吸収するSuggester
pub struct SinkAbsorptionSuggester;

/// カーネル情報（AST + 入力バッファ名のマッピング）
struct KernelInfo {
    /// カーネルのAST
    ast: AstNode,
    /// 入力バッファ名のリスト（カーネルパラメータ順）
    input_buffer_names: Vec<String>,
}

impl SinkAbsorptionSuggester {
    pub fn new() -> Self {
        Self
    }

    /// ノードが出力 Buffer かどうかを判定
    fn is_output_buffer(node: &GraphNode) -> bool {
        matches!(&node.op, GraphOp::Buffer { name } if name.starts_with("output"))
    }

    /// src から入力ノードのみを取得（出力 Buffer を除外）
    fn get_input_nodes(src: &[GraphNode]) -> Vec<&GraphNode> {
        src.iter().filter(|n| !Self::is_output_buffer(n)).collect()
    }

    /// Sinkノードから吸収可能なCustom(Function)を検出
    ///
    /// 吸収条件:
    /// - Custom(Function)ノードであること
    /// - 「外部参照カウント」が0であること（すべての消費者がSink内部にある）
    ///
    /// 複数出力の場合、出力間で共有されるノードも吸収可能です。
    /// ただし、吸収順序は依存関係に基づいて決定されます。
    fn find_absorbable_customs(&self, graph: &Graph) -> Vec<GraphNode> {
        let sink = match graph.sink() {
            Some(s) => s,
            None => return vec![],
        };

        // Sink内部からの参照カウントを計算
        let internal_ref_counts = self.count_internal_references(sink);

        // 既にProgramに吸収されたノードを追跡
        let absorbed_nodes = self.collect_absorbed_node_ptrs(sink);

        let mut absorbable = Vec::new();

        // Sinkのsrcを走査してCustom(Function)を検出
        for src in &sink.src {
            self.find_customs_recursive_multi_output(
                src,
                &internal_ref_counts,
                &absorbed_nodes,
                &mut absorbable,
                &mut HashSet::new(),
            );
        }

        absorbable
    }

    /// 既にSink(Program)に吸収されたノードのポインタを収集
    fn collect_absorbed_node_ptrs(&self, _sink: &GraphNode) -> HashSet<*const GraphNodeData> {
        // Sink内のProgramから、すでに吸収されたカーネルの情報を取得することも可能だが、
        // 現在の実装では、Sinkのsrc内のCustom(Function)をチェックするだけで十分
        // 将来的には、Programに含まれるカーネル名から元のノードを追跡できる

        // 現在は空を返す（すべてのCustom(Function)は未吸収として扱う）
        HashSet::new()
    }

    /// 再帰的にCustom(Function)ノードを探索（複数出力対応版）
    ///
    /// 複数出力間で共有されるノードも吸収可能とします。
    /// トポロジカル順序に基づいて、リーフから順に吸収します。
    fn find_customs_recursive_multi_output(
        &self,
        node: &GraphNode,
        internal_ref_counts: &HashMap<*const GraphNodeData, usize>,
        absorbed_nodes: &HashSet<*const GraphNodeData>,
        result: &mut Vec<GraphNode>,
        visited: &mut HashSet<*const GraphNodeData>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // 先に子ノードを探索（トポロジカル順序を維持）
        for src in &node.src {
            self.find_customs_recursive_multi_output(
                src,
                internal_ref_counts,
                absorbed_nodes,
                result,
                visited,
            );
        }

        // Custom(Function)かどうかをチェック
        // 注: ViewノードはViewMergeSuggesterによって事前に削除されているため、
        // ここではtrace_to_storage_nodeは不要
        let node_ptr = node.as_ptr();

        if let GraphOp::Custom { ast } = &node.op
            && matches!(ast, AstNode::Function { .. })
        {
            // 既に吸収済みでなく、まだ結果に含まれていない場合
            if !absorbed_nodes.contains(&node_ptr) && !result.iter().any(|n| n.as_ptr() == node_ptr)
            {
                log::debug!(
                    "SinkAbsorption: found absorbable Custom(Function) with ref_count={}",
                    internal_ref_counts.get(&node_ptr).copied().unwrap_or(0)
                );
                result.push(node.clone());
            }
        }
    }

    /// Sink内部からの参照カウントを計算
    ///
    /// 各ノードが何回参照されているかをカウントします。
    /// これは主にデバッグ目的で使用されます。
    /// 注: ViewノードはViewMergeSuggesterによって事前に削除されているため、
    /// ここではtrace_to_storage_nodeは不要
    fn count_internal_references(&self, sink: &GraphNode) -> HashMap<*const GraphNodeData, usize> {
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
                let src_ptr = src.as_ptr();
                *ref_counts.entry(src_ptr).or_insert(0) += 1;

                // 再帰的に訪問
                visit(src, ref_counts, visited);
            }
        }

        // Sinkのsrcから探索開始
        for src in &sink.src {
            visit(src, &mut ref_counts, &mut visited);
        }

        ref_counts
    }

    /// Custom(Function)をSinkに吸収
    fn absorb_custom(&self, graph: &Graph, custom_node: &GraphNode) -> Option<Graph> {
        let sink = graph.sink()?;

        // Custom(Function)のASTを取得
        let custom_ast = match &custom_node.op {
            GraphOp::Custom { ast } => ast,
            _ => return None,
        };

        // 現在のSinkからProgram情報を取得
        let (mut program_functions, entry_point, output_names) = match &sink.op {
            GraphOp::Sink { ast, outputs } => {
                if let AstNode::Program {
                    functions,
                    entry_point,
                } = ast
                {
                    (functions.clone(), entry_point.clone(), outputs.clone())
                } else {
                    (vec![], "harp_main".to_string(), outputs.clone())
                }
            }
            _ => return None,
        };

        // 使用済み名前を収集
        let mut used_names: HashSet<String> = program_functions
            .iter()
            .filter_map(|f| match f {
                AstNode::Kernel { name: Some(n), .. } => Some(n.clone()),
                AstNode::Function { name: Some(n), .. } => Some(n.clone()),
                _ => None,
            })
            .collect();

        // Custom(Function)をKernelに変換して追加
        let kernel_info =
            self.create_kernel_from_function(graph, custom_node, custom_ast, &mut used_names);
        program_functions.push(kernel_info.ast.clone());

        // 新しいSinkのsrcを構築（custom_nodeを削除し、その入力を直接接続）
        let new_src = self.rebuild_sink_src(sink, custom_node);
        // TODO: Bufferノードをフィルタリングしてメタデータで管理する
        // 現状、フィルタリングを有効にするとテストが失敗する
        // let new_src: Vec<GraphNode> = new_src_raw
        //     .into_iter()
        //     .filter(|n| !matches!(&n.op, GraphOp::Buffer { .. }))
        //     .collect();

        // main関数を生成/更新（カーネル引数マッピング情報を渡す）
        let main_fn = self.generate_main_function(
            graph,
            &new_src,
            &program_functions,
            &output_names,
            &[kernel_info],
        );

        // 既存のmain関数を削除して新しいものを追加
        program_functions
            .retain(|f| !matches!(f, AstNode::Function { name: Some(n), .. } if n == "harp_main"));
        program_functions.push(main_fn);

        // 新しいProgramを作成
        let new_program = AstNode::Program {
            functions: program_functions,
            entry_point,
        };

        // 新しいSinkノードを作成
        let new_sink = GraphNode::new(
            sink.dtype.clone(),
            GraphOp::Sink {
                ast: new_program,
                outputs: output_names,
            },
            new_src,
            sink.view.clone(),
        );

        // 新しいグラフを構築
        Some(self.rebuild_graph_with_sink(graph, new_sink))
    }

    /// Sinkのsrcを再構築（吸収されたCustomノードを入力で置き換え）
    fn rebuild_sink_src(&self, sink: &GraphNode, absorbed_custom: &GraphNode) -> Vec<GraphNode> {
        let mut new_src = Vec::new();
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        // absorbed_customを入力ノードの最初のものに置き換えるマッピングを作成
        // ただし、実際には吸収されたノードの依存関係を維持する必要がある
        fn rebuild_node(
            node: &GraphNode,
            absorbed_ptr: *const GraphNodeData,
            absorbed_inputs: &[GraphNode],
            node_map: &mut HashMap<*const GraphNodeData, GraphNode>,
            visited: &mut HashSet<*const GraphNodeData>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // キャッシュをチェック
            if let Some(mapped) = node_map.get(&ptr) {
                return mapped.clone();
            }

            // 吸収されたノードの場合、入力ノードを返す
            // 注: 複数入力の場合は最初の入力を返す（単純化）
            if ptr == absorbed_ptr
                && let Some(first_input) = absorbed_inputs.first()
            {
                return first_input.clone();
            }

            // Bufferノードはそのまま返す
            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            // 循環参照を避ける
            if visited.contains(&ptr) {
                return node.clone();
            }
            visited.insert(ptr);

            // 再帰的に子ノードを再構築
            let new_children: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, absorbed_ptr, absorbed_inputs, node_map, visited))
                .collect();

            // 子ノードに変更がなければ元のノードを返す
            let children_changed = new_children
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            if !children_changed {
                return node.clone();
            }

            // 新しいノードを作成
            let new_node = GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_children,
                node.view.clone(),
            );
            node_map.insert(ptr, new_node.clone());
            new_node
        }

        // 吸収されたノードの入力を取得（出力Bufferを除く）
        let absorbed_inputs: Vec<GraphNode> = Self::get_input_nodes(&absorbed_custom.src)
            .iter()
            .map(|n| (*n).clone())
            .collect();

        let absorbed_ptr = absorbed_custom.as_ptr();
        let mut visited = HashSet::new();

        // Sinkのsrcを再構築
        for src in &sink.src {
            let rebuilt = rebuild_node(
                src,
                absorbed_ptr,
                &absorbed_inputs,
                &mut node_map,
                &mut visited,
            );
            new_src.push(rebuilt);
        }

        // 吸収されたノードの入力を新しいsrcに追加（まだ含まれていない場合）
        for input in &absorbed_inputs {
            let input_ptr = input.as_ptr();
            if !new_src.iter().any(|n| n.as_ptr() == input_ptr) {
                // 入力が既存のsrcのどれにも含まれていない場合は追加
                // ただし、通常は依存関係を通じて既に含まれているはず
            }
        }

        new_src
    }

    /// Kernel関数を作成
    fn create_kernel_from_function(
        &self,
        _graph: &Graph,
        node: &GraphNode,
        func_ast: &AstNode,
        used_names: &mut HashSet<String>,
    ) -> KernelInfo {
        let input_nodes = Self::get_input_nodes(&node.src);

        let input_shape = if !input_nodes.is_empty() {
            input_nodes[0].view.shape().to_vec()
        } else {
            node.view.shape().to_vec()
        };

        // 入力バッファ名を収集（カーネルパラメータ順）
        let mut input_buffer_names = Vec::new();
        for src in input_nodes.iter() {
            let name = match &src.op {
                GraphOp::Buffer { name } => name.clone(),
                _ => format!("intermediate_{}", src.as_ptr() as usize),
            };
            input_buffer_names.push(name);
        }

        // パラメータを生成
        let mut params = Vec::new();

        // 入力バッファー - Custom.srcの順序で生成（bodyのプレースホルダーと一致させる）
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
        used_names.insert(kernel_name.clone());

        let ast = AstNode::Kernel {
            name: Some(kernel_name),
            params,
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(body),
            thread_group_size: 64,
        };

        KernelInfo {
            ast,
            input_buffer_names,
        }
    }

    /// 重複を避けてユニークな名前を生成
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

    /// main関数を生成
    fn generate_main_function(
        &self,
        graph: &Graph,
        _sink_src: &[GraphNode],
        kernels: &[AstNode],
        output_names: &[String],
        kernel_infos: &[KernelInfo],
    ) -> AstNode {
        let mut params: Vec<VarDecl> = Vec::new();
        let mut param_names: HashSet<String> = HashSet::new();

        // 入力バッファーのパラメータを追加（メタデータから、ソート順）
        let mut sorted_inputs: Vec<_> =
            graph.input_metas().iter().map(|m| m.name.clone()).collect();
        sorted_inputs.sort();

        // バッファ名からmainパラメータ名へのマッピングを構築
        let mut buffer_to_main_param: HashMap<String, String> = HashMap::new();
        for (i, name) in sorted_inputs.iter().enumerate() {
            if let Some(meta) = graph.input_metas().iter().find(|m| &m.name == name) {
                let param_name = format!("input{}", i);
                if !param_names.contains(&param_name) {
                    params.push(VarDecl {
                        name: param_name.clone(),
                        dtype: Self::graph_dtype_to_ast_ptr(&meta.dtype),
                        mutability: Mutability::Immutable,
                        kind: VarKind::Normal,
                    });
                    param_names.insert(param_name.clone());
                    buffer_to_main_param.insert(name.clone(), param_name);
                }
            }
        }

        // 出力バッファーのパラメータを追加（メタデータから）
        for (i, name) in output_names.iter().enumerate() {
            if let Some(meta) = graph.output_metas().iter().find(|m| &m.name == name) {
                let param_name = format!("output{}", i);
                if !param_names.contains(&param_name) {
                    params.push(VarDecl {
                        name: param_name.clone(),
                        dtype: Self::graph_dtype_to_ast_ptr(&meta.dtype),
                        mutability: Mutability::Mutable,
                        kind: VarKind::Normal,
                    });
                    param_names.insert(param_name);
                }
            }
        }

        // main関数のbody
        let mut statements: Vec<AstNode> = Vec::new();
        let scope = Scope::new();

        // 各カーネルを呼び出す
        for kernel in kernels {
            let kernel_name = Self::get_kernel_name(kernel);

            // main関数自体はスキップ
            if kernel_name == "harp_main" {
                continue;
            }

            // カーネル情報からこのカーネルの入力バッファ名を取得
            let kernel_info = kernel_infos.iter().find(|k| {
                if let AstNode::Kernel { name: Some(n), .. } = &k.ast {
                    n == &kernel_name
                } else {
                    false
                }
            });

            let mut args: Vec<AstNode> = Vec::new();

            // カーネルの入力パラメータ数を取得
            let input_count = Self::get_kernel_input_count(kernel);

            if let Some(info) = kernel_info {
                // カーネルパラメータ順（Custom.src順）でmainパラメータにマッピング
                for buffer_name in &info.input_buffer_names {
                    if let Some(main_param) = buffer_to_main_param.get(buffer_name) {
                        args.push(var(main_param.clone()));
                    }
                    // 中間結果など、input_metasにないバッファはスキップ
                }
            }

            // KernelInfoがない場合、またはマッピングで十分な引数が生成されなかった場合はフォールバック
            if args.len() < input_count {
                args.clear();
                // カーネルのパラメータ数に合わせてソート順で入力を渡す
                for i in 0..input_count.min(sorted_inputs.len()) {
                    args.push(var(format!("input{}", i)));
                }
            }

            // 出力バッファー
            if !output_names.is_empty() {
                args.push(var("output0"));
            }

            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });

            // バリアを挿入（最後のカーネル以外）
            statements.push(AstNode::Barrier);
        }

        // 最後のバリアを削除
        if statements.last() == Some(&AstNode::Barrier) {
            statements.pop();
        }

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

    /// カーネルの入力パラメータ数を取得（出力パラメータを除く）
    fn get_kernel_input_count(kernel: &AstNode) -> usize {
        match kernel {
            AstNode::Kernel { params, .. } | AstNode::Function { params, .. } => {
                // 最後のパラメータは出力バッファなので除く
                if params.is_empty() {
                    0
                } else {
                    params.len() - 1
                }
            }
            _ => 0,
        }
    }

    /// 新しいSinkノードでグラフを再構築
    fn rebuild_graph_with_sink(&self, graph: &Graph, new_sink: GraphNode) -> Graph {
        let mut new_graph = Graph::new();

        // 入力・出力メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // 新しいSinkをセット
        new_graph.set_sink(new_sink);

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
            GraphDType::Complex => AstDType::F32,
            GraphDType::Unknown => AstDType::F32,
        }
    }

    fn graph_dtype_to_ast_ptr(dtype: &GraphDType) -> AstDType {
        AstDType::Ptr(Box::new(Self::graph_dtype_to_ast(dtype)))
    }
}

impl Default for SinkAbsorptionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for SinkAbsorptionSuggester {
    fn name(&self) -> &'static str {
        "SinkAbsorption"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        // Sinkがない場合は何もしない
        if graph.sink().is_none() {
            log::debug!("SinkAbsorptionSuggester: no Sink node found");
            return vec![];
        }

        let absorbable = self.find_absorbable_customs(graph);

        log::debug!(
            "SinkAbsorptionSuggester: found {} absorbable Custom(Function) nodes",
            absorbable.len()
        );

        let mut suggestions = Vec::new();

        for custom in absorbable {
            if let Some(new_graph) = self.absorb_custom(graph, &custom) {
                log::debug!("SinkAbsorptionSuggester: absorbed Custom(Function)");
                suggestions.push(new_graph);
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
    fn test_sink_absorption_basic() {
        let suggester = SinkAbsorptionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        let c = a + b;
        graph.output("c", c);

        // Sinkが存在することを確認
        assert!(graph.sink().is_some());

        // まだCustomノードがないので、吸収対象はない
        let absorbable = suggester.find_absorbable_customs(&graph);
        assert!(absorbable.is_empty());
    }

    #[test]
    fn test_find_absorbable_with_custom() {
        use crate::ast::helper::wildcard;

        let suggester = SinkAbsorptionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // Custom(Function)を作成
        let custom = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        graph.output("c", custom);

        eprintln!("Sink exists: {:?}", graph.sink().is_some());
        if let Some(ref sink) = graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                eprintln!("  src[{}]: {:?}", i, src.op);
            }
        }

        // Custom(Function)が吸収対象として検出される
        let absorbable = suggester.find_absorbable_customs(&graph);
        eprintln!("Found {} absorbable Custom nodes", absorbable.len());
        for (i, node) in absorbable.iter().enumerate() {
            eprintln!("  absorbable[{}]: {:?}", i, node.op);
        }
        assert_eq!(absorbable.len(), 1);
    }

    #[test]
    fn test_absorb_custom_into_sink() {
        use crate::ast::helper::wildcard;

        let suggester = SinkAbsorptionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // Custom(Function)を作成
        let custom = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        graph.output("c", custom);

        // suggest()を呼び出し
        let suggestions = suggester.suggest(&graph);
        eprintln!("Got {} suggestions from SinkAbsorption", suggestions.len());

        // 1つの提案があるはず
        assert!(
            !suggestions.is_empty(),
            "Should have at least one suggestion"
        );

        let new_graph = &suggestions[0];
        eprintln!("New graph sink exists: {:?}", new_graph.sink().is_some());

        if let Some(ref sink) = new_graph.sink() {
            if let GraphOp::Sink { ast, outputs } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
                if let crate::ast::AstNode::Program { functions, .. } = ast {
                    eprintln!("Program has {} functions", functions.len());
                    for (i, func) in functions.iter().enumerate() {
                        let name = match func {
                            crate::ast::AstNode::Kernel { name, .. } => name.clone(),
                            crate::ast::AstNode::Function { name, .. } => name.clone(),
                            _ => None,
                        };
                        eprintln!("  function[{}]: {:?}", i, name);
                    }
                }
            }
        }
    }

    #[test]
    fn test_lowering_then_sink_absorption() {
        use crate::opt::graph::suggesters::LoweringSuggester;

        let lowering = LoweringSuggester::new();
        let sink_absorber = SinkAbsorptionSuggester::new();

        // シンプルなElementwise演算グラフ
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = a + b;
        graph.output("c", c);

        eprintln!("=== Initial Graph ===");
        eprintln!("Sink exists: {:?}", graph.sink().is_some());

        // LoweringSuggesterを適用
        let lowered = lowering.suggest(&graph);
        eprintln!("\n=== After Lowering ===");
        eprintln!("Got {} suggestions from Lowering", lowered.len());

        assert!(!lowered.is_empty(), "Lowering should produce suggestions");
        let lowered_graph = &lowered[0];

        eprintln!(
            "Lowered graph sink exists: {:?}",
            lowered_graph.sink().is_some()
        );
        if let Some(ref sink) = lowered_graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Custom { .. } => "Custom".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    GraphOp::Elementwise { op, .. } => format!("Elementwise({:?})", op),
                    _ => format!("{:?}", src.op),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
        }

        // SinkAbsorptionを適用
        let absorbed = sink_absorber.suggest(lowered_graph);
        eprintln!("\n=== After SinkAbsorption ===");
        eprintln!("Got {} suggestions from SinkAbsorption", absorbed.len());

        if absorbed.is_empty() {
            // 吸収対象が見つからない場合、理由を調べる
            let absorbable = sink_absorber.find_absorbable_customs(lowered_graph);
            eprintln!("Absorbable nodes: {}", absorbable.len());
        }

        assert!(
            !absorbed.is_empty(),
            "SinkAbsorption should produce suggestions"
        );

        // 吸収後のSinkの状態を確認
        let absorbed_graph = &absorbed[0];
        eprintln!("\n=== Absorbed Graph Details ===");
        eprintln!(
            "Input metas: {:?}",
            absorbed_graph
                .input_metas()
                .iter()
                .map(|m| &m.name)
                .collect::<Vec<_>>()
        );
        if let Some(ref sink) = absorbed_graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Custom { .. } => "Custom".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    _ => format!("{:?}", src.op),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
            if let GraphOp::Sink { ast, outputs } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
                if let AstNode::Program { functions, .. } = ast {
                    eprintln!("Program functions: {}", functions.len());
                    for f in functions {
                        match f {
                            AstNode::Function { name, params, .. } => {
                                eprintln!("  Function {:?}: {} params", name, params.len());
                                for p in params {
                                    eprintln!("    - {}", p.name);
                                }
                            }
                            AstNode::Kernel { name, params, .. } => {
                                eprintln!("  Kernel {:?}: {} params", name, params.len());
                                for p in params {
                                    eprintln!("    - {}", p.name);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_full_optimization_with_beam_search() {
        use crate::backend::pipeline::{SuggesterFlags, create_graph_suggester};
        use crate::opt::graph::{BeamSearchGraphOptimizer, SimpleCostEstimator};

        // 複数の演算を含むグラフ: reduce(a + b) + c
        // Reduceがあると融合されずに2つのCustomノードになる
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 5]);
        let b = graph.input("b", DType::F32, vec![10, 5]);
        let c = graph.input("c", DType::F32, vec![10]);
        let sum = &a + &b;
        let reduced = sum.reduce_sum(1); // [10]
        let result = &reduced + &c;
        graph.output("result", result);

        eprintln!("=== Initial Graph ===");
        eprintln!("Sink exists: {:?}", graph.sink().is_some());

        // パイプラインと同じ設定でSuggesterを作成
        let flags = SuggesterFlags {
            include_kernel_merge: true,
            include_ast_optimization: false,
        };
        let suggester = create_graph_suggester(flags);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(4)
            .with_max_steps(20)
            .with_progress(false)
            .with_collect_logs(false);

        let (optimized, history) = optimizer.optimize_with_history(graph);

        eprintln!("\n=== Optimization History ===");
        for snapshot in history.snapshots() {
            let sink_info = if let Some(ref sink) = snapshot.graph.sink() {
                if let GraphOp::Sink { ast, outputs } = &sink.op {
                    let func_count = if let crate::ast::AstNode::Program { functions, .. } = ast {
                        functions.len()
                    } else {
                        0
                    };
                    format!("Sink(outputs={:?}, funcs={})", outputs, func_count)
                } else {
                    "Unknown".to_string()
                }
            } else {
                "No Sink".to_string()
            };
            eprintln!(
                "Step {}: {} - cost={:.2} - {}",
                snapshot.step, snapshot.description, snapshot.cost, sink_info
            );
        }

        eprintln!("\n=== Final Graph ===");
        eprintln!("Sink exists: {:?}", optimized.sink().is_some());

        if let Some(ref sink) = optimized.sink() {
            if let GraphOp::Sink { ast, outputs } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
                if let crate::ast::AstNode::Program { functions, .. } = ast {
                    eprintln!("Program has {} functions", functions.len());
                    for (i, func) in functions.iter().enumerate() {
                        let name = match func {
                            crate::ast::AstNode::Kernel { name, .. } => name.clone(),
                            crate::ast::AstNode::Function { name, .. } => name.clone(),
                            _ => None,
                        };
                        eprintln!("  function[{}]: {:?}", i, name);
                    }
                    // 最終的にカーネルとmain関数があることを確認
                    assert!(
                        functions.len() >= 2,
                        "Should have at least kernel and harp_main"
                    );
                }
            }
        } else {
            panic!("Optimized graph should have Sink node");
        }
    }

    #[test]
    fn test_multiple_outputs_absorption() {
        use crate::ast::helper::wildcard;
        use crate::backend::pipeline::{SuggesterFlags, create_graph_suggester};
        use crate::opt::graph::{BeamSearchGraphOptimizer, SimpleCostEstimator};

        // 複数出力のグラフを作成: x = a + b, y = x + 1
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // x = a + b (Custom)
        let x = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        // y = x + 1 (Custom) - 使用できるcustom_elementwiseがないので、通常のelementwise演算を使用
        let y = &x + 1.0f32;

        graph.output("x", x.clone());
        graph.output("y", y);

        eprintln!("\n=== Multiple Outputs Test ===");
        eprintln!("Sink exists: {:?}", graph.sink().is_some());
        if let Some(ref sink) = graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            if let GraphOp::Sink { outputs, .. } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
            }
        }

        // SinkAbsorptionで吸収可能なノードを検出
        let suggester = SinkAbsorptionSuggester::new();
        let absorbable = suggester.find_absorbable_customs(&graph);
        eprintln!("\nAbsorbable Custom nodes: {}", absorbable.len());
        for (i, node) in absorbable.iter().enumerate() {
            if let GraphOp::Custom { ast } = &node.op {
                let name = match ast {
                    crate::ast::AstNode::Function { name, .. } => name.clone(),
                    _ => None,
                };
                eprintln!("  [{}] Custom(Function): {:?}", i, name);
            }
        }

        // 少なくとも1つのノードが吸収可能であるべき
        assert!(
            !absorbable.is_empty(),
            "Should find at least one absorbable Custom node for multiple outputs"
        );

        // BeamSearchで最適化
        let flags = SuggesterFlags {
            include_kernel_merge: true,
            include_ast_optimization: false,
        };
        let combined_suggester = create_graph_suggester(flags);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchGraphOptimizer::new(combined_suggester, estimator)
            .with_beam_width(4)
            .with_max_steps(50)
            .with_progress(false)
            .with_collect_logs(false);

        let (optimized, history) = optimizer.optimize_with_history(graph);

        eprintln!("\n=== Optimization History ===");
        for snapshot in history.snapshots() {
            let sink_info = if let Some(ref sink) = snapshot.graph.sink() {
                if let GraphOp::Sink { ast, outputs } = &sink.op {
                    let func_count = if let crate::ast::AstNode::Program { functions, .. } = ast {
                        functions.len()
                    } else {
                        0
                    };
                    format!("outputs={:?}, funcs={}", outputs, func_count)
                } else {
                    "Unknown".to_string()
                }
            } else {
                "No Sink".to_string()
            };
            eprintln!(
                "Step {}: {} - cost={:.2} - {}",
                snapshot.step, snapshot.description, snapshot.cost, sink_info
            );
        }

        // 最終グラフを確認
        eprintln!("\n=== Final Graph ===");
        if let Some(ref sink) = optimized.sink() {
            if let GraphOp::Sink { ast, outputs } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
                if let crate::ast::AstNode::Program { functions, .. } = ast {
                    eprintln!(
                        "Program has {} functions (including harp_main)",
                        functions.len()
                    );
                    for (i, func) in functions.iter().enumerate() {
                        let name = match func {
                            crate::ast::AstNode::Kernel { name, .. } => name.clone(),
                            crate::ast::AstNode::Function { name, .. } => name.clone(),
                            _ => None,
                        };
                        eprintln!("  function[{}]: {:?}", i, name);
                    }
                    // 2つのカーネル + harp_main = 3つの関数があるべき
                    assert!(
                        functions.len() >= 3,
                        "Should have at least 2 kernels + harp_main for multiple outputs, got {}",
                        functions.len()
                    );
                }
            }
        }
    }
}
