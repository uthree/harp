//! Kernel Merge Suggester
//!
//! 複数のCustomノード（Function/Program）を1つのCustomノード（Program）にマージします。
//! これにより、グラフ全体が1つのProgramとして表現され、Lowererがほぼパススルーになります。
//!
//! # サポートするマージパターン
//! - Custom(Function) × N → Custom(Program)
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
use std::collections::{HashMap, HashSet, VecDeque};

/// 複数のCustomノードを1つのProgram にマージするSuggester
pub struct KernelMergeSuggester;

impl KernelMergeSuggester {
    pub fn new() -> Self {
        Self
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

    /// トポロジカルソート（出力→入力の順序で世代別にグループ化）
    fn topological_sort(&self, graph: &Graph) -> Vec<Vec<GraphNode>> {
        let all_nodes = self.collect_all_nodes(graph);

        // 各ノードの入次数を計算
        let mut in_degree: HashMap<*const GraphNodeData, usize> = HashMap::new();
        for node in &all_nodes {
            let ptr = node.as_ptr();
            in_degree.entry(ptr).or_insert(0);

            for src in &node.src {
                let src_ptr = src.as_ptr();
                *in_degree.entry(src_ptr).or_insert(0) += 1;
            }
        }

        // Kahnのアルゴリズム
        let mut result: Vec<Vec<GraphNode>> = Vec::new();
        let mut queue: VecDeque<GraphNode> = VecDeque::new();

        for node in &all_nodes {
            let ptr = node.as_ptr();
            if in_degree[&ptr] == 0 {
                queue.push_back(node.clone());
            }
        }

        while !queue.is_empty() {
            let generation_size = queue.len();
            let mut current_generation = Vec::new();

            for _ in 0..generation_size {
                let node = queue.pop_front().unwrap();
                current_generation.push(node.clone());

                for src in &node.src {
                    let src_ptr = src.as_ptr();
                    let degree = in_degree.get_mut(&src_ptr).unwrap();
                    *degree -= 1;

                    if *degree == 0 {
                        queue.push_back(src.clone());
                    }
                }
            }

            result.push(current_generation);
        }

        result
    }

    /// グラフ内のCustomノードの統計を取得
    fn count_custom_nodes(&self, graph: &Graph) -> (usize, usize) {
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

    /// マージ可能かチェック
    /// - Custom(Function)が2つ以上
    /// - または Custom(Program)が1つ以上 かつ Custom(Function)が1つ以上
    /// - または Custom(Program)が2つ以上
    fn can_merge(&self, graph: &Graph) -> bool {
        let (function_count, program_count) = self.count_custom_nodes(graph);

        log::debug!(
            "KernelMergeSuggester: {} Custom(Function), {} Custom(Program)",
            function_count,
            program_count
        );

        // マージ可能な条件
        function_count >= 2 || (program_count >= 1 && function_count >= 1) || program_count >= 2
    }

    /// 複数のCustomノードを1つのProgram にマージ
    fn merge_to_program(&self, graph: &Graph) -> Option<Graph> {
        if !self.can_merge(graph) {
            return None;
        }

        let generations = self.topological_sort(graph);

        // 既存のProgramから関数を抽出
        let mut existing_kernels: Vec<ExistingKernel> = Vec::new();
        let mut kernel_name_counter = 0;

        // 入力ノードのバッファー名を設定
        let mut node_buffer_map: HashMap<*const GraphNodeData, String> = HashMap::new();
        let mut input_counter = 0;
        let mut sorted_input_names: Vec<_> = graph.inputs().keys().cloned().collect();
        sorted_input_names.sort();

        for (_name, weak_node) in graph.inputs().iter() {
            if let Some(rc_node) = weak_node.upgrade() {
                let node = GraphNode::from_rc(rc_node);
                let buffer_name = format!("input{}", input_counter);
                node_buffer_map.insert(node.as_ptr(), buffer_name);
                input_counter += 1;
            }
        }

        // 出力ノードのポインタを収集
        let final_output_ptrs: HashSet<*const GraphNodeData> = if !generations.is_empty() {
            generations[0].iter().map(|n| n.as_ptr()).collect()
        } else {
            graph.outputs().values().map(|n| n.as_ptr()).collect()
        };

        // 新しいカーネル情報を収集
        let mut new_kernel_infos: Vec<KernelInfo> = Vec::new();
        let mut tmp_counter = 0;

        // 各世代のCustomノードを処理（入力→出力の順）
        for generation in generations.iter().rev() {
            for node in generation {
                if let GraphOp::Custom { ast } = &node.op {
                    match ast {
                        AstNode::Function { .. } => {
                            // Custom(Function)を新しいカーネルとして処理
                            let input_buffers = self.collect_input_buffers(node, &node_buffer_map);
                            let output_buffer = self.determine_output_buffer(
                                node,
                                &final_output_ptrs,
                                &mut tmp_counter,
                            );
                            node_buffer_map.insert(node.as_ptr(), output_buffer.clone());

                            let function_name = format!("kernel_{}", kernel_name_counter);
                            let kernel_fn = self.create_kernel_function(node, &function_name, ast);

                            existing_kernels.push(ExistingKernel {
                                function: kernel_fn,
                                original_name: function_name.clone(),
                            });

                            new_kernel_infos.push(KernelInfo {
                                kernel_name: function_name,
                                input_buffers,
                                output_buffer,
                                output_dtype: Self::graph_dtype_to_ast(&node.dtype),
                                output_size: Self::compute_buffer_size(node),
                            });

                            kernel_name_counter += 1;
                        }
                        AstNode::Program { functions, .. } => {
                            // Custom(Program)から既存のカーネルを抽出
                            let extracted = self
                                .extract_kernels_from_program(functions, &mut kernel_name_counter);

                            // 入力バッファーマッピングを設定
                            let input_buffers = self.collect_input_buffers(node, &node_buffer_map);
                            let output_buffer = self.determine_output_buffer(
                                node,
                                &final_output_ptrs,
                                &mut tmp_counter,
                            );
                            node_buffer_map.insert(node.as_ptr(), output_buffer.clone());

                            // 抽出したカーネルを追加
                            for (i, kernel) in extracted.kernels.into_iter().enumerate() {
                                existing_kernels.push(kernel.clone());

                                // KernelInfoを生成
                                // 最後のカーネルのみ出力バッファーを使用
                                let is_last = i == extracted.kernel_count - 1;
                                let info_output = if is_last {
                                    output_buffer.clone()
                                } else {
                                    let name = format!("tmp{}", tmp_counter);
                                    tmp_counter += 1;
                                    name
                                };

                                // 入力バッファーは最初のカーネルのみ外部入力を使用
                                let info_inputs = if i == 0 {
                                    input_buffers.clone()
                                } else {
                                    // 前のカーネルの出力を入力として使用
                                    vec![
                                        new_kernel_infos
                                            .last()
                                            .map(|k| k.output_buffer.clone())
                                            .unwrap_or_default(),
                                    ]
                                };

                                new_kernel_infos.push(KernelInfo {
                                    kernel_name: kernel.original_name.clone(),
                                    input_buffers: info_inputs,
                                    output_buffer: info_output,
                                    output_dtype: Self::graph_dtype_to_ast(&node.dtype),
                                    output_size: Self::compute_buffer_size(node),
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if existing_kernels.is_empty() {
            return None;
        }

        // main関数を生成（バリア挿入付き）
        let main_fn = self.generate_main_function_with_barriers(&new_kernel_infos, input_counter);

        // Programを作成
        let mut all_functions: Vec<AstNode> =
            existing_kernels.into_iter().map(|k| k.function).collect();
        all_functions.push(main_fn);

        let program = AstNode::Program {
            functions: all_functions,
            entry_point: "harp_main".to_string(),
        };

        // グラフ全体を1つのCustom(Program)ノードに置き換え
        let mut new_graph = Graph::new();

        // 入力ノードを再作成
        let mut new_inputs: HashMap<String, GraphNode> = HashMap::new();
        for name in &sorted_input_names {
            if let Some(weak_node) = graph.inputs().get(name)
                && let Some(rc_node) = weak_node.upgrade()
            {
                let old_node = GraphNode::from_rc(rc_node);
                let new_input =
                    new_graph.input(name, old_node.dtype.clone(), old_node.view.shape().to_vec());
                new_inputs.insert(name.clone(), new_input);
            }
        }

        // すべての入力を収集
        let inputs: Vec<GraphNode> = sorted_input_names
            .iter()
            .filter_map(|name| new_inputs.get(name).cloned())
            .collect();

        // 出力のshapeとdtypeを取得
        let (output_dtype, output_view) = if let Some(output) = graph.outputs().values().next() {
            (output.dtype.clone(), output.view.clone())
        } else {
            return None;
        };

        // Custom(Program)ノードを作成
        let custom_program = GraphNode::new(
            output_dtype,
            GraphOp::Custom { ast: program },
            inputs,
            output_view,
        );

        new_graph.output("result", custom_program);

        Some(new_graph)
    }

    /// 入力バッファー名を収集
    fn collect_input_buffers(
        &self,
        node: &GraphNode,
        node_buffer_map: &HashMap<*const GraphNodeData, String>,
    ) -> Vec<String> {
        node.src
            .iter()
            .filter_map(|src| {
                if matches!(src.op, GraphOp::Const(_) | GraphOp::ComplexConst { .. }) {
                    None
                } else {
                    let storage_node = Self::trace_to_storage(src);
                    node_buffer_map.get(&storage_node.as_ptr()).cloned()
                }
            })
            .collect()
    }

    /// 出力バッファー名を決定
    fn determine_output_buffer(
        &self,
        node: &GraphNode,
        final_output_ptrs: &HashSet<*const GraphNodeData>,
        tmp_counter: &mut usize,
    ) -> String {
        if final_output_ptrs.contains(&node.as_ptr()) {
            "output".to_string()
        } else {
            let name = format!("tmp{}", *tmp_counter);
            *tmp_counter += 1;
            name
        }
    }

    /// Custom(Program)から既存のカーネルを抽出
    fn extract_kernels_from_program(
        &self,
        functions: &[AstNode],
        kernel_name_counter: &mut usize,
    ) -> ExtractedKernels {
        let mut kernels = Vec::new();
        let mut kernel_count = 0;

        for func in functions {
            match func {
                AstNode::Kernel {
                    name: _,
                    params,
                    return_type,
                    body,
                    thread_group_size,
                } => {
                    // カーネル関数をリネームして追加
                    let new_name = format!("kernel_{}", *kernel_name_counter);
                    *kernel_name_counter += 1;

                    let renamed_kernel = AstNode::Kernel {
                        name: Some(new_name.clone()),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: body.clone(),
                        thread_group_size: *thread_group_size,
                    };

                    kernels.push(ExistingKernel {
                        function: renamed_kernel,
                        original_name: new_name,
                    });
                    kernel_count += 1;
                }
                AstNode::Function {
                    name,
                    params,
                    return_type,
                    body,
                } => {
                    // main関数以外の通常関数もリネームして追加
                    if name.as_deref() != Some("harp_main") {
                        let new_name = format!("kernel_{}", *kernel_name_counter);
                        *kernel_name_counter += 1;

                        // FunctionをKernelに変換
                        let kernel = AstNode::Kernel {
                            name: Some(new_name.clone()),
                            params: params.clone(),
                            return_type: return_type.clone(),
                            body: body.clone(),
                            thread_group_size: 64,
                        };

                        kernels.push(ExistingKernel {
                            function: kernel,
                            original_name: new_name,
                        });
                        kernel_count += 1;
                    }
                }
                _ => {}
            }
        }

        ExtractedKernels {
            kernels,
            kernel_count,
        }
    }

    /// Viewノードの場合、実際のストレージノードまでトレースバック
    fn trace_to_storage(node: &GraphNode) -> &GraphNode {
        match &node.op {
            GraphOp::View(_) => {
                if let Some(src) = node.src.first() {
                    Self::trace_to_storage(src)
                } else {
                    node
                }
            }
            _ => node,
        }
    }

    /// カーネル関数を作成
    fn create_kernel_function(
        &self,
        node: &GraphNode,
        function_name: &str,
        custom_fn: &AstNode,
    ) -> AstNode {
        let input_shape = if !node.src.is_empty() {
            node.src[0].view.shape().to_vec()
        } else {
            node.view.shape().to_vec()
        };

        // パラメータを生成
        let mut params = Vec::new();

        // 入力バッファー
        for (i, src) in node.src.iter().enumerate() {
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
        let body = if let AstNode::Function { body, .. } = custom_fn {
            // shape変数を置換
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

        AstNode::Kernel {
            name: Some(function_name.to_string()),
            params,
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(body),
            thread_group_size: 64, // GPU用のカーネル関数として作成
        }
    }

    /// main関数を生成（バリア挿入付き）
    fn generate_main_function_with_barriers(
        &self,
        kernel_infos: &[KernelInfo],
        input_count: usize,
    ) -> AstNode {
        let mut params: Vec<VarDecl> = Vec::new();
        let mut param_names: HashSet<String> = HashSet::new();

        // 入力バッファー
        for i in 0..input_count {
            let input_name = format!("input{}", i);
            if !param_names.contains(&input_name) {
                params.push(VarDecl {
                    name: input_name.clone(),
                    dtype: AstDType::Ptr(Box::new(AstDType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                });
                param_names.insert(input_name);
            }
        }

        // 出力バッファー
        params.push(VarDecl {
            name: "output".to_string(),
            dtype: AstDType::Ptr(Box::new(AstDType::F32)),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

        // main関数のbody
        let mut statements: Vec<AstNode> = Vec::new();
        let mut scope = Scope::new();
        let mut allocated_buffers: HashSet<String> = HashSet::new();

        // 中間バッファーの確保
        for info in kernel_infos {
            if info.output_buffer.starts_with("tmp")
                && !allocated_buffers.contains(&info.output_buffer)
            {
                let ptr_dtype = AstDType::Ptr(Box::new(info.output_dtype.clone()));
                if scope
                    .declare(
                        info.output_buffer.clone(),
                        ptr_dtype.clone(),
                        Mutability::Mutable,
                    )
                    .is_ok()
                {
                    let alloc_expr = AstNode::Allocate {
                        dtype: Box::new(info.output_dtype.clone()),
                        size: Box::new(info.output_size.clone()),
                    };
                    statements.push(assign(&info.output_buffer, alloc_expr));
                    allocated_buffers.insert(info.output_buffer.clone());
                }
            }
        }

        // カーネル呼び出し（バリア挿入付き）
        for (i, info) in kernel_infos.iter().enumerate() {
            let mut args: Vec<AstNode> = Vec::new();

            // 入力バッファー
            for buf in &info.input_buffers {
                args.push(var(buf));
            }
            // 出力バッファー
            args.push(var(&info.output_buffer));

            statements.push(AstNode::Call {
                name: info.kernel_name.clone(),
                args,
            });

            // 最後のカーネル以外の後にバリアを挿入
            // バリアは次のカーネルがこのカーネルの出力を読む前に
            // 書き込みが完了していることを保証する
            if i < kernel_infos.len() - 1 {
                statements.push(AstNode::Barrier);
            }
        }

        // 中間バッファーの解放
        for buffer_name in &allocated_buffers {
            statements.push(AstNode::Deallocate {
                ptr: Box::new(var(buffer_name)),
            });
        }

        let body = block(statements, scope);

        function(Some("harp_main"), params, AstDType::Tuple(vec![]), body)
    }

    fn graph_dtype_to_ast(dtype: &GraphDType) -> AstDType {
        match dtype {
            GraphDType::Bool => AstDType::Bool,
            GraphDType::I32 => AstDType::Int,
            GraphDType::F32 => AstDType::F32,
            GraphDType::Complex => AstDType::F32, // 複素数は2xF32として扱う
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
        if let Some(merged_graph) = self.merge_to_program(graph) {
            log::debug!("KernelMergeSuggester: merged Custom nodes into Program");
            vec![merged_graph]
        } else {
            vec![]
        }
    }
}

/// カーネル情報（main関数生成用）
struct KernelInfo {
    kernel_name: String,
    input_buffers: Vec<String>,
    output_buffer: String,
    output_dtype: AstDType,
    output_size: AstNode,
}

/// 既存のカーネル（Programから抽出）
#[derive(Clone)]
struct ExistingKernel {
    function: AstNode,
    original_name: String,
}

/// Programから抽出されたカーネル情報
struct ExtractedKernels {
    kernels: Vec<ExistingKernel>,
    kernel_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    #[test]
    fn test_kernel_merge_basic() {
        // a + b と c * d の2つの演算を持つグラフを作成
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        // Customノードを手動で作成（通常はLoweringSuggesterが生成）
        use crate::ast::helper::wildcard;
        let sum = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let result = sum.custom_elementwise_binary(c, wildcard("0") * wildcard("1"));
        graph.output("result", result);

        let suggester = KernelMergeSuggester::new();
        let suggestions = suggester.suggest(&graph);

        // マージが提案されることを確認
        assert!(
            !suggestions.is_empty(),
            "KernelMergeSuggester should suggest a merge"
        );

        // マージ後のグラフを確認
        let merged = &suggestions[0];
        let outputs = merged.outputs();
        assert_eq!(outputs.len(), 1);

        // 出力がCustom(Program)であることを確認
        if let Some(output) = outputs.values().next() {
            match &output.op {
                GraphOp::Custom { ast } => {
                    assert!(matches!(ast, AstNode::Program { .. }), "Should be Program");
                }
                _ => panic!("Output should be Custom node"),
            }
        }
    }

    #[test]
    fn test_kernel_merge_with_barriers() {
        use crate::ast::helper::wildcard;

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        // 3つのCustomノードを作成（チェーン状の依存関係）
        let sum1 = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let sum2 = sum1.custom_elementwise_binary(c.clone(), wildcard("0") * wildcard("1"));
        let result = sum2.custom_elementwise_binary(c, wildcard("0") + wildcard("1"));
        graph.output("result", result);

        let suggester = KernelMergeSuggester::new();
        let suggestions = suggester.suggest(&graph);

        assert!(!suggestions.is_empty());

        // Programを検査してバリアが挿入されていることを確認
        let merged = &suggestions[0];
        if let Some(output) = merged.outputs().values().next() {
            if let GraphOp::Custom { ast } = &output.op {
                if let AstNode::Program { functions, .. } = ast {
                    // main関数を探す
                    let main_fn = functions.iter().find(|f| {
                        matches!(f, AstNode::Function { name: Some(n), .. } if n == "harp_main")
                    });

                    assert!(main_fn.is_some(), "Should have harp_main function");

                    if let Some(AstNode::Function { body, .. }) = main_fn {
                        // bodyの中にBarrierがあることを確認
                        let has_barrier = contains_barrier(body);
                        assert!(has_barrier, "Main function should contain barriers");
                    }
                }
            }
        }
    }

    /// ASTノード内にBarrierが含まれているかチェック
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

        // 3つの演算
        let sum = &a + &b;
        let prod = &sum * &c;
        let result = &prod + 1.0f32;
        graph.output("result", result);

        // Suggesterを作成
        let suggester = create_graph_suggester(SuggesterFlags::new());
        let estimator = SimpleCostEstimator::new();
        let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(4)
            .with_max_steps(100);

        let optimized = optimizer.optimize(graph);

        // 最適化後のグラフを確認
        let mut custom_function_count = 0;
        let mut custom_program_count = 0;

        fn count_customs(
            node: &GraphNode,
            visited: &mut std::collections::HashSet<*const GraphNodeData>,
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

        let mut visited = std::collections::HashSet::new();
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

        // 3つのElementwise演算があり、FusionSuggesterで全てfusionされて
        // 1つのCustom(Function)になるはず
        // この場合、KernelMergeSuggesterは複数のCustomがないためトリガーされない
        assert!(
            custom_function_count == 1 || custom_program_count >= 1,
            "Expected either 1 Custom(Function) after fusion, or 1 Custom(Program) after merge"
        );
    }

    #[test]
    fn test_kernel_merge_with_reduce() {
        use crate::backend::pipeline::{SuggesterFlags, create_graph_suggester};
        use crate::graph::ReduceOp;
        use crate::opt::graph::{BeamSearchGraphOptimizer, GraphOptimizer, SimpleCostEstimator};

        let _ = env_logger::try_init();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 10]);
        let b = graph.input("b", DType::F32, vec![10, 10]);

        // Reduce演算は式レベルでの融合ができないため、複数のCustomノードが生成される
        // a.reduce_sum(0) + b.reduce_sum(0) のようなパターン
        let sum_a = a.reduce(ReduceOp::Sum, 0); // [10]
        let sum_b = b.reduce(ReduceOp::Sum, 0); // [10]

        // 2つのReduceの結果を足す
        let result = sum_a + sum_b;
        graph.output("result", result);

        // Suggesterを作成
        let suggester = create_graph_suggester(SuggesterFlags::new());
        let estimator = SimpleCostEstimator::new();
        let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(4)
            .with_max_steps(100);

        let optimized = optimizer.optimize(graph);

        // 最適化後のグラフを確認
        let mut custom_function_count = 0;
        let mut custom_program_count = 0;

        fn count_customs(
            node: &GraphNode,
            visited: &mut std::collections::HashSet<*const GraphNodeData>,
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

        let mut visited = std::collections::HashSet::new();
        for output in optimized.outputs().values() {
            count_customs(
                output,
                &mut visited,
                &mut custom_function_count,
                &mut custom_program_count,
            );
        }

        println!(
            "With Reduce - Custom(Function): {}, Custom(Program): {}",
            custom_function_count, custom_program_count
        );

        // オプティマイザーはコストが最も低いパスを選択する
        // FusionSuggesterで全て融合できる場合は1つのCustom(Function)になる
        // KernelMergeSuggesterはFusionで融合できないケースで有効
        // いずれの場合も、最終的には少なくとも1つのCustomノードが生成される
        assert!(
            custom_function_count >= 1 || custom_program_count >= 1,
            "Expected at least 1 Custom node"
        );
    }

    /// KernelMergeSuggesterが直接正しく動作することを確認するテスト
    /// （オプティマイザーを通さずに直接テスト）
    #[test]
    fn test_kernel_merge_suggester_direct() {
        use crate::ast::helper::wildcard;

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);
        let d = graph.input("d", DType::F32, vec![10]);

        // 2つの独立したCustomノードを作成
        // これらはFusionSuggesterでは融合できない（依存関係がない）
        let custom1 = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let custom2 = c.custom_elementwise_binary(d, wildcard("0") * wildcard("1"));

        // 2つのCustomの結果を使う
        let result = custom1 + custom2;
        graph.output("result", result);

        // KernelMergeSuggesterを直接テスト
        let suggester = KernelMergeSuggester::new();

        // まず、Customノード数を確認
        let (fn_count, prog_count) = suggester.count_custom_nodes(&graph);

        println!(
            "Before merge: {} Custom(Function), {} Custom(Program)",
            fn_count, prog_count
        );
        assert_eq!(fn_count, 2, "Should have 2 Custom(Function) nodes");

        // マージを実行
        let suggestions = suggester.suggest(&graph);
        assert!(
            !suggestions.is_empty(),
            "KernelMergeSuggester should suggest a merge"
        );

        // マージ後のグラフを確認
        let merged = &suggestions[0];
        let (fn_count_after, prog_count_after) = suggester.count_custom_nodes(merged);

        println!(
            "After merge: Custom(Function): {}, Custom(Program): {}",
            fn_count_after, prog_count_after
        );
        assert_eq!(
            prog_count_after, 1,
            "Should have exactly 1 Custom(Program) node"
        );
    }

    /// Custom(Program)とCustom(Function)の増分マージをテスト
    #[test]
    fn test_incremental_merge_program_and_function() {
        use crate::ast::helper::{block as ast_block, const_int, load, range, store, wildcard};
        use crate::ast::{DType as AstDType, Scope};
        use crate::graph::shape::View;

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);

        // 最初の2つの演算をCustom(Program)として作成
        let kernel_body = ast_block(
            vec![range(
                "i",
                const_int(0),
                const_int(1),
                const_int(10),
                ast_block(
                    vec![store(
                        var("output"),
                        var("i"),
                        load(var("input0"), var("i"), AstDType::F32)
                            + load(var("input1"), var("i"), AstDType::F32),
                    )],
                    Scope::new(),
                ),
            )],
            Scope::new(),
        );

        let kernel = AstNode::Kernel {
            name: Some("kernel_0".to_string()),
            params: vec![
                VarDecl {
                    name: "input0".to_string(),
                    dtype: AstDType::Ptr(Box::new(AstDType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "input1".to_string(),
                    dtype: AstDType::Ptr(Box::new(AstDType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: AstDType::Ptr(Box::new(AstDType::F32)),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(kernel_body),
            thread_group_size: 64,
        };

        let main_body = ast_block(
            vec![AstNode::Call {
                name: "kernel_0".to_string(),
                args: vec![var("input0"), var("input1"), var("output")],
            }],
            Scope::new(),
        );

        let main_fn = function(
            Some("harp_main"),
            vec![
                VarDecl {
                    name: "input0".to_string(),
                    dtype: AstDType::Ptr(Box::new(AstDType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "input1".to_string(),
                    dtype: AstDType::Ptr(Box::new(AstDType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: AstDType::Ptr(Box::new(AstDType::F32)),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            AstDType::Tuple(vec![]),
            main_body,
        );

        let program = AstNode::Program {
            functions: vec![kernel, main_fn],
            entry_point: "harp_main".to_string(),
        };

        // Custom(Program)ノードを作成
        let program_node = GraphNode::new(
            DType::F32,
            GraphOp::Custom { ast: program },
            vec![a, b],
            View::contiguous(vec![10]),
        );

        // 追加のCustom(Function)を作成
        let final_result = program_node.custom_elementwise_binary(c, wildcard("0") * wildcard("1"));
        graph.output("result", final_result);

        let suggester = KernelMergeSuggester::new();
        let (fn_count, prog_count) = suggester.count_custom_nodes(&graph);

        println!(
            "Before incremental merge: {} Custom(Function), {} Custom(Program)",
            fn_count, prog_count
        );

        // マージが可能であることを確認
        assert!(
            suggester.can_merge(&graph),
            "Should be able to merge Program + Function"
        );

        // マージを実行
        let suggestions = suggester.suggest(&graph);
        assert!(!suggestions.is_empty(), "Should suggest a merge");

        let merged = &suggestions[0];
        let (fn_after, prog_after) = suggester.count_custom_nodes(merged);

        println!(
            "After incremental merge: {} Custom(Function), {} Custom(Program)",
            fn_after, prog_after
        );

        // 結果は1つのCustom(Program)になるはず
        assert_eq!(prog_after, 1, "Should have 1 Custom(Program)");
        assert_eq!(fn_after, 0, "Should have 0 Custom(Function)");
    }
}
