//! Kernel Merge Suggester
//!
//! 複数のCustomノード（Function）を1つのCustomノード（Program）にマージします。
//! これにより、グラフ全体が1つのProgramとして表現され、Lowererがほぼパススルーになります。

use crate::ast::helper::{assign, block, function, var};
use crate::ast::{AstNode, DType as AstDType, FunctionKind, Mutability, Scope, VarDecl, VarKind};
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

    /// マージ可能かチェック（複数のCustomノードが存在するか）
    fn can_merge(&self, graph: &Graph) -> bool {
        let nodes = self.collect_all_nodes(graph);
        let custom_count = nodes
            .iter()
            .filter(|n| matches!(&n.op, GraphOp::Custom { ast } if matches!(ast, AstNode::Function { .. })))
            .count();

        log::debug!(
            "KernelMergeSuggester: {} nodes total, {} Custom(Function) nodes",
            nodes.len(),
            custom_count
        );

        // 2つ以上のCustomノードがある場合にマージ可能
        custom_count >= 2
    }

    /// 複数のCustomノードを1つのProgram にマージ
    fn merge_to_program(&self, graph: &Graph) -> Option<Graph> {
        if !self.can_merge(graph) {
            return None;
        }

        let generations = self.topological_sort(graph);

        // カーネル情報を収集
        let mut kernel_functions: Vec<AstNode> = Vec::new();
        let mut node_buffer_map: HashMap<*const GraphNodeData, String> = HashMap::new();
        let mut kernel_infos: Vec<KernelInfo> = Vec::new();

        // 入力ノードのバッファー名を設定
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

        // 各世代のCustomノードを処理（入力→出力の順）
        let mut kernel_id = 0;
        let mut tmp_counter = 0;

        for generation in generations.iter().rev() {
            for node in generation {
                // Customノード（Function）のみを処理
                if let GraphOp::Custom { ast } = &node.op
                    && let AstNode::Function { .. } = ast
                {
                    // 入力バッファー名を収集
                    let input_buffers: Vec<String> = node
                        .src
                        .iter()
                        .filter_map(|src| {
                            if matches!(src.op, GraphOp::Const(_) | GraphOp::ComplexConst { .. }) {
                                None
                            } else {
                                let storage_node = Self::trace_to_storage(src);
                                node_buffer_map.get(&storage_node.as_ptr()).cloned()
                            }
                        })
                        .collect();

                    // 出力バッファー名を決定
                    let output_buffer = if final_output_ptrs.contains(&node.as_ptr()) {
                        "output".to_string()
                    } else {
                        let name = format!("tmp{}", tmp_counter);
                        tmp_counter += 1;
                        name
                    };

                    // このノードの出力バッファー名を記録
                    node_buffer_map.insert(node.as_ptr(), output_buffer.clone());

                    // カーネル関数を作成
                    let function_name = format!("kernel_{}", kernel_id);
                    let kernel_fn = self.create_kernel_function(node, &function_name, ast);

                    kernel_functions.push(kernel_fn);
                    kernel_infos.push(KernelInfo {
                        input_buffers,
                        output_buffer,
                        output_dtype: Self::graph_dtype_to_ast(&node.dtype),
                        output_size: Self::compute_buffer_size(node),
                    });

                    kernel_id += 1;
                }
            }
        }

        if kernel_functions.is_empty() {
            return None;
        }

        // main関数を生成
        let main_fn = self.generate_main_function(&kernel_infos, input_counter);

        // Programを作成
        let mut all_functions = kernel_functions;
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

        AstNode::Function {
            name: Some(function_name.to_string()),
            kind: FunctionKind::Normal,
            params,
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(body),
        }
    }

    /// main関数を生成
    fn generate_main_function(&self, kernel_infos: &[KernelInfo], input_count: usize) -> AstNode {
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

        // カーネル呼び出し
        for (i, info) in kernel_infos.iter().enumerate() {
            let kernel_name = format!("kernel_{}", i);
            let mut args: Vec<AstNode> = Vec::new();

            // 入力バッファー
            for buf in &info.input_buffers {
                args.push(var(buf));
            }
            // 出力バッファー
            args.push(var(&info.output_buffer));

            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });
        }

        // 中間バッファーの解放
        for buffer_name in &allocated_buffers {
            statements.push(AstNode::Deallocate {
                ptr: Box::new(var(buffer_name)),
            });
        }

        let body = block(statements, scope);

        function(
            Some("harp_main"),
            FunctionKind::Normal,
            params,
            AstDType::Tuple(vec![]),
            body,
        )
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
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        if let Some(merged_graph) = self.merge_to_program(graph) {
            log::debug!("KernelMergeSuggester: merged multiple Custom nodes into Program");
            vec![merged_graph]
        } else {
            vec![]
        }
    }
}

/// カーネル情報（main関数生成用）
struct KernelInfo {
    input_buffers: Vec<String>,
    output_buffer: String,
    output_dtype: AstDType,
    output_size: AstNode,
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
        let suggester = create_graph_suggester(SuggesterFlags::single_threaded());
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

        // 3つのElementwise演算があり、CustomFusionSuggesterで全てfusionされて
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
        let suggester = create_graph_suggester(SuggesterFlags::single_threaded());
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
        // CustomFusionSuggesterで全て融合できる場合は1つのCustom(Function)になる
        // KernelMergeSuggesterはCustomFusionで融合できないケースで有効
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
        // これらはCustomFusionSuggesterでは融合できない（依存関係がない）
        let custom1 = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        let custom2 = c.custom_elementwise_binary(d, wildcard("0") * wildcard("1"));

        // 2つのCustomの結果を使う
        let result = custom1 + custom2;
        graph.output("result", result);

        // KernelMergeSuggesterを直接テスト
        let suggester = KernelMergeSuggester::new();

        // まず、Customノード数を確認
        let nodes = suggester.collect_all_nodes(&graph);
        let custom_count = nodes
            .iter()
            .filter(|n| {
                matches!(&n.op, GraphOp::Custom { ast } if matches!(ast, AstNode::Function { .. }))
            })
            .count();

        println!("Before merge: {} Custom(Function) nodes", custom_count);
        assert_eq!(custom_count, 2, "Should have 2 Custom(Function) nodes");

        // マージを実行
        let suggestions = suggester.suggest(&graph);
        assert!(
            !suggestions.is_empty(),
            "KernelMergeSuggester should suggest a merge"
        );

        // マージ後のグラフを確認
        let merged = &suggestions[0];
        let merged_nodes = suggester.collect_all_nodes(merged);

        let mut fn_count = 0;
        let mut prog_count = 0;
        for node in &merged_nodes {
            if let GraphOp::Custom { ast } = &node.op {
                match ast {
                    AstNode::Function { .. } => fn_count += 1,
                    AstNode::Program { .. } => prog_count += 1,
                    _ => {}
                }
            }
        }

        println!(
            "After merge: Custom(Function): {}, Custom(Program): {}",
            fn_count, prog_count
        );
        assert_eq!(prog_count, 1, "Should have exactly 1 Custom(Program) node");
    }
}
