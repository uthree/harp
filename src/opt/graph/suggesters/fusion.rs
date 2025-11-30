use crate::ast::{AstNode, helper::wildcard};
use crate::graph::ops::ElementwiseOp;
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// ノード融合を提案するSuggester
pub struct FusionSuggester;

impl FusionSuggester {
    /// 新しいFusionSuggesterを作成
    pub fn new() -> Self {
        Self
    }

    /// グラフ内の各ノードの被参照数をカウント
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

            // ソースノードの被参照数をカウント
            for src in &node.src {
                let src_ptr = src.as_ptr();
                *ref_counts.entry(src_ptr).or_insert(0) += 1;
                visit(src, ref_counts, visited);
            }
        }

        for output in graph.outputs().values() {
            visit(output, &mut ref_counts, &mut visited);
        }

        ref_counts
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

    /// ElementwiseOpをAstNodeに変換するヘルパー
    fn elementwise_op_to_ast(op: &ElementwiseOp, args: Vec<AstNode>) -> AstNode {
        match op {
            ElementwiseOp::Add => {
                assert_eq!(args.len(), 2);
                args[0].clone() + args[1].clone()
            }
            ElementwiseOp::Mul => {
                assert_eq!(args.len(), 2);
                args[0].clone() * args[1].clone()
            }
            ElementwiseOp::Max => {
                assert_eq!(args.len(), 2);
                AstNode::Max(Box::new(args[0].clone()), Box::new(args[1].clone()))
            }
            ElementwiseOp::Rem => {
                assert_eq!(args.len(), 2);
                args[0].clone() % args[1].clone()
            }
            ElementwiseOp::Idiv => {
                assert_eq!(args.len(), 2);
                args[0].clone() / args[1].clone()
            }
            ElementwiseOp::Neg => {
                assert_eq!(args.len(), 1);
                -args[0].clone()
            }
            ElementwiseOp::Recip => {
                assert_eq!(args.len(), 1);
                AstNode::Recip(Box::new(args[0].clone()))
            }
            ElementwiseOp::Log2 => {
                assert_eq!(args.len(), 1);
                AstNode::Log2(Box::new(args[0].clone()))
            }
            ElementwiseOp::Exp2 => {
                assert_eq!(args.len(), 1);
                AstNode::Exp2(Box::new(args[0].clone()))
            }
            ElementwiseOp::Sin => {
                assert_eq!(args.len(), 1);
                AstNode::Sin(Box::new(args[0].clone()))
            }
            ElementwiseOp::Sqrt => {
                assert_eq!(args.len(), 1);
                AstNode::Sqrt(Box::new(args[0].clone()))
            }
        }
    }

    /// AstNode内のWildcardインデックスを再マッピング
    fn remap_wildcards(expr: &AstNode, old_to_new: &HashMap<usize, usize>) -> AstNode {
        use crate::ast::AstNode::*;
        match expr {
            Wildcard(name) => {
                if let Ok(old_idx) = name.parse::<usize>() {
                    if let Some(&new_idx) = old_to_new.get(&old_idx) {
                        wildcard(new_idx.to_string())
                    } else {
                        expr.clone()
                    }
                } else {
                    expr.clone()
                }
            }
            // Binary operations
            Add(a, b) => Add(
                Box::new(Self::remap_wildcards(a, old_to_new)),
                Box::new(Self::remap_wildcards(b, old_to_new)),
            ),
            Mul(a, b) => Mul(
                Box::new(Self::remap_wildcards(a, old_to_new)),
                Box::new(Self::remap_wildcards(b, old_to_new)),
            ),
            Rem(a, b) => Rem(
                Box::new(Self::remap_wildcards(a, old_to_new)),
                Box::new(Self::remap_wildcards(b, old_to_new)),
            ),
            Idiv(a, b) => Idiv(
                Box::new(Self::remap_wildcards(a, old_to_new)),
                Box::new(Self::remap_wildcards(b, old_to_new)),
            ),
            Max(a, b) => Max(
                Box::new(Self::remap_wildcards(a, old_to_new)),
                Box::new(Self::remap_wildcards(b, old_to_new)),
            ),
            // Unary operations
            Recip(a) => Recip(Box::new(Self::remap_wildcards(a, old_to_new))),
            Log2(a) => Log2(Box::new(Self::remap_wildcards(a, old_to_new))),
            Exp2(a) => Exp2(Box::new(Self::remap_wildcards(a, old_to_new))),
            Sin(a) => Sin(Box::new(Self::remap_wildcards(a, old_to_new))),
            Sqrt(a) => Sqrt(Box::new(Self::remap_wildcards(a, old_to_new))),
            Cast(a, dtype) => Cast(
                Box::new(Self::remap_wildcards(a, old_to_new)),
                dtype.clone(),
            ),
            // 他のノードはそのまま（Const, Var, etc.）
            _ => expr.clone(),
        }
    }

    /// Elementwise演算チェーンを検出して融合可能な場合にパターンを返す
    ///
    /// 連続する2つのElementwise演算を融合する
    /// 例: (a + b) * c -> FusedElementwise([a, b, c], Wildcard("0") + Wildcard("1") * Wildcard("2"))
    fn detect_elementwise_chain(&self, node: &GraphNode) -> Option<(Vec<GraphNode>, AstNode)> {
        // このノードがElementwiseでない場合はNone
        let current_op = match &node.op {
            GraphOp::Elementwise { op, .. } => op.clone(),
            _ => return None,
        };

        // 入力のうち、少なくとも1つがElementwiseまたはFusedElementwiseの場合に融合可能
        let mut found_fusable_input = false;
        for src in &node.src {
            if matches!(
                src.op,
                GraphOp::Elementwise { .. } | GraphOp::FusedElementwise { .. }
            ) {
                found_fusable_input = true;
                break;
            }
        }

        if !found_fusable_input {
            return None;
        }

        // 融合するノードチェーンを構築
        let mut graph_inputs = Vec::new();
        let mut input_mapping: HashMap<*const GraphNodeData, usize> = HashMap::new();

        // 入力ノードを処理し、AstNode式を構築
        let mut current_args = Vec::new();

        for src in &node.src {
            match &src.op {
                GraphOp::Elementwise { op, .. } => {
                    // Elementwiseノードの場合、その入力を展開し式を構築
                    let mut sub_args = Vec::new();
                    for sub_src in &src.src {
                        let ptr = sub_src.as_ptr();
                        let idx = if let Some(&existing_idx) = input_mapping.get(&ptr) {
                            existing_idx
                        } else {
                            let new_idx = graph_inputs.len();
                            input_mapping.insert(ptr, new_idx);
                            graph_inputs.push(sub_src.clone());
                            new_idx
                        };
                        sub_args.push(wildcard(idx.to_string()));
                    }

                    // 中間演算の結果をAstNodeとして構築
                    let sub_expr = Self::elementwise_op_to_ast(op, sub_args);
                    current_args.push(sub_expr);
                }
                GraphOp::FusedElementwise { expr, .. } => {
                    // FusedElementwiseノードの場合、その入力を展開し式を再マッピング
                    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
                    for (old_idx, sub_src) in src.src.iter().enumerate() {
                        let ptr = sub_src.as_ptr();
                        let new_idx = if let Some(&existing_idx) = input_mapping.get(&ptr) {
                            existing_idx
                        } else {
                            let idx = graph_inputs.len();
                            input_mapping.insert(ptr, idx);
                            graph_inputs.push(sub_src.clone());
                            idx
                        };
                        old_to_new.insert(old_idx, new_idx);
                    }

                    // 式のWildcardインデックスを再マッピング
                    let remapped_expr = Self::remap_wildcards(expr, &old_to_new);
                    current_args.push(remapped_expr);
                }
                _ => {
                    // 通常の入力の場合
                    let ptr = src.as_ptr();
                    let idx = if let Some(&existing_idx) = input_mapping.get(&ptr) {
                        existing_idx
                    } else {
                        let new_idx = graph_inputs.len();
                        input_mapping.insert(ptr, new_idx);
                        graph_inputs.push(src.clone());
                        new_idx
                    };
                    current_args.push(wildcard(idx.to_string()));
                }
            }
        }

        // 現在のノードの演算をAstNodeとして構築
        let final_expr = Self::elementwise_op_to_ast(&current_op, current_args);

        // 元の入力ノードが複数使われている（融合がある）場合のみ適用
        // つまり、graph_inputsの数がnodeの直接入力数より少ない = 融合が発生
        if graph_inputs.len() >= node.src.len() {
            // 融合によるメリットがない場合もチェック
            // 少なくとも1つのElementwiseまたはFusedElementwise入力が展開されているか確認
            let has_expansion = node.src.iter().any(|s| {
                matches!(
                    &s.op,
                    GraphOp::Elementwise { .. } | GraphOp::FusedElementwise { .. }
                )
            });
            if !has_expansion {
                return None;
            }
        }

        Some((graph_inputs, final_expr))
    }

    /// 連続するView変更を検出
    ///
    /// View(v1) -> View(v2) のようなパターンを検出し、View(v2)だけに簡略化
    fn detect_consecutive_views(&self, node: &GraphNode) -> Option<GraphNode> {
        // このノードがViewでない場合はNone
        let final_view = match &node.op {
            GraphOp::View(v) => v.clone(),
            _ => return None,
        };

        // 入力が1つでない場合はパターンに一致しない
        if node.src.len() != 1 {
            return None;
        }

        let input = &node.src[0];

        // 入力もViewの場合、融合可能
        if matches!(input.op, GraphOp::View(_)) {
            // さらにその入力（元のデータソース）を取得
            if input.src.len() == 1 {
                let original_source = &input.src[0];

                // 最終的なViewだけを適用した新しいノードを作成
                let fused_node = GraphNode::new(
                    node.dtype.clone(),
                    GraphOp::View(final_view),
                    vec![original_source.clone()],
                    node.view.clone(),
                );

                return Some(fused_node);
            }
        }

        None
    }

    /// Elementwise -> Reduceパターンを検出
    fn detect_elementwise_reduce_pattern(
        &self,
        node: &GraphNode,
    ) -> Option<(Vec<GraphNode>, AstNode, crate::graph::ops::ReduceOp, usize)> {
        // このノードがReduceでない場合はNone
        let (reduce_op, axis) = match &node.op {
            GraphOp::Reduce { op, axis, .. } => (op.clone(), *axis),
            _ => return None,
        };

        // 入力がElementwiseの場合、融合可能
        if node.src.len() != 1 {
            return None;
        }

        let input = &node.src[0];
        let elementwise_op = match &input.op {
            GraphOp::Elementwise { op, .. } => op.clone(),
            _ => return None,
        };

        // Elementwise入力のさらに入力をgraph_inputsとして収集
        let graph_inputs = input.src.clone();

        // AstNode式を構築: Wildcard("0"), Wildcard("1"), ... を入力として使用
        let args: Vec<AstNode> = (0..graph_inputs.len())
            .map(|i| wildcard(i.to_string()))
            .collect();

        let expr = Self::elementwise_op_to_ast(&elementwise_op, args);

        Some((graph_inputs, expr, reduce_op, axis))
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

            // Inputノードは常に元のノードをそのまま返す（再構築しない）
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

        // 入力メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // Sinkノードがある場合は、Program構造を保持しながらsrcを再構築
        if let Some(old_sink) = graph.sink() {
            let new_sink_src: Vec<GraphNode> = old_sink
                .src
                .iter()
                .map(|src| rebuild_node(src, &node_map, &mut visited))
                .collect();

            // 元のSinkのast（Program）とoutputsを保持して新しいSinkを作成
            if let GraphOp::Sink { ast, outputs } = &old_sink.op {
                let new_sink = GraphNode::new(
                    old_sink.dtype.clone(),
                    GraphOp::Sink {
                        ast: ast.clone(),
                        outputs: outputs.clone(),
                    },
                    new_sink_src,
                    old_sink.view.clone(),
                );
                new_graph.set_sink(new_sink);
            }
        } else {
            // Sinkがない場合は従来通りoutputsを使用
            let outputs_map = graph.outputs();
            let mut outputs: Vec<_> = outputs_map.iter().collect();
            outputs.sort_by_key(|(name, _)| name.as_str());

            for (name, output_node) in outputs {
                let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
                new_graph.output(name, rebuilt);
            }
        }

        // shape変数のデフォルト値をコピー
        for (name, value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(name.clone(), *value);
        }

        new_graph
    }
}

impl Default for FusionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for FusionSuggester {
    fn name(&self) -> &'static str {
        "Fusion"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);
        let ref_counts = self.count_node_references(graph);

        for node in &nodes {
            // 連続するView変更を検出して融合
            if let Some(fused_node) = self.detect_consecutive_views(node) {
                // 融合される中間Viewノードの被参照数をチェック
                if node.src.len() == 1 {
                    let input_ptr = node.src[0].as_ptr();
                    let ref_count = ref_counts.get(&input_ptr).copied().unwrap_or(0);
                    // 被参照数が1より大きい場合は融合しない
                    if ref_count > 1 {
                        continue;
                    }
                }

                let new_graph = self.replace_node_in_graph(graph, node, fused_node);
                suggestions.push(new_graph);
            }

            // Elementwise -> Reduceパターンを検出して融合
            if let Some((graph_inputs, expr, reduce_op, axis)) =
                self.detect_elementwise_reduce_pattern(node)
            {
                // 融合されるElementwiseノードの被参照数をチェック
                if node.src.len() == 1 {
                    let input_ptr = node.src[0].as_ptr();
                    let ref_count = ref_counts.get(&input_ptr).copied().unwrap_or(0);
                    // 被参照数が1より大きい場合は融合しない
                    if ref_count > 1 {
                        continue;
                    }
                }

                // FusedElementwiseReduceノードを作成
                let fused_node = crate::graph::ops::fused_elementwise_reduce(
                    graph_inputs,
                    expr,
                    reduce_op,
                    axis,
                );

                let new_graph = self.replace_node_in_graph(graph, node, fused_node);
                suggestions.push(new_graph);
            }

            // Elementwise チェーンを検出して融合
            if let Some((graph_inputs, expr)) = self.detect_elementwise_chain(node) {
                // 融合されるElementwiseまたはFusedElementwiseノードの被参照数をチェック
                // チェーン内の全ノードが1回しか参照されていない場合のみ融合
                let can_fuse = node.src.iter().all(|src| {
                    if matches!(
                        src.op,
                        GraphOp::Elementwise { .. } | GraphOp::FusedElementwise { .. }
                    ) {
                        let src_ptr = src.as_ptr();
                        let ref_count = ref_counts.get(&src_ptr).copied().unwrap_or(0);
                        ref_count <= 1
                    } else {
                        true
                    }
                });

                if !can_fuse {
                    continue;
                }

                // FusedElementwiseノードを作成
                let fused_node = crate::graph::ops::fused_elementwise(graph_inputs, expr);

                let new_graph = self.replace_node_in_graph(graph, node, fused_node);
                suggestions.push(new_graph);
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_fusion_suggester() {
        let suggester = FusionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let b = graph.input("b", DType::F32, vec![10, 20]);

        // (a + b).reduce_sum(0) -> FusedElementwiseReduce
        let c = (a + b).reduce_sum(0);
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // Elementwise -> Reduceパターンが検出され、1つの候補が生成されるはず
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_fusion_suggester_no_pattern() {
        let suggester = FusionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        // 単純なReduce（Elementwiseなし）
        let c = a.reduce_sum(0);
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // パターンがないため候補なし
        assert_eq!(suggestions.len(), 0);
    }

    // Note: test_no_fusion_multiple_references は複数出力が
    // 現在サポートされていないため削除されました。
    // 詳細は spec/TODO.md を参照してください。
}
