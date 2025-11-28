use crate::ast::{AstNode, helper::wildcard};
use crate::graph::ops::{CustomKind, ElementwiseOp};
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// 連続するElementwise演算をGraphOp::Customに融合するSuggester
///
/// 例: (a + b) * c -> Custom { ast: Mul(Add(W("0"), W("1")), W("2")), kind: Elementwise }
pub struct CustomFusionSuggester;

impl CustomFusionSuggester {
    /// 新しいCustomFusionSuggesterを作成
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

    /// 既存のCustomノード、Elementwiseノード、FusedElementwiseノードからAST式を取得
    fn node_to_ast_expr(
        node: &GraphNode,
        wildcard_base: &str,
    ) -> Option<(AstNode, Vec<GraphNode>)> {
        match &node.op {
            GraphOp::Elementwise { op, .. } => {
                let args: Vec<AstNode> = (0..node.src.len())
                    .map(|i| wildcard(format!("{}{}", wildcard_base, i)))
                    .collect();
                let expr = Self::elementwise_op_to_ast(op, args);
                Some((expr, node.src.clone()))
            }
            GraphOp::FusedElementwise { expr, .. } => Some((expr.clone(), node.src.clone())),
            GraphOp::Custom {
                ast,
                kind: CustomKind::Elementwise,
                ..
            } => Some((ast.clone(), node.src.clone())),
            _ => None,
        }
    }

    /// Elementwise演算チェーンを検出して融合可能な場合にパターンを返す
    ///
    /// 連続する2つのElementwise/Custom演算を融合する
    /// 例: (a + b) * c -> Custom([a, b, c], Mul(Add(W("0"), W("1")), W("2")))
    fn detect_elementwise_chain(&self, node: &GraphNode) -> Option<(Vec<GraphNode>, AstNode)> {
        // このノードがElementwise/FusedElementwise/Customでない場合はNone
        let (current_expr, _) = Self::node_to_ast_expr(node, "")?;

        // 入力のうち、少なくとも1つがElementwise/FusedElementwise/Customの場合に融合可能
        let found_fuseable_input = node.src.iter().any(|src| {
            matches!(
                &src.op,
                GraphOp::Elementwise { .. }
                    | GraphOp::FusedElementwise { .. }
                    | GraphOp::Custom {
                        kind: CustomKind::Elementwise,
                        ..
                    }
            )
        });

        if !found_fuseable_input {
            return None;
        }

        // 融合するノードチェーンを構築
        let mut graph_inputs = Vec::new();
        let mut input_mapping: HashMap<*const GraphNodeData, usize> = HashMap::new();

        // 入力ノードを処理し、AstNode式を構築
        let mut current_args = Vec::new();

        for src in &node.src {
            if let Some((src_expr, src_inputs)) = Self::node_to_ast_expr(src, "") {
                // Elementwise/FusedElementwise/Customノードの場合、その入力を展開し式を構築
                let mut sub_args = Vec::new();
                for sub_src in &src_inputs {
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

                // 中間演算の結果をAstNodeとして構築（Wildcardを置換）
                let src_expr_mapping: HashMap<String, AstNode> = src_inputs
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (i.to_string(), sub_args[i].clone()))
                    .collect();
                let substituted_expr = src_expr.substitute(&src_expr_mapping);
                current_args.push(substituted_expr);
            } else {
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

        // 現在のノードの演算をAstNodeとして構築（Wildcardを置換）
        let node_expr_mapping: HashMap<String, AstNode> = node
            .src
            .iter()
            .enumerate()
            .map(|(i, _)| (i.to_string(), current_args[i].clone()))
            .collect();
        let final_expr = current_expr.substitute(&node_expr_mapping);

        Some((graph_inputs, final_expr))
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
            if matches!(node.op, GraphOp::Input) {
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

            GraphNode::with_elementwise_strategies(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
                node.elementwise_strategies.clone(),
            )
        }

        let mut new_graph = Graph::new();

        // 入力ノードを保持
        for (name, weak_input) in graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                new_graph.register_input(name.clone(), input_node);
            }
        }

        // 出力ノードを名前順でソートして再構築
        let mut outputs: Vec<_> = graph.outputs().iter().collect();
        outputs.sort_by_key(|(name, _)| name.as_str());

        for (name, output_node) in outputs {
            let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
            new_graph.output(name, rebuilt);
        }

        new_graph
    }
}

impl Default for CustomFusionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for CustomFusionSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);
        let ref_counts = self.count_node_references(graph);

        for node in &nodes {
            // Elementwise チェーンを検出してGraphOp::Customに融合
            if let Some((graph_inputs, expr)) = self.detect_elementwise_chain(node) {
                // 融合されるノードの被参照数をチェック
                // チェーン内の全ノードが1回しか参照されていない場合のみ融合
                let can_fuse = node.src.iter().all(|src| {
                    let is_fuseable = matches!(
                        &src.op,
                        GraphOp::Elementwise { .. }
                            | GraphOp::FusedElementwise { .. }
                            | GraphOp::Custom {
                                kind: CustomKind::Elementwise,
                                ..
                            }
                    );
                    if is_fuseable {
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

                // GraphOp::Customノードを作成
                let fused_node = GraphNode::new(
                    node.dtype.clone(),
                    GraphOp::Custom {
                        ast: expr,
                        kind: CustomKind::Elementwise,
                        elementwise_strategies: None,
                    },
                    graph_inputs,
                    node.view.clone(),
                );

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
    fn test_custom_fusion_suggester_basic() {
        let suggester = CustomFusionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = graph.input("c", DType::F32, vec![10, 20]);

        // (a + b) * c -> Custom { ast: Mul(Add(W("0"), W("1")), W("2")) }
        let sum = a + b;
        let result = sum * c;
        graph.output("result", result);

        let suggestions = suggester.suggest(&graph);

        // Elementwiseチェーンが検出され、1つの候補が生成されるはず
        assert_eq!(suggestions.len(), 1);

        // 融合後のグラフを確認
        let fused_graph = &suggestions[0];
        let output = fused_graph.outputs().get("result").unwrap();

        // Customノードであることを確認
        match &output.op {
            GraphOp::Custom {
                kind: CustomKind::Elementwise,
                ..
            } => {}
            _ => panic!("Expected Custom node with Elementwise kind"),
        }

        // 入力が3つであることを確認
        assert_eq!(output.src.len(), 3);
    }

    #[test]
    fn test_custom_fusion_no_pattern() {
        let suggester = CustomFusionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);

        // 単純なElementwise（融合パターンなし）
        let result = a + b;
        graph.output("result", result);

        let suggestions = suggester.suggest(&graph);

        // パターンがないため候補なし
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_no_fusion_multiple_references() {
        let suggester = CustomFusionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = graph.input("c", DType::F32, vec![10, 20]);

        // (a + b)を計算
        let sum = a + b;

        // sumを2つの異なる演算で使用（複数の被参照）
        let result1 = sum.clone() * c.clone();
        let result2 = sum.clone() - c;

        graph.output("result1", result1);
        graph.output("result2", result2);

        let suggestions = suggester.suggest(&graph);

        // sumが複数回参照されているため、融合は提案されないはず
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_fusion_with_three_operations() {
        let suggester = CustomFusionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = graph.input("c", DType::F32, vec![10]);
        let d = graph.input("d", DType::F32, vec![10]);

        // ((a + b) * c) - d
        let sum = a + b;
        let mul = sum * c;
        let result = mul - d;
        graph.output("result", result);

        let suggestions = suggester.suggest(&graph);

        // 複数の融合候補が生成される可能性がある
        assert!(!suggestions.is_empty());
    }
}
