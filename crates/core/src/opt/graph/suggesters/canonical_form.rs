//! Canonical Form Suggester
//!
//! Elementwise、Reduce、FusedElementwiseをFusedElementwiseReduceに統一変換するSuggester。
//! これによりグラフレベルでの統一的な表現が得られ、後続の最適化やloweringが簡素化されます。

use crate::ast::helper::wildcard;
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp, ReduceOp};
use crate::opt::graph::{GraphSuggester, SuggestResult};
use std::collections::{HashMap, HashSet};

use super::lowering::build_elementwise_expr;

/// Elementwise/Reduce/FusedElementwiseをFusedElementwiseReduceに統一変換するSuggester
///
/// # 変換ルール
/// - `Elementwise { op }` → `FusedElementwiseReduce { expr: op_to_ast(op), axes: [] }`
/// - `Reduce { op, axis }` → `FusedElementwiseReduce { expr: wildcard("0"), axes: [axis] }`
/// - `FusedElementwise { expr }` → `FusedElementwiseReduce { expr, axes: [] }`
pub struct CanonicalFormSuggester;

impl CanonicalFormSuggester {
    pub fn new() -> Self {
        CanonicalFormSuggester
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

    /// ノードを正規形に変換可能かチェック
    fn can_canonicalize(&self, node: &GraphNode) -> bool {
        matches!(
            node.op,
            GraphOp::Elementwise { .. } | GraphOp::Reduce { .. } | GraphOp::FusedElementwise { .. }
        )
    }

    /// ノードを正規形（FusedElementwiseReduce）に変換
    fn canonicalize(&self, node: &GraphNode) -> Option<GraphNode> {
        match &node.op {
            GraphOp::Elementwise { op } => {
                let expr = build_elementwise_expr(op);
                Some(GraphNode::new(
                    node.dtype.clone(),
                    GraphOp::FusedElementwiseReduce {
                        expr,
                        reduce_op: ReduceOp::Sum, // 未使用（axes=[]の場合）
                        axes: vec![],
                        reduce_strategy: None,
                    },
                    node.src.clone(),
                    node.view.clone(),
                ))
            }
            GraphOp::Reduce {
                op,
                axis,
                reduce_strategy,
            } => {
                let expr = wildcard("0");
                Some(GraphNode::new(
                    node.dtype.clone(),
                    GraphOp::FusedElementwiseReduce {
                        expr,
                        reduce_op: op.clone(),
                        axes: vec![*axis],
                        reduce_strategy: reduce_strategy.clone(),
                    },
                    node.src.clone(),
                    node.view.clone(),
                ))
            }
            GraphOp::FusedElementwise { expr } => Some(GraphNode::new(
                node.dtype.clone(),
                GraphOp::FusedElementwiseReduce {
                    expr: expr.clone(),
                    reduce_op: ReduceOp::Sum, // 未使用（axes=[]の場合）
                    axes: vec![],
                    reduce_strategy: None,
                },
                node.src.clone(),
                node.view.clone(),
            )),
            _ => None,
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
        node_map.insert(old_node.as_ptr(), new_node);

        let mut cache: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            cache: &mut HashMap<*const GraphNodeData, GraphNode>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

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
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        for (name, output_node) in graph.outputs() {
            let rebuilt = rebuild_node(output_node, &node_map, &mut cache);
            new_graph.set_output_node(name.clone(), rebuilt);
        }

        for (name, value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(name.clone(), *value);
        }

        new_graph
    }
}

impl Default for CanonicalFormSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for CanonicalFormSuggester {
    fn name(&self) -> &'static str {
        "CanonicalForm"
    }

    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        for node in &nodes {
            if !self.can_canonicalize(node) {
                continue;
            }

            if let Some(canonical_node) = self.canonicalize(node) {
                let new_graph = self.replace_node_in_graph(graph, node, canonical_node);
                suggestions.push(SuggestResult::new(new_graph, self.name()));
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
    fn test_canonicalize_elementwise() {
        let suggester = CanonicalFormSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b; // Elementwise Add
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);
        assert_eq!(suggestions.len(), 1);

        let new_graph = &suggestions[0].graph;
        let output = new_graph.outputs().get("c").unwrap();
        assert!(matches!(
            output.op,
            GraphOp::FusedElementwiseReduce { ref axes, .. } if axes.is_empty()
        ));
    }

    #[test]
    fn test_canonicalize_reduce() {
        let suggester = CanonicalFormSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = a.reduce_sum(1); // Reduce Sum
        graph.output("b", b);

        let suggestions = suggester.suggest(&graph);
        assert_eq!(suggestions.len(), 1);

        let new_graph = &suggestions[0].graph;
        let output = new_graph.outputs().get("b").unwrap();
        assert!(matches!(
            output.op,
            GraphOp::FusedElementwiseReduce { ref axes, .. } if axes == &[1]
        ));
    }

    #[test]
    fn test_canonicalize_chain() {
        let suggester = CanonicalFormSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let sum = a + b; // Elementwise
        let reduced = sum.reduce_sum(1); // Reduce
        graph.output("result", reduced);

        let suggestions = suggester.suggest(&graph);
        // 2つのノードが変換対象: Elementwise + Reduce
        assert_eq!(suggestions.len(), 2);
    }
}
