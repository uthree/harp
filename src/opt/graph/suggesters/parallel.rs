use crate::graph::{ElementwiseStrategy, Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// 並列化戦略を変更するSuggester
pub struct ParallelStrategyChanger {
    /// 試行する戦略の候補
    strategy_candidates: Vec<ElementwiseStrategy>,
}

impl Default for ParallelStrategyChanger {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelStrategyChanger {
    /// 新しいParallelStrategyChangerを作成
    pub fn new() -> Self {
        Self {
            strategy_candidates: vec![
                ElementwiseStrategy::sequential(),
                ElementwiseStrategy::sequential_simd(4),
                ElementwiseStrategy::sequential_unroll(4),
                ElementwiseStrategy::thread(),
                ElementwiseStrategy::thread_simd(4),
                ElementwiseStrategy::thread_group(),
            ],
        }
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

            // 先に依存ノードを訪問
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

    /// ノードの戦略を変更した新しいノードを作成
    fn change_node_strategy(
        &self,
        node: &GraphNode,
        axis: usize,
        new_strategy: ElementwiseStrategy,
    ) -> GraphNode {
        let mut new_strategies = node.elementwise_strategies.clone();
        if axis < new_strategies.len() {
            new_strategies[axis] = new_strategy;
        }

        GraphNode::with_elementwise_strategies(
            node.dtype.clone(),
            node.op.clone(),
            node.src.clone(),
            node.view.clone(),
            new_strategies,
        )
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
            if matches!(node.op, GraphOp::Input) {
                return node.clone();
            }

            // すでに置き換えマップに存在する場合はそれを返す
            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            // すでに訪問済みならそのまま返す
            if visited.contains(&ptr) {
                return node.clone();
            }
            visited.insert(ptr);

            // 依存ノードを再構築
            let new_src: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, visited))
                .collect();

            // srcが変わっていなければそのまま返す
            let src_changed = new_src
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            if !src_changed {
                return node.clone();
            }

            // 新しいノードを作成
            GraphNode::with_elementwise_strategies(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
                node.elementwise_strategies.clone(),
            )
        }

        // 新しいグラフを構築
        let mut new_graph = Graph::new();

        // 入力ノードを保持（再構築しない - Inputノードは変更されないため）
        for (name, weak_input) in graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                new_graph.register_input(name.clone(), input_node);
            }
        }

        // 出力ノードを名前順でソートして再構築（順序を固定）
        let mut outputs: Vec<_> = graph.outputs().iter().collect();
        outputs.sort_by_key(|(name, _)| name.as_str());

        for (name, output_node) in outputs {
            let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
            new_graph.output(name, rebuilt);
        }

        new_graph
    }
}

impl GraphSuggester for ParallelStrategyChanger {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各ノードの各軸について、異なる戦略を試す
        for node in &nodes {
            // elementwise_strategiesが空のノードはスキップ
            if node.elementwise_strategies.is_empty() {
                continue;
            }

            for axis in 0..node.elementwise_strategies.len() {
                let current_strategy = &node.elementwise_strategies[axis];

                for new_strategy in &self.strategy_candidates {
                    // 現在の戦略と同じならスキップ
                    if current_strategy == new_strategy {
                        continue;
                    }

                    // 新しい戦略を適用したノードを作成
                    let new_node = self.change_node_strategy(node, axis, new_strategy.clone());

                    // グラフ全体を再構築
                    let new_graph = self.replace_node_in_graph(graph, node, new_node);
                    suggestions.push(new_graph);
                }
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
    fn test_parallel_strategy_changer() {
        let changer = ParallelStrategyChanger::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = changer.suggest(&graph);

        // 複数の戦略変更候補が生成されるはず
        assert!(!suggestions.is_empty());
    }
}
