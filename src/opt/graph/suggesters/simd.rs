use crate::graph::{ElementwiseStrategy, Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// SIMD化を提案するSuggester
pub struct SimdSuggester {
    /// 試行するSIMD幅の候補
    pub simd_widths: Vec<usize>,
}

impl Default for SimdSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdSuggester {
    /// 新しいSimdSuggesterを作成
    pub fn new() -> Self {
        Self {
            simd_widths: vec![2, 3, 4, 8],
        }
    }

    /// 指定したSIMD幅の候補を使用するSimdSuggesterを作成
    pub fn with_simd_widths(simd_widths: Vec<usize>) -> Self {
        Self { simd_widths }
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

    /// ノードの最内軸（最後の軸）にSIMD化を適用した新しいノードを作成
    fn apply_simd_to_node(&self, node: &GraphNode, simd_width: usize) -> Option<GraphNode> {
        // Inputノードはスキップ
        if matches!(node.op, GraphOp::Input) {
            return None;
        }

        // elementwise_strategiesが空のノードはスキップ
        if node.elementwise_strategies.is_empty() {
            return None;
        }

        let innermost_axis = node.elementwise_strategies.len() - 1;
        let current_strategy = &node.elementwise_strategies[innermost_axis];

        // 現在の戦略を判定
        match current_strategy {
            ElementwiseStrategy::Sequential {
                simd_width: current_simd,
                unroll_factor,
            } => {
                // すでにSIMD化されている場合（simd_width > 1）はスキップ
                if *current_simd > 1 {
                    return None;
                }

                // SIMD化されていない場合のみSIMD化を提案
                let mut new_strategies = node.elementwise_strategies.clone();
                new_strategies[innermost_axis] = ElementwiseStrategy::Sequential {
                    simd_width,
                    unroll_factor: *unroll_factor,
                };

                Some(GraphNode::with_elementwise_strategies(
                    node.dtype.clone(),
                    node.op.clone(),
                    node.src.clone(),
                    node.view.clone(),
                    new_strategies,
                ))
            }
            ElementwiseStrategy::Thread {
                simd_width: current_simd,
                unroll_factor,
            } => {
                // すでにSIMD化されている場合（simd_width > 1）はスキップ
                if *current_simd > 1 {
                    return None;
                }

                // Thread並列化にSIMD化を追加
                let mut new_strategies = node.elementwise_strategies.clone();
                new_strategies[innermost_axis] = ElementwiseStrategy::Thread {
                    simd_width,
                    unroll_factor: *unroll_factor,
                };

                Some(GraphNode::with_elementwise_strategies(
                    node.dtype.clone(),
                    node.op.clone(),
                    node.src.clone(),
                    node.view.clone(),
                    new_strategies,
                ))
            }
            ElementwiseStrategy::ThreadGroup {
                simd_width: current_simd,
                unroll_factor,
            } => {
                // すでにSIMD化されている場合（simd_width > 1）はスキップ
                if *current_simd > 1 {
                    return None;
                }

                // ThreadGroup並列化にSIMD化を追加
                let mut new_strategies = node.elementwise_strategies.clone();
                new_strategies[innermost_axis] = ElementwiseStrategy::ThreadGroup {
                    simd_width,
                    unroll_factor: *unroll_factor,
                };

                Some(GraphNode::with_elementwise_strategies(
                    node.dtype.clone(),
                    node.op.clone(),
                    node.src.clone(),
                    node.view.clone(),
                    new_strategies,
                ))
            }
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

impl GraphSuggester for SimdSuggester {
    fn name(&self) -> &'static str {
        "Simd"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各ノードについて、異なるSIMD幅を試す
        for node in &nodes {
            for &simd_width in &self.simd_widths {
                if let Some(new_node) = self.apply_simd_to_node(node, simd_width) {
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
    fn test_simd_suggester() {
        let suggester = SimdSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let b = graph.input("b", DType::F32, vec![10, 20]);

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 4つのSIMD幅候補(2, 3, 4, 8)が生成されるはず
        assert_eq!(suggestions.len(), 4);
    }

    #[test]
    fn test_simd_suggester_with_custom_widths() {
        let suggester = SimdSuggester::with_simd_widths(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![16]);

        let b = graph.input("b", DType::F32, vec![16]);

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 1つのSIMD幅候補(4)のみが生成されるはず
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_simd_suggester_skips_already_simd() {
        use crate::graph::{
            Expr, GraphNode, View,
            ops::{ElementwiseOp, GraphOp},
        };

        let suggester = SimdSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![16]);

        let b = graph.input("b", DType::F32, vec![16]);

        // すでにSIMD化されたノードを作成（simd_width = 4）
        let view = View::contiguous(vec![Expr::from(16)]);
        let simd_node = GraphNode::with_elementwise_strategies(
            DType::F32,
            GraphOp::Elementwise {
                op: ElementwiseOp::Add,
                elementwise_strategies: None,
            },
            vec![a, b],
            view,
            vec![ElementwiseStrategy::Sequential {
                simd_width: 4,
                unroll_factor: 1,
            }],
        );

        graph.output("c", simd_node);

        let suggestions = suggester.suggest(&graph);

        // すでにSIMD化されているのでスキップされる
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_simd_suggester_integration_with_beam_search() {
        use crate::opt::graph::{BeamSearchGraphOptimizer, SimpleCostEstimator};

        let suggester = SimdSuggester::new();
        let estimator = SimpleCostEstimator::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![16]);

        let b = graph.input("b", DType::F32, vec![16]);

        let c = a + b;
        graph.output("c", c);

        // ビームサーチ最適化器でSIMD化を試す
        let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(5)
            .with_max_steps(10)
            .with_progress(false);

        let (optimized_graph, _history) = optimizer.optimize_with_history(graph);

        // 最適化されたグラフを確認
        let output = optimized_graph.outputs().get("c").unwrap();

        // elementwise_strategiesが設定されていることを確認
        assert!(!output.elementwise_strategies.is_empty());
    }
}
