use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp, View};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// 非contiguousなViewを持つノードの前にContiguousノードを挿入するSuggester
///
/// View操作（permute、flipなど）によって非連続なメモリレイアウトになったテンソルを、
/// Contiguousノードを挿入してメモリ連続にすることで、
/// メモリアクセスパターンを最適化する可能性を提案します。
///
/// # 挿入条件
/// - 入力ノードが非contiguousなViewを持つ
/// - ノード自体がViewノードまたはContiguousノードでない
///
/// # 最適化の考え方
/// 非連続なメモリアクセスはキャッシュミスを増やし、パフォーマンスを低下させることがあります。
/// 一方、Contiguousノードはメモリコピーを伴うため、コストも発生します。
/// どちらが有利かはケースバイケースであり、ビームサーチで探索する価値があります。
pub struct ContiguousInsertionSuggester;

impl ContiguousInsertionSuggester {
    /// 新しいContiguousInsertionSuggesterを作成
    pub fn new() -> Self {
        ContiguousInsertionSuggester
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

    /// 指定されたノードの特定入力にContiguousノードを挿入
    fn insert_contiguous_before_input(&self, node: &GraphNode, input_index: usize) -> GraphNode {
        let input = &node.src[input_index];

        // Contiguousノードを作成
        let contiguous_view = View::contiguous(input.view.shape().to_vec());
        let contiguous_node = GraphNode::new(
            input.dtype.clone(),
            GraphOp::Contiguous {
                elementwise_strategies: None,
            },
            vec![input.clone()],
            contiguous_view,
        );

        // ノードの入力を置き換え
        let mut new_src = node.src.clone();
        new_src[input_index] = contiguous_node;

        GraphNode::with_elementwise_strategies(
            node.dtype.clone(),
            node.op.clone(),
            new_src,
            node.view.clone(),
            node.elementwise_strategies.clone(),
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

impl Default for ContiguousInsertionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ContiguousInsertionSuggester {
    fn name(&self) -> &'static str {
        "ContiguousInsertion"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各ノードについて、非contiguousな入力を持つかチェック
        for node in &nodes {
            // InputノードとConstノードはスキップ（入力がない）
            if matches!(node.op, GraphOp::Input | GraphOp::Const(_)) {
                continue;
            }

            // ViewノードとContiguousノードはスキップ
            // （Viewノードの前にContiguousを入れても意味がない、
            //  Contiguousノードの前にContiguousを入れても冗長）
            if matches!(node.op, GraphOp::View(_) | GraphOp::Contiguous { .. }) {
                continue;
            }

            // 各入力について、非contiguousなViewを持つかチェック
            for (input_idx, input) in node.src.iter().enumerate() {
                if !input.view.is_contiguous() {
                    // 非contiguousな入力の前にContiguousノードを挿入
                    let new_node = self.insert_contiguous_before_input(node, input_idx);
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
    fn test_contiguous_insertion_after_permute() {
        let suggester = ContiguousInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        // 転置してViewを非contiguousにする
        let transposed_view = a.view.clone().permute(vec![1, 0]);
        let a_transposed = a.view(transposed_view);

        // 非contiguousなViewを持つテンソルに演算を適用
        let b = graph.input("b", DType::F32, vec![20, 10]);

        let c = a_transposed + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 2つの候補が生成されるはず：
        // 1. a_transposedの前にContiguousを挿入
        // 2. bの前にContiguousを挿入（bはcontiguousだが、候補として生成される可能性）
        //
        // 実際にはa_transposedのみが非contiguousなので、1つの候補のみ
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_contiguous_insertion_after_flip() {
        let suggester = ContiguousInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        // 反転してViewを非contiguousにする
        let flipped_view = a.view.clone().flip(0);
        let a_flipped = a.view(flipped_view);

        let c = -a_flipped;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // a_flippedの前にContiguousを挿入する候補が1つ生成される
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_contiguous_insertion_no_suggestion_for_contiguous() {
        let suggester = ContiguousInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let b = graph.input("b", DType::F32, vec![10, 20]);

        // contiguousなViewのみなので、Contiguous挿入の必要なし
        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_contiguous_insertion_skip_view_node() {
        let suggester = ContiguousInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        // 転置（Viewノード）
        let transposed_view = a.view.clone().permute(vec![1, 0]);
        let a_transposed = a.view(transposed_view);

        graph.output("c", a_transposed);

        let suggestions = suggester.suggest(&graph);

        // Viewノード自体にはContiguousを挿入しない
        // （Viewノードの前にContiguousを入れても意味がない）
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_contiguous_insertion_multiple_noncontiguous_inputs() {
        let suggester = ContiguousInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let b = graph.input("b", DType::F32, vec![10, 20]);

        // 両方のテンソルを転置して非contiguousに
        let a_transposed = a.view(a.view.clone().permute(vec![1, 0]));
        let b_transposed = b.view(b.view.clone().permute(vec![1, 0]));

        let c = a_transposed + b_transposed;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 2つの候補が生成される：
        // 1. a_transposedの前にContiguousを挿入
        // 2. b_transposedの前にContiguousを挿入
        assert_eq!(suggestions.len(), 2);
    }
}
