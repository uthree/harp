use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// View変更ノード（転置など）を挿入するSuggester
pub struct ViewInsertionSuggester {
    /// 転置を試みるかどうか
    try_transpose: bool,
}

impl ViewInsertionSuggester {
    /// 新しいViewInsertionSuggesterを作成
    pub fn new() -> Self {
        Self {
            try_transpose: true,
        }
    }

    /// 転置を試みるかどうかを設定
    pub fn with_transpose(mut self, enable: bool) -> Self {
        self.try_transpose = enable;
        self
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

    /// ノードの入力にView変更を挿入した新しいノードを作成
    fn insert_view_before_input(
        &self,
        node: &GraphNode,
        input_idx: usize,
        permutation: Vec<usize>,
    ) -> GraphNode {
        if input_idx >= node.src.len() {
            return node.clone();
        }

        let input = &node.src[input_idx];

        // permuteを適用した新しいノードを作成
        let permuted = input.view.clone().permute(permutation.clone());
        let view_node = GraphNode::new(
            input.dtype.clone(),
            GraphOp::View(permuted.clone()),
            vec![input.clone()],
            permuted,
        );

        // Contiguousノードを挿入してメモリレイアウトを実体化
        let contiguous_view = view_node.view.clone();
        let contiguous_dtype = view_node.dtype.clone();
        let contiguous_node = GraphNode::new(
            contiguous_dtype,
            GraphOp::Contiguous {
                elementwise_strategies: None,
            },
            vec![view_node],
            contiguous_view,
        );

        // 元のノードのsrcを置き換え
        let mut new_src = node.src.clone();
        new_src[input_idx] = contiguous_node;

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
                let rebuilt = rebuild_node(&input_node, &node_map, &mut visited);
                new_graph.register_input(name.clone(), rebuilt);
            }
        }

        // 出力ノードを再構築
        for (name, output_node) in graph.outputs() {
            let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
            new_graph.output(name, rebuilt);
        }

        new_graph
    }

    /// 2次元転置のpermutationを生成
    fn transpose_2d_permutation(&self, ndim: usize) -> Option<Vec<usize>> {
        if ndim == 2 {
            Some(vec![1, 0])
        } else if ndim > 2 {
            // 最後の2次元を転置
            let mut perm: Vec<usize> = (0..ndim).collect();
            perm.swap(ndim - 2, ndim - 1);
            Some(perm)
        } else {
            None
        }
    }
}

impl Default for ViewInsertionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ViewInsertionSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        if !self.try_transpose {
            return vec![];
        }

        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各ノードの各入力について、転置を試みる
        for node in &nodes {
            // 入力が複数あるノードのみ対象（例: Add, Mul）
            if node.src.len() < 2 {
                continue;
            }

            for input_idx in 0..node.src.len() {
                let input = &node.src[input_idx];
                let ndim = input.view.ndim();

                // 2次元以上のテンソルのみ転置可能
                if let Some(permutation) = self.transpose_2d_permutation(ndim) {
                    let new_node = self.insert_view_before_input(node, input_idx, permutation);
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
    fn test_view_insertion_suggester() {
        let suggester = ViewInsertionSuggester::new();

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

        let suggestions = suggester.suggest(&graph);

        // 2つの入力それぞれに転置を試みるので、2つの候補が生成されるはず
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_view_insertion_disabled() {
        let suggester = ViewInsertionSuggester::new().with_transpose(false);

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

        let suggestions = suggester.suggest(&graph);

        // 転置が無効なので、候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }
}
