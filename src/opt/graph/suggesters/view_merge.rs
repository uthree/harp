use crate::graph::{ElementwiseStrategy, Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// Viewノードをマージして、ノード数を削減するSuggester
///
/// Viewノードの入力ノードのViewパラメータを直接書き換えることで、
/// 中間のViewノードを削除してノード数を削減します。
///
/// 例：
/// ```text
/// Input(view=V1) -> View(view=V2) -> Op
/// ```
/// を以下のように最適化：
/// ```text
/// Input(view=V2) -> Op
/// ```
pub struct ViewMergeSuggester;

impl ViewMergeSuggester {
    /// 新しいViewMergeSuggesterを作成
    pub fn new() -> Self {
        ViewMergeSuggester {}
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

    /// Viewノードをマージして、入力ノードのViewを更新
    ///
    /// View(view=V2)ノードの入力input_nodeのviewをV2で置き換えることで、
    /// Viewノードを削除します。
    fn merge_view_node(&self, view_node: &GraphNode) -> Option<GraphNode> {
        // ViewノードであることをGraphOp::View(_)で確認
        let target_view = match &view_node.op {
            GraphOp::View(v) => v.clone(),
            _ => return None,
        };

        // Viewノードは入力が1つのはず
        if view_node.src.len() != 1 {
            return None;
        }

        let input_node = &view_node.src[0];

        // 入力ノードのviewをViewノードのviewで置き換えた新しいノードを作成
        // elementwise_strategiesの長さを新しいviewのndimに合わせて調整
        let new_ndim = target_view.ndim();

        // スカラー（ndim=0）の場合や、elementwise_strategiesが空の場合はマージをスキップ
        // （これらは特殊なケースで、elementwise演算の対象外）
        if new_ndim == 0 || input_node.elementwise_strategies.is_empty() {
            return None;
        }

        let old_strategies = &input_node.elementwise_strategies;
        let mut new_strategies = Vec::new();

        for i in 0..new_ndim {
            if i < old_strategies.len() {
                // 既存のstrategyを使用
                new_strategies.push(old_strategies[i].clone());
            } else {
                // デフォルトのstrategyを追加（Sequential, simd_width=1, unroll_factor=1）
                new_strategies.push(ElementwiseStrategy::Sequential {
                    simd_width: 1,
                    unroll_factor: 1,
                });
            }
        }

        let new_input = GraphNode::with_elementwise_strategies(
            input_node.dtype.clone(),
            input_node.op.clone(),
            input_node.src.clone(),
            target_view, // Viewノードのviewを使用
            new_strategies,
        );

        Some(new_input)
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

impl Default for ViewMergeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ViewMergeSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各Viewノードについて、マージを試みる
        for node in &nodes {
            // Viewノードのみを対象
            if !matches!(node.op, GraphOp::View(_)) {
                continue;
            }

            // Viewノードをマージして、入力ノードのViewを更新
            if let Some(merged_input) = self.merge_view_node(node) {
                // Viewノードを新しい入力ノードで置き換える
                // （実際にはViewノードを削除して、その入力を直接使う）
                let new_graph = self.replace_node_in_graph(graph, node, merged_input);
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
    fn test_view_merge_basic() {
        let suggester = ViewMergeSuggester::new();

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

        // a + b を計算
        let sum = a.clone() + b.clone();

        // sumに対してpermuteを適用（Viewノードが作られる）
        let permuted = sum.view(sum.view.clone().permute(vec![1, 0]));

        graph.output("result", permuted);

        let suggestions = suggester.suggest(&graph);

        // Viewノードが1つあるので、1つの提案が生成される
        // （Addノードのviewが直接書き換えられる）
        assert_eq!(suggestions.len(), 1);

        // 提案されたグラフを確認
        let new_graph = &suggestions[0];
        let result = new_graph.outputs().get("result").unwrap();

        // 結果ノードがViewノードではなく、Elementwise(Add)ノードであることを確認
        // （ViewノードがマージされてElementwiseノードに統合されている）
        assert!(matches!(result.op, GraphOp::Elementwise { .. }));

        // Addノードのviewがpermuteされていることを確認
        assert_eq!(result.view.shape(), &[20.into(), 10.into()]);
    }

    #[test]
    fn test_view_merge_skip_input() {
        let suggester = ViewMergeSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // Inputノードに対してpermuteを適用
        let permuted = a.view(a.view.clone().permute(vec![1, 0]));

        graph.output("result", permuted);

        let suggestions = suggester.suggest(&graph);

        // InputノードのViewは書き換えられないので、提案は0個
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_view_merge_chain() {
        let suggester = ViewMergeSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3, 4])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3, 4])
            .build();

        let sum = a + b;

        // 複数のView変換を連鎖
        let view1 = sum.view(sum.view.clone().permute(vec![1, 0, 2])); // [3, 2, 4]
        let view2 = view1.view(view1.view.clone().permute(vec![2, 1, 0])); // [4, 2, 3]

        graph.output("result", view2);

        let suggestions = suggester.suggest(&graph);

        // 2つのViewノードがあるので、2つの提案が生成される
        // 1. view1をマージ
        // 2. view2をマージ
        assert_eq!(suggestions.len(), 2);
    }
}
