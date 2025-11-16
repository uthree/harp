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

        // ConstノードはContiguousなViewを持つべきで、
        // そのviewを変更すると物理メモリレイアウトとの不整合が発生する
        // Constノードにはマージを適用しない
        if matches!(input_node.op, GraphOp::Const(_)) {
            return None;
        }

        // Inputノードの場合は特別処理：View融合を許可
        // Inputノードは外部から提供されるデータで、Viewは論理的なアクセスパターンを定義
        if matches!(input_node.op, GraphOp::Input) {
            return self.merge_input_view_node(input_node, target_view);
        }

        // 入力ノードのviewをViewノードのviewで置き換えた新しいノードを作成
        // elementwise_strategiesの長さを新しいviewのndimに合わせて調整
        let new_ndim = target_view.ndim();
        let old_ndim = input_node.view.ndim();

        // 次元数が変わる場合、input_nodeのソースノードとの次元数不整合が発生する
        // 例えば、Add[2D] → View[3D] をマージすると、Addのソースは2Dのままなので、
        // lowering時に3Dインデックスを2Dストライドに適用しようとしてエラーになる
        if new_ndim != old_ndim {
            return None;
        }

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

    /// InputノードとViewノードをマージ
    ///
    /// Inputノードに対してViewを直接適用することで、中間のViewノードを削除します。
    fn merge_input_view_node(
        &self,
        input_node: &GraphNode,
        target_view: crate::graph::shape::View,
    ) -> Option<GraphNode> {
        let new_ndim = target_view.ndim();

        // elementwise_strategiesを新しい次元数に合わせて調整
        let mut new_strategies = Vec::new();
        let old_strategies = &input_node.elementwise_strategies;

        for i in 0..new_ndim {
            if i < old_strategies.len() {
                new_strategies.push(old_strategies[i].clone());
            } else {
                new_strategies.push(ElementwiseStrategy::Sequential {
                    simd_width: 1,
                    unroll_factor: 1,
                });
            }
        }

        // Inputノードのviewを新しいviewで置き換えた新しいノードを作成
        let new_input = GraphNode::with_elementwise_strategies(
            input_node.dtype.clone(),
            GraphOp::Input,
            vec![], // Inputノードはsrcを持たない
            target_view,
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

            // まずnode_mapを確認（Inputノードも置き換え対象になりうる）
            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            // Inputノードで置き換え対象でない場合はそのまま返す
            if matches!(node.op, GraphOp::Input) {
                return node.clone();
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

        // 入力ノードを保持（置き換え対象の場合は新しいノードを登録）
        for (name, weak_input) in graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                // 入力ノードが置き換え対象の場合は新しいノードを登録
                if let Some(replaced) = node_map.get(&input_node.as_ptr()) {
                    new_graph.register_input(name.clone(), replaced.clone());
                } else {
                    new_graph.register_input(name.clone(), input_node);
                }
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

    /// InputノードとViewノードを両方置き換えた新しいグラフを作成
    ///
    /// 元のInputノードを新しいノードで置き換え、Viewノードも同じ新しいノードで置き換える
    fn replace_input_and_view(
        &self,
        graph: &Graph,
        old_input: &GraphNode,
        old_view: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
        // InputノードとViewノード両方を新しいノードにマップ
        node_map.insert(old_input.as_ptr(), new_node.clone());
        node_map.insert(old_view.as_ptr(), new_node.clone());

        let mut visited = HashSet::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            visited: &mut HashSet<*const GraphNodeData>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // まずnode_mapを確認
            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            // Inputノードで置き換え対象でない場合はそのまま返す
            if matches!(node.op, GraphOp::Input) {
                return node.clone();
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

        // 入力ノードを登録（置き換え対象の場合は新しいノードを登録）
        for (name, weak_input) in graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                if let Some(replaced) = node_map.get(&input_node.as_ptr()) {
                    new_graph.register_input(name.clone(), replaced.clone());
                } else {
                    new_graph.register_input(name.clone(), input_node);
                }
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
                // Inputノードの場合は特別処理：元のInputノードを置き換える
                if node.src.len() == 1 && matches!(node.src[0].op, GraphOp::Input) {
                    // 元のInputノードを新しいノードで置き換え、Viewノードも削除
                    let new_graph =
                        self.replace_input_and_view(graph, &node.src[0], node, merged_input);
                    suggestions.push(new_graph);
                } else {
                    // 通常のケース：Viewノードを新しい入力ノードで置き換える
                    let new_graph = self.replace_node_in_graph(graph, node, merged_input);
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

    #[test]
    fn test_input_view_merge() {
        let suggester = ViewMergeSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // Inputノードに直接Viewを適用
        let permuted = a.view(a.view.clone().permute(vec![1, 0]));

        graph.output("result", permuted);

        let suggestions = suggester.suggest(&graph);

        // Input -> View の融合が1つ提案される
        assert_eq!(suggestions.len(), 1);

        // 提案されたグラフを確認
        let new_graph = &suggestions[0];
        let result = new_graph.outputs().get("result").unwrap();

        // 結果ノードがInputノードであることを確認（Viewノードが削除された）
        assert!(matches!(result.op, GraphOp::Input));

        // Inputノードのviewがpermuteされていることを確認
        assert_eq!(result.view.shape(), &[20.into(), 10.into()]);

        // 入力ノードも更新されていることを確認
        let input_a = new_graph.inputs().get("a").unwrap().upgrade().unwrap();
        let input_node = GraphNode::from_rc(input_a);
        assert_eq!(input_node.view.shape(), &[20.into(), 10.into()]);
    }

    #[test]
    fn test_input_view_merge_with_computation() {
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
            .with_shape(vec![20, 10])
            .build();

        // aをpermuteしてbと同じshapeにする
        let a_permuted = a.view(a.view.clone().permute(vec![1, 0]));

        // permute後のaとbを加算
        let sum = a_permuted + b;

        graph.output("result", sum);

        let suggestions = suggester.suggest(&graph);

        // Input -> View の融合が1つ提案される
        assert_eq!(suggestions.len(), 1);

        // 提案されたグラフを確認
        let new_graph = &suggestions[0];
        let result = new_graph.outputs().get("result").unwrap();

        // 結果ノードがElementwise(Add)ノードであることを確認
        assert!(matches!(result.op, GraphOp::Elementwise { .. }));

        // Addノードのソースが両方ともInputノードであることを確認
        // （Viewノードが削除されている）
        assert_eq!(result.src.len(), 2);
        assert!(matches!(result.src[0].op, GraphOp::Input));
        assert!(matches!(result.src[1].op, GraphOp::Input));

        // 最初のソース（元のaをpermute）のviewが変更されていることを確認
        assert_eq!(result.src[0].view.shape(), &[20.into(), 10.into()]);
    }

    #[test]
    fn test_input_view_unsqueeze() {
        let suggester = ViewMergeSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        // unsqueezeでshapeを変更: [10] -> [1, 10]
        let unsqueezed = a.view(a.view.clone().unsqueeze(0));

        graph.output("result", unsqueezed);

        let suggestions = suggester.suggest(&graph);

        // Input -> View (unsqueeze) の融合が1つ提案される
        assert_eq!(suggestions.len(), 1);

        let new_graph = &suggestions[0];
        let result = new_graph.outputs().get("result").unwrap();

        // 結果ノードがInputノードであることを確認
        assert!(matches!(result.op, GraphOp::Input));

        // shapeが[1, 10]に変更されていることを確認
        assert_eq!(result.view.shape(), &[1.into(), 10.into()]);
    }
}
