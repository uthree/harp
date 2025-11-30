use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
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

        // Constノードの場合: ViewをConstに吸収させる
        // Constノードは単一の値を持つスカラーで、Viewは論理的なアクセスパターンを定義
        // View(Const)パターンをConst[View適用済み]に変換することで、
        // 中間のViewノードを削除できる
        if matches!(
            input_node.op,
            GraphOp::Const(_) | GraphOp::ComplexConst { .. }
        ) {
            return self.merge_const_view_node(input_node, target_view);
        }

        // Inputノードの場合は特別処理：View融合を許可
        // Inputノードは外部から提供されるデータで、Viewは論理的なアクセスパターンを定義
        if matches!(input_node.op, GraphOp::Buffer { .. }) {
            return self.merge_input_view_node(input_node, target_view);
        }

        // Customノードの場合は特別処理：Viewを出力に取り込む
        // Custom → View → Consumer のパターンを Custom[View適用済み] → Consumer に変換
        // これにより、後続のKernelMergeSuggesterがCustom同士を直接マージできる
        if matches!(input_node.op, GraphOp::Custom { .. }) {
            return self.merge_custom_view_node(input_node, target_view);
        }

        // 入力ノードのviewをViewノードのviewで置き換えた新しいノードを作成
        let new_ndim = target_view.ndim();
        let old_ndim = input_node.view.ndim();

        // 次元数が変わる場合、input_nodeのソースノードとの次元数不整合が発生する
        // 例えば、Add[2D] → View[3D] をマージすると、Addのソースは2Dのままなので、
        // lowering時に3Dインデックスを2Dストライドに適用しようとしてエラーになる
        if new_ndim != old_ndim {
            return None;
        }

        // スカラー（ndim=0）の場合はマージをスキップ
        if new_ndim == 0 {
            return None;
        }

        let new_input = GraphNode::new(
            input_node.dtype.clone(),
            input_node.op.clone(),
            input_node.src.clone(),
            target_view, // Viewノードのviewを使用
        );

        Some(new_input)
    }

    /// CustomノードとViewノードをマージ
    ///
    /// Custom → View のパターンで、CustomノードにView変換を取り込む。
    /// これにより、Custom[View適用済み] → Consumer の形になり、
    /// KernelMergeSuggesterがCustom同士を直接マージできるようになる。
    fn merge_custom_view_node(
        &self,
        custom_node: &GraphNode,
        target_view: crate::graph::shape::View,
    ) -> Option<GraphNode> {
        // Customノードの出力Viewを新しいViewで置き換えた新しいノードを作成
        // ASTは変更せず、GraphNodeのviewフィールドのみを更新
        // lowering時にこのviewが使用されるため、メモリアクセスパターンが反映される
        let new_custom = GraphNode::new(
            custom_node.dtype.clone(),
            custom_node.op.clone(),
            custom_node.src.clone(),
            target_view,
        );

        Some(new_custom)
    }

    /// BufferノードとViewノードをマージ
    ///
    /// Bufferノードに対してViewを直接適用することで、中間のViewノードを削除します。
    fn merge_input_view_node(
        &self,
        input_node: &GraphNode,
        target_view: crate::graph::shape::View,
    ) -> Option<GraphNode> {
        // Bufferノードのviewを新しいviewで置き換えた新しいノードを作成
        let new_input = GraphNode::new(
            input_node.dtype.clone(),
            input_node.op.clone(), // 名前を保持
            vec![],                // Bufferノードはsrcを持たない
            target_view,
        );

        Some(new_input)
    }

    /// ConstノードとViewノードをマージ
    ///
    /// Constノードに対してViewを直接適用することで、中間のViewノードを削除します。
    /// Constノードは単一の値を持つスカラーなので、Viewの適用は論理的なブロードキャストを表します。
    ///
    /// 例：
    /// ```text
    /// Const(1.0, view=[]) -> View(view=[10,10], strides=[0,0]) -> Elementwise
    /// ```
    /// を以下のように最適化：
    /// ```text
    /// Const(1.0, view=[10,10], strides=[0,0]) -> Elementwise
    /// ```
    fn merge_const_view_node(
        &self,
        const_node: &GraphNode,
        target_view: crate::graph::shape::View,
    ) -> Option<GraphNode> {
        // ConstノードのviewをViewノードのviewで置き換えた新しいノードを作成
        let new_const = GraphNode::new(
            const_node.dtype.clone(),
            const_node.op.clone(), // Const値を保持
            vec![],                // Constノードはsrcを持たない
            target_view,
        );

        Some(new_const)
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

            // Buffer/Const/ComplexConstノードで置き換え対象でない場合はそのまま返す
            // これらのノードはsrcを持たないため、再帰的な処理は不要
            if matches!(
                node.op,
                GraphOp::Buffer { .. } | GraphOp::Const(_) | GraphOp::ComplexConst { .. }
            ) {
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

            GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
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
            if matches!(node.op, GraphOp::Buffer { .. }) {
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

            GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
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

impl Default for ViewMergeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ViewMergeSuggester {
    fn name(&self) -> &'static str {
        "ViewMerge"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);
        let ref_counts = self.count_node_references(graph);

        // 各Viewノードについて、マージを試みる
        for node in &nodes {
            // Viewノードのみを対象
            if !matches!(node.op, GraphOp::View(_)) {
                continue;
            }

            // 融合されるノード（Viewノードの入力ノード）の被参照数をチェック
            // 被参照数が複数ある場合は融合をスキップ（ダングリングポインタを防ぐ）
            if node.src.len() == 1 {
                let input_node = &node.src[0];
                let input_ptr = input_node.as_ptr();
                let ref_count = ref_counts.get(&input_ptr).copied().unwrap_or(0);

                // 被参照数が1より大きい場合は融合しない
                if ref_count > 1 {
                    continue;
                }
            }

            // Viewノードをマージして、入力ノードのViewを更新
            if let Some(merged_input) = self.merge_view_node(node) {
                // Inputノードの場合は特別処理：元のInputノードを置き換える
                if node.src.len() == 1 && matches!(node.src[0].op, GraphOp::Buffer { .. }) {
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
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let b = graph.input("b", DType::F32, vec![10, 20]);

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
        let outputs = new_graph.outputs();
        let result = outputs.get("result").unwrap();

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
        let a = graph.input("a", DType::F32, vec![2, 3, 4]);

        let b = graph.input("b", DType::F32, vec![2, 3, 4]);

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
        let a = graph.input("a", DType::F32, vec![10, 20]);

        // Inputノードに直接Viewを適用
        let permuted = a.view(a.view.clone().permute(vec![1, 0]));

        graph.output("result", permuted);

        let suggestions = suggester.suggest(&graph);

        // Input -> View の融合が1つ提案される
        assert_eq!(suggestions.len(), 1);

        // 提案されたグラフを確認
        let new_graph = &suggestions[0];
        let outputs = new_graph.outputs();
        let result = outputs.get("result").unwrap();

        // 結果ノードがInputノードであることを確認（Viewノードが削除された）
        assert!(matches!(result.op, GraphOp::Buffer { .. }));

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
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let b = graph.input("b", DType::F32, vec![20, 10]);

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
        let outputs = new_graph.outputs();
        let result = outputs.get("result").unwrap();

        // 結果ノードがElementwise(Add)ノードであることを確認
        assert!(matches!(result.op, GraphOp::Elementwise { .. }));

        // Addノードのソースが両方ともInputノードであることを確認
        // （Viewノードが削除されている）
        assert_eq!(result.src.len(), 2);
        assert!(matches!(result.src[0].op, GraphOp::Buffer { .. }));
        assert!(matches!(result.src[1].op, GraphOp::Buffer { .. }));

        // 最初のソース（元のaをpermute）のviewが変更されていることを確認
        assert_eq!(result.src[0].view.shape(), &[20.into(), 10.into()]);
    }

    #[test]
    fn test_input_view_unsqueeze() {
        let suggester = ViewMergeSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);

        // unsqueezeでshapeを変更: [10] -> [1, 10]
        let unsqueezed = a.view(a.view.clone().unsqueeze(0));

        graph.output("result", unsqueezed);

        let suggestions = suggester.suggest(&graph);

        // Input -> View (unsqueeze) の融合が1つ提案される
        assert_eq!(suggestions.len(), 1);

        let new_graph = &suggestions[0];
        let outputs = new_graph.outputs();
        let result = outputs.get("result").unwrap();

        // 結果ノードがInputノードであることを確認
        assert!(matches!(result.op, GraphOp::Buffer { .. }));

        // shapeが[1, 10]に変更されていることを確認
        assert_eq!(result.view.shape(), &[1.into(), 10.into()]);
    }

    // Note: test_no_merge_multiple_references と test_no_merge_when_input_has_multiple_users は
    // 複数出力が現在サポートされていないため削除されました。
    // 詳細は spec/TODO.md を参照してください。

    #[test]
    fn test_custom_view_merge() {
        use crate::opt::graph::GraphSuggester as _;
        use crate::opt::graph::suggesters::LoweringSuggester;

        let view_suggester = ViewMergeSuggester::new();
        let lowering_suggester = LoweringSuggester::new();

        // グラフを作成: a + b
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let sum = a + b;
        graph.output("result", sum);

        // LoweringSuggesterでCustomノードに変換
        let lowered_graphs = lowering_suggester.suggest(&graph);
        assert!(
            !lowered_graphs.is_empty(),
            "LoweringSuggester should produce suggestions"
        );
        let lowered_graph = &lowered_graphs[0];

        // Customノードの出力にViewを適用
        let lowered_outputs = lowered_graph.outputs();
        let result_node = lowered_outputs.get("result").unwrap();
        assert!(
            matches!(result_node.op, GraphOp::Custom { .. }),
            "Should be Custom node"
        );

        // Viewを適用した新しいグラフを作成
        let permuted = result_node.view(result_node.view.clone().permute(vec![1, 0]));
        let mut graph_with_view = Graph::new();
        for (name, weak_input) in lowered_graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                graph_with_view.register_input(name.clone(), input_node);
            }
        }
        graph_with_view.output("result", permuted);

        // ViewMergeSuggesterでCustom→ViewをCustom[View適用済み]にマージ
        let merged_graphs = view_suggester.suggest(&graph_with_view);
        assert_eq!(merged_graphs.len(), 1, "Should produce 1 merged graph");

        // マージ後のグラフを確認
        let merged_outputs = merged_graphs[0].outputs();
        let merged_result = merged_outputs.get("result").unwrap();

        // 結果がCustomノードであることを確認（Viewノードではない）
        assert!(
            matches!(merged_result.op, GraphOp::Custom { .. }),
            "Result should be Custom node after merge, got {:?}",
            merged_result.op
        );

        // Viewが適用されていることを確認
        assert_eq!(
            merged_result.view.shape(),
            &[20.into(), 10.into()],
            "Custom node should have permuted view"
        );
    }
}
