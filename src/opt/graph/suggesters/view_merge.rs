//! View Merge Suggester
//!
//! Viewノードを前のノードに吸収して、ノード数を削減するSuggester。
//!
//! # 処理内容
//! Viewノードの入力ノードのviewフィールドをViewノードのview値で置き換えることで、
//! 中間のViewノードを削除します。
//!
//! # 例
//! ```text
//! SomeOp(view=V1) -> View(view=V2) -> Consumer
//! ```
//! を以下のように最適化：
//! ```text
//! SomeOp(view=V2) -> Consumer
//! ```
//!
//! # 次元数の変更
//! - Buffer/Const/Custom: 次元数の変更を許可（reshape, unsqueezeなど）
//! - その他（Elementwise等）: 次元数の変更を許可しない（srcとの整合性の問題）

use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// Viewノードをマージして、ノード数を削減するSuggester
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

    /// 入力ノードが次元数変更を許可するタイプかどうかを判定
    ///
    /// Buffer/Const/Custom は次元数の変更を許可する（独立したノードのため）
    /// Elementwise等は次元数を変更するとsrcとの整合性が崩れるため許可しない
    fn allows_ndim_change(op: &GraphOp) -> bool {
        matches!(
            op,
            GraphOp::Buffer { .. }
                | GraphOp::Const(_)
                | GraphOp::ComplexConst { .. }
                | GraphOp::Kernel { .. }
        )
    }

    /// Viewノードを入力ノードにマージ
    ///
    /// 入力ノードのviewをViewノードのviewで置き換えた新しいノードを返す
    fn merge_view_node(&self, view_node: &GraphNode) -> Option<GraphNode> {
        // ViewノードのView値を取得
        let target_view = match &view_node.op {
            GraphOp::View(v) => v.clone(),
            _ => return None,
        };

        // Viewノードは入力が1つのはず
        if view_node.src.len() != 1 {
            return None;
        }

        let input_node = &view_node.src[0];

        // 次元数変更のチェック
        if !Self::allows_ndim_change(&input_node.op) {
            let new_ndim = target_view.ndim();
            let old_ndim = input_node.view.ndim();

            // 次元数が変わる場合はマージしない
            if new_ndim != old_ndim {
                return None;
            }

            // スカラー（ndim=0）の場合もスキップ
            if new_ndim == 0 {
                return None;
            }
        }

        // 入力ノードのviewを置き換えた新しいノードを作成
        Some(GraphNode::new(
            input_node.dtype.clone(),
            input_node.op.clone(),
            input_node.src.clone(),
            target_view,
        ))
    }

    /// グラフ内のViewノードを新しいノードで置き換えた新しいグラフを作成
    fn replace_view_in_graph(
        &self,
        graph: &Graph,
        view_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        // Viewノードを新しいノードにマップ
        node_map.insert(view_node.as_ptr(), new_node.clone());

        // 入力ノード（Buffer/Const等）も同じ新しいノードにマップ
        // これにより、元の入力ノードへの参照も新しいノードに置き換わる
        if view_node.src.len() == 1 {
            let input_node = &view_node.src[0];
            // srcを持たないノード（Buffer/Const）のみ置き換え
            // srcを持つノード（Elementwise等）は他の参照があるため置き換えない
            if input_node.src.is_empty() {
                node_map.insert(input_node.as_ptr(), new_node);
            }
        }

        self.rebuild_graph_with_map(graph, &node_map)
    }

    /// node_mapを使ってグラフを再構築
    fn rebuild_graph_with_map(
        &self,
        graph: &Graph,
        node_map: &HashMap<*const GraphNodeData, GraphNode>,
    ) -> Graph {
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

            // srcを持たないノードはそのまま返す
            if node.src.is_empty() {
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

        // メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // Sinkノードがある場合
        if let Some(old_sink) = graph.sink() {
            let new_sink_src: Vec<GraphNode> = old_sink
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, &mut visited))
                .collect();

            if let GraphOp::ProgramRoot { ast, outputs } = &old_sink.op {
                let new_sink = GraphNode::new(
                    old_sink.dtype.clone(),
                    GraphOp::ProgramRoot {
                        ast: ast.clone(),
                        outputs: outputs.clone(),
                    },
                    new_sink_src,
                    old_sink.view.clone(),
                );
                new_graph.set_sink(new_sink);
            }
        } else {
            // Sinkがない場合
            let outputs_map = graph.outputs();
            let mut outputs: Vec<_> = outputs_map.iter().collect();
            outputs.sort_by_key(|(name, _)| name.as_str());

            for (name, output_node) in outputs {
                let rebuilt = rebuild_node(output_node, node_map, &mut visited);
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

        for node in &nodes {
            // Viewノードのみを対象
            if !matches!(node.op, GraphOp::View(_)) {
                continue;
            }

            // 入力ノードの被参照数をチェック
            // 被参照数が複数ある場合は融合をスキップ
            if node.src.len() == 1 {
                let input_node = &node.src[0];
                let input_ptr = input_node.as_ptr();
                let ref_count = ref_counts.get(&input_ptr).copied().unwrap_or(0);

                if ref_count > 1 {
                    continue;
                }
            }

            // Viewノードをマージ
            if let Some(merged_node) = self.merge_view_node(node) {
                let new_graph = self.replace_view_in_graph(graph, node, merged_node);
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
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);

        // a + b を計算
        let sum = a.clone() + b.clone();

        // sumに対してpermuteを適用（Viewノードが作られる）
        let permuted = sum.view(sum.view.clone().permute(vec![1, 0]));

        graph.output("result", permuted);

        let suggestions = suggester.suggest(&graph);

        // Viewノードが1つあるので、1つの提案が生成される
        assert_eq!(suggestions.len(), 1);

        // 提案されたグラフを確認
        let new_graph = &suggestions[0];
        let outputs = new_graph.outputs();
        let result = outputs.get("result").unwrap();

        // 結果ノードがViewノードではなく、Elementwise(Add)ノードであることを確認
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

        // 入力メタデータが正しくコピーされていることを確認
        assert_eq!(new_graph.input_metas().len(), 1);
        assert_eq!(new_graph.input_metas()[0].name, "a");
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
            matches!(result_node.op, GraphOp::Kernel { .. }),
            "Should be Custom node"
        );

        // Viewを適用した新しいグラフを作成
        let permuted = result_node.view(result_node.view.clone().permute(vec![1, 0]));
        let mut graph_with_view = Graph::new();
        graph_with_view.copy_input_metas_from(lowered_graph);
        graph_with_view.output("result", permuted);

        // ViewMergeSuggesterでCustom→ViewをCustom[View適用済み]にマージ
        let merged_graphs = view_suggester.suggest(&graph_with_view);
        assert_eq!(merged_graphs.len(), 1, "Should produce 1 merged graph");

        // マージ後のグラフを確認
        let merged_outputs = merged_graphs[0].outputs();
        let merged_result = merged_outputs.get("result").unwrap();

        // 結果がCustomノードであることを確認（Viewノードではない）
        assert!(
            matches!(merged_result.op, GraphOp::Kernel { .. }),
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

    #[test]
    fn test_allows_ndim_change() {
        // Buffer/Const/Custom は次元数変更を許可
        assert!(ViewMergeSuggester::allows_ndim_change(&GraphOp::Buffer {
            name: "a".to_string()
        }));
        assert!(ViewMergeSuggester::allows_ndim_change(&GraphOp::Const(
            1.0f32.into()
        )));
        assert!(ViewMergeSuggester::allows_ndim_change(&GraphOp::Kernel {
            ast: crate::ast::AstNode::Block {
                statements: vec![],
                scope: Box::new(crate::ast::Scope::new())
            },
            input_buffers: None
        }));

        // Elementwise等は許可しない
        assert!(!ViewMergeSuggester::allows_ndim_change(
            &GraphOp::Elementwise {
                op: crate::graph::ElementwiseOp::Add
            }
        ));
    }

    #[test]
    fn test_no_merge_ndim_change_for_elementwise() {
        let suggester = ViewMergeSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);

        let sum = a + b;

        // Elementwiseに対してunsqueezeを適用（次元数が変わる）
        let unsqueezed = sum.view(sum.view.clone().unsqueeze(0)); // [10, 20] -> [1, 10, 20]

        graph.output("result", unsqueezed);

        let suggestions = suggester.suggest(&graph);

        // Elementwiseノードは次元数変更を許可しないのでマージされない
        assert_eq!(suggestions.len(), 0);
    }
}
