use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// View変更ノード（転置）を挿入してループ順序を入れ替えるSuggester
///
/// 全入力を転置 → 演算実行 → 出力を逆転置することで、
/// メモリコピーなしでメモリアクセスパターンを最適化します。
pub struct ViewInsertionSuggester;

impl ViewInsertionSuggester {
    /// 新しいViewInsertionSuggesterを作成
    pub fn new() -> Self {
        ViewInsertionSuggester {}
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

    /// ノードの全入力を転置し、演算後に逆転置を適用
    /// ループの順序を入れ替えることでメモリアクセスパターンを最適化
    fn insert_transpose_around_node(&self, node: &GraphNode, permutation: Vec<usize>) -> GraphNode {
        // 全ての入力を転置（Viewノード挿入、ゼロコスト）
        let new_src: Vec<GraphNode> = node
            .src
            .iter()
            .map(|input| {
                let permuted_view = input.view.clone().permute(permutation.clone());
                GraphNode::new(
                    input.dtype.clone(),
                    GraphOp::View(permuted_view.clone()),
                    vec![input.clone()],
                    permuted_view,
                )
            })
            .collect();

        // ノードのviewも転置
        let node_permuted_view = node.view.clone().permute(permutation.clone());

        // 転置された入力で新しいノードを作成（ループ順序が変わる）
        let permuted_node = GraphNode::with_elementwise_strategies(
            node.dtype.clone(),
            node.op.clone(),
            new_src,
            node_permuted_view,
            node.elementwise_strategies.clone(),
        );

        // 出力を逆転置（Viewノード挿入、ゼロコスト）
        let inv_perm = self.inverse_permutation(&permutation);
        let output_view = permuted_node.view.clone().permute(inv_perm);
        GraphNode::new(
            permuted_node.dtype.clone(),
            GraphOp::View(output_view.clone()),
            vec![permuted_node],
            output_view,
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

    /// permutationの逆変換を計算
    fn inverse_permutation(&self, perm: &[usize]) -> Vec<usize> {
        let mut inv = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        inv
    }

    /// 転置パターンを生成（様々な軸ペアの入れ替え）
    fn generate_transpose_permutations(&self, ndim: usize) -> Vec<Vec<usize>> {
        if ndim < 2 {
            return vec![];
        }

        let mut permutations = Vec::new();

        // 全ての隣接軸ペアを入れ替え
        for i in 0..ndim - 1 {
            let mut perm: Vec<usize> = (0..ndim).collect();
            perm.swap(i, i + 1);
            permutations.push(perm);
        }

        permutations
    }
}

impl Default for ViewInsertionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ViewInsertionSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各ノードについて、様々な転置パターンを試みる
        for node in &nodes {
            // InputノードとViewノードはスキップ
            // ViewノードにView変更を挿入しても意味がないため
            if matches!(node.op, GraphOp::Input | GraphOp::View(_)) {
                continue;
            }

            // 入力がないノードはスキップ
            if node.src.is_empty() {
                continue;
            }

            let ndim = node.view.ndim();

            // 2次元以上のテンソルのみ転置可能
            if ndim < 2 {
                continue;
            }

            // 全ての入力が同じndimを持つか確認
            let all_same_ndim = node.src.iter().all(|input| input.view.ndim() == ndim);
            if !all_same_ndim {
                // 入力の次元数が異なる場合はスキップ
                continue;
            }

            // 様々な転置パターンを生成
            let permutations = self.generate_transpose_permutations(ndim);

            for permutation in permutations {
                let new_node = self.insert_transpose_around_node(node, permutation);
                let new_graph = self.replace_node_in_graph(graph, node, new_node);
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

        // 2次元テンソルなので、1つの転置パターン（最後の2軸入れ替え）が生成される
        // Addノード1つに対して1パターンなので、1つの候補が生成される
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_view_insertion_disabled() {
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

        // 転置が無効なので、候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_view_insertion_3d() {
        let suggester = ViewInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![4, 5, 6])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![4, 5, 6])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 3次元テンソルなので、2つの転置パターンが生成される
        // (0,1,2) -> (1,0,2): 軸0と1を入れ替え
        // (0,1,2) -> (0,2,1): 軸1と2を入れ替え
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_view_insertion_4d() {
        let suggester = ViewInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3, 4, 5])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3, 4, 5])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 4次元テンソルなので、3つの転置パターンが生成される
        // (0,1,2,3) -> (1,0,2,3): 軸0と1を入れ替え
        // (0,1,2,3) -> (0,2,1,3): 軸1と2を入れ替え
        // (0,1,2,3) -> (0,1,3,2): 軸2と3を入れ替え
        assert_eq!(suggestions.len(), 3);
    }

    #[test]
    fn test_inverse_permutation() {
        let suggester = ViewInsertionSuggester::new();

        // 2次元
        assert_eq!(suggester.inverse_permutation(&[1, 0]), vec![1, 0]);

        // 3次元
        assert_eq!(suggester.inverse_permutation(&[1, 0, 2]), vec![1, 0, 2]);
        assert_eq!(suggester.inverse_permutation(&[0, 2, 1]), vec![0, 2, 1]);
        assert_eq!(suggester.inverse_permutation(&[2, 1, 0]), vec![2, 1, 0]);

        // 4次元
        assert_eq!(
            suggester.inverse_permutation(&[1, 0, 2, 3]),
            vec![1, 0, 2, 3]
        );
        assert_eq!(
            suggester.inverse_permutation(&[0, 2, 1, 3]),
            vec![0, 2, 1, 3]
        );
        assert_eq!(
            suggester.inverse_permutation(&[0, 1, 3, 2]),
            vec![0, 1, 3, 2]
        );
    }

    #[test]
    fn test_view_insertion_mixed_dimensions() {
        let suggester = ViewInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // unsqueezeで次元を追加: [10, 20] -> [10, 20, 1]
        let a_unsqueezed = a.view(a.view.clone().unsqueeze(2));

        // expandで次元を拡大: [10, 20, 1] -> [10, 20, 30]
        use crate::graph::shape::Expr;
        let a_expanded = a_unsqueezed.view(a_unsqueezed.view.clone().expand(vec![
            Expr::Const(10),
            Expr::Const(20),
            Expr::Const(30),
        ]));

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20, 30])
            .build();

        // 2次元のaを変換した3次元テンソルと、3次元のbを加算
        let c = a_expanded + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // グラフ内のノード:
        // - a: Input [10, 20] → Inputなのでスキップ
        // - a_unsqueezed: View [10, 20, 1] → Viewノードなのでスキップ
        // - a_expanded: View [10, 20, 30] → Viewノードなのでスキップ
        // - b: Input [10, 20, 30] → Inputなのでスキップ
        // - c: Add [10, 20, 30] (入力: a_expanded [10, 20, 30], b [10, 20, 30]) → 転置可能
        //
        // 3次元なので2パターン × 1ノード(Add) = 2候補
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_view_insertion_skip_mismatched_dimensions() {
        let suggester = ViewInsertionSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // このケースでは、aを[10, 1]にunsqueezeして、[10, 20]にexpandする必要がある
        // しかし、これは複雑なので、単純にreduce_sumで次元を合わせる
        let b_sum = b.reduce_sum(1); // [10, 20] -> [10]

        let c = a + b_sum;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 1次元テンソルなので、転置パターンは生成されない（2次元未満）
        assert_eq!(suggestions.len(), 0);
    }
}
