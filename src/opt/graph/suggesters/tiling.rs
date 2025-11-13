use crate::graph::shape::Expr;
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// ループのタイル化に相当するView変更を提案するSuggester
///
/// 各軸を個別にタイル化し、キャッシュ効率を向上させます。
///
/// # タイル化の手順
///
/// 例: shape [128, 256] の軸 0 を tile_size=32 でタイル化
/// - reshape: [128, 256] -> [4, 32, 256]
///
/// これにより、ループが外側ループ（4回）と内側ループ（32要素）に分割され、
/// キャッシュの局所性が向上します。
///
/// ループの順序入れ替えはAST最適化で行われるため、ここではreshapeのみを実施します。
///
/// 複数の軸をタイル化したい場合は、このSuggesterを複数回適用します。
pub struct TilingSuggester {
    /// 試行するタイルサイズの候補（1次元用）
    pub tile_sizes: Vec<usize>,
}

impl TilingSuggester {
    /// 新しいTilingSuggesterを作成
    pub fn new(tile_sizes: Vec<usize>) -> Self {
        Self { tile_sizes }
    }

    /// デフォルトのタイルサイズを使用
    pub fn with_default_tile_sizes() -> Self {
        Self::new(vec![8, 16, 32, 64, 128])
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

    /// ノードの指定された軸をタイル化
    ///
    /// 軸のサイズがタイルサイズで割り切れない場合はNoneを返す
    fn tile_node_axis(&self, node: &GraphNode, axis: usize, tile_size: usize) -> Option<GraphNode> {
        // Inputノードは変更しない
        if matches!(node.op, GraphOp::Input) {
            return None;
        }

        // Viewが連続していない場合はタイル化できない
        if !node.view.is_contiguous() {
            return None;
        }

        // 全ての入力ノードもcontiguousである必要がある
        for src in &node.src {
            if !src.view.is_contiguous() {
                return None;
            }
        }

        let shape = node.view.shape();

        // 全ての入力ノードが出力ノードと同じ形状である必要がある
        // (ブロードキャストされた入力はタイル化できない)
        for src in &node.src {
            if src.view.shape() != shape {
                return None;
            }
        }

        // 軸が範囲外の場合はスキップ
        if axis >= shape.len() {
            return None;
        }

        // 軸のサイズを取得（定数の場合のみタイル化可能）
        let axis_size = match &shape[axis] {
            Expr::Const(size) => *size as usize,
            _ => return None, // シンボリックな軸はスキップ
        };

        // タイルサイズで割り切れない、または既にタイルサイズ以下の場合はスキップ
        if axis_size % tile_size != 0 || axis_size <= tile_size {
            return None;
        }

        let num_tiles = axis_size / tile_size;

        // 新しいshapeを構築: [..., N, ...] -> [..., N/tile, tile, ...]
        let mut new_shape = Vec::new();
        for (i, s) in shape.iter().enumerate() {
            if i == axis {
                new_shape.push(Expr::from(num_tiles));
                new_shape.push(Expr::from(tile_size));
            } else {
                new_shape.push(s.clone());
            }
        }

        // reshape: [..., N, ...] -> [..., N/tile, tile, ...]
        // ループの順序入れ替えはAST最適化で行われるため、permuteは不要
        let tiled_view = node.view.clone().reshape(new_shape.clone());

        // srcノードにも同じreshapeを適用
        // これにより、入力ノードと出力ノードの次元数が一致し、lowering時にエラーが発生しない
        let new_src: Vec<GraphNode> = node
            .src
            .iter()
            .map(|src| src.reshape(new_shape.clone()))
            .collect();

        // elementwise_strategiesを調整
        // タイル化によって軸が挿入されるため、正しいマッピングが必要
        // 例: 元のshape [64, 128]、軸0をタイル化 → [2, 32, 128]
        // - 新軸0（num_tiles） → 元軸0に対応 → old_strategies[0]
        // - 新軸1（tile_size） → 新規追加 → デフォルト
        // - 新軸2 → 元軸1に対応 → old_strategies[1]
        let old_strategies = &node.elementwise_strategies;
        let mut new_strategies = Vec::new();

        for new_idx in 0..tiled_view.shape().len() {
            if new_idx < axis {
                // タイル化された軸より前の軸: そのままマッピング
                new_strategies.push(old_strategies[new_idx].clone());
            } else if new_idx == axis {
                // タイル化された軸（num_tiles）: 元の軸のstrategyを使用
                new_strategies.push(old_strategies[axis].clone());
            } else if new_idx == axis + 1 {
                // 挿入されたタイル軸（tile_size）: デフォルトのstrategy
                new_strategies.push(crate::graph::ElementwiseStrategy::Sequential {
                    simd_width: 1,
                    unroll_factor: 1,
                });
            } else {
                // タイル化された軸より後の軸: old_strategies[new_idx - 1]を使用
                // （1つ軸が挿入されたため、インデックスを1つ戻す）
                new_strategies.push(old_strategies[new_idx - 1].clone());
            }
        }

        // 新しいノードを作成
        Some(GraphNode::with_elementwise_strategies(
            node.dtype.clone(),
            node.op.clone(),
            new_src,
            tiled_view,
            new_strategies,
        ))
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

            // Inputノードは常に元のノードをそのまま返す
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

        // 入力ノードを保持
        for (name, weak_input) in graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                new_graph.register_input(name.clone(), input_node);
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

impl Default for TilingSuggester {
    fn default() -> Self {
        Self::with_default_tile_sizes()
    }
}

impl GraphSuggester for TilingSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各ノードの各軸について、各タイルサイズでタイル化を試みる
        for node in &nodes {
            let ndim = node.view.ndim();

            // 各軸について
            for axis in 0..ndim {
                // 各タイルサイズについて
                for &tile_size in &self.tile_sizes {
                    if let Some(tiled_node) = self.tile_node_axis(node, axis, tile_size) {
                        let new_graph = self.replace_node_in_graph(graph, node, tiled_node);
                        suggestions.push(new_graph);
                    }
                }
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_tiling_suggester_basic() {
        // tile_size=32 でテスト
        let suggester = TilingSuggester::new(vec![32]);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 128])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 128])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 64と128の両方が32で割り切れるので、2つの軸それぞれについて提案が生成される
        // 軸0: [64, 128] -> [2, 32, 128]
        // 軸1: [64, 128] -> [64, 4, 32]
        assert_eq!(suggestions.len(), 2);

        // 軸0をタイル化した結果を確認
        let graph0 = &suggestions[0];
        let output0 = graph0.outputs().get("c").unwrap();
        assert_eq!(
            output0.view.shape(),
            &[Expr::from(2), Expr::from(32), Expr::from(128)]
        );

        // 軸1をタイル化した結果を確認
        let graph1 = &suggestions[1];
        let output1 = graph1.outputs().get("c").unwrap();
        assert_eq!(
            output1.view.shape(),
            &[Expr::from(64), Expr::from(4), Expr::from(32)]
        );
    }

    #[test]
    fn test_tiling_suggester_no_divisible() {
        let suggester = TilingSuggester::new(vec![32]);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![63, 127]) // 32で割り切れない
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![63, 127])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // どちらの軸も32で割り切れないので、提案は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_tiling_suggester_multiple_tile_sizes() {
        let suggester = TilingSuggester::new(vec![16, 32]);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 128])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 128])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 2つの軸 × 2つのタイルサイズ = 4つの提案
        assert_eq!(suggestions.len(), 4);
    }

    #[test]
    fn test_tiling_suggester_skip_input() {
        let suggester = TilingSuggester::new(vec![32]);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 128])
            .build();

        // Inputノードをそのまま出力
        graph.output("a", a);

        let suggestions = suggester.suggest(&graph);

        // Inputノードは変更されないので、提案は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_tiling_suggester_skip_small_axis() {
        let suggester = TilingSuggester::new(vec![64]);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![32, 128]) // 軸0は64以下
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![32, 128])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 軸0は32でタイルサイズ64以下なのでスキップ
        // 軸1は128で64の倍数なのでタイル化可能
        assert_eq!(suggestions.len(), 1);

        let graph0 = &suggestions[0];
        let output0 = graph0.outputs().get("c").unwrap();
        assert_eq!(
            output0.view.shape(),
            &[Expr::from(32), Expr::from(2), Expr::from(64)]
        );
    }

    #[test]
    fn test_tiling_with_lowering() {
        use crate::lowerer::lower;

        let suggester = TilingSuggester::new(vec![32]);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 128])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 128])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);
        assert_eq!(suggestions.len(), 2); // 2軸それぞれにタイル化候補

        // 最初の候補をloweringしてみる
        let tiled_graph = suggestions[0].clone();
        let _ast = lower(tiled_graph); // 成功することを期待
    }
}
