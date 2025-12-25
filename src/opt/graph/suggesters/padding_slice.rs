//! パディング・スライス変換Suggester
//!
//! SIMD演算やタイリング最適化のために、演算の入出力を
//! 指定されたアライメントに揃える変換を提案します。
//!
//! # 変換パターン
//!
//! ```text
//! 入力 [N] → 演算 → 出力 [N]
//! ↓
//! 入力 [N] → Pad → [N'] → 演算 → [N''] → Slice → 出力 [N]
//! ```
//!
//! ここで N' = ceil(N / alignment) * alignment
//!
//! # パディング値
//!
//! - Elementwise演算: 0.0（結果はスライスで切り捨てられる）
//! - Reduce演算: 中性値（Sum→0, Prod→1, Max→-∞）

use crate::graph::ops::{GraphOp, ReduceOp};
use crate::graph::shape::{Expr, View};
use crate::graph::{DType, Graph, GraphNode, GraphNodeData};
use crate::opt::graph::{GraphSuggester, SuggestResult};
use std::collections::{HashMap, HashSet};

/// パディング・スライス変換を提案するSuggester
///
/// SIMD演算やタイリング最適化のために、演算の入出力を
/// 指定されたアライメントに揃える変換を提案します。
///
/// # 例
///
/// ```no_run
/// use harp::opt::graph::suggesters::PaddingSliceSuggester;
/// use harp::opt::graph::GraphSuggester;
/// use harp::prelude::*;
///
/// // 単一のアライメント値
/// let suggester = PaddingSliceSuggester::new(vec![4]);
///
/// // 複数のアライメント値（各値に対して候補を生成）
/// let suggester = PaddingSliceSuggester::new(vec![4, 8, 16]);
///
/// let mut graph = Graph::new();
/// let a = graph.input("a", DType::F32, vec![10]); // 10 % 4 != 0
/// let b = graph.input("b", DType::F32, vec![10]);
/// let c = a + b;
/// graph.output("c", c);
///
/// let suggestions = suggester.suggest(&graph);
/// // 各アライメント値に対して変換候補が生成される
/// ```
pub struct PaddingSliceSuggester {
    /// アライメント値のリスト（2のべき乗を推奨）
    alignments: Vec<usize>,
}

impl PaddingSliceSuggester {
    /// 新しいPaddingSliceSuggesterを作成
    ///
    /// # 引数
    /// * `alignments` - アライメント値のリスト（例: vec![4, 8, 16]）
    ///
    /// # パニック
    /// * alignments が空の場合
    /// * いずれかの値が 1 より小さい場合
    pub fn new(alignments: Vec<usize>) -> Self {
        assert!(!alignments.is_empty(), "alignments must not be empty");
        for &a in &alignments {
            assert!(a >= 1, "alignment must be at least 1, got {}", a);
        }
        Self { alignments }
    }

    /// 指定されたshapeでパディングが必要な軸を全て検出
    ///
    /// 静的shapeのみを対象とします（動的shapeは現時点ではスキップ）。
    fn find_unaligned_axes(&self, shape: &[Expr], alignment: usize) -> Vec<usize> {
        shape
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| {
                // 静的に評価可能な場合のみチェック
                if let Some(size) = dim.as_const() {
                    let size = size as usize;
                    if !size.is_multiple_of(alignment) {
                        Some(axis)
                    } else {
                        None
                    }
                } else {
                    // 動的shapeは現時点ではスキップ
                    // （slice()が静的のみ対応のため）
                    None
                }
            })
            .collect()
    }

    /// 指定された軸のパディング量を計算（静的）
    fn calculate_padding_static(&self, size: usize, alignment: usize) -> usize {
        let padded_size = size.div_ceil(alignment) * alignment;
        padded_size - size
    }

    /// Reduce演算の中性値をf32として取得
    fn get_reduce_neutral_value(dtype: &DType, reduce_op: &ReduceOp) -> f32 {
        match reduce_op {
            ReduceOp::Sum => 0.0,
            ReduceOp::Prod => 1.0,
            ReduceOp::Max => match dtype {
                DType::Bool => 0.0, // false
                DType::I32 => i32::MIN as f32,
                _ => f32::NEG_INFINITY,
            },
        }
    }

    /// Elementwise演算に対するパディング・スライス変換を生成
    fn transform_elementwise(
        &self,
        node: &GraphNode,
        target_axis: usize,
        alignment: usize,
    ) -> Option<GraphNode> {
        let shape = node.view.shape();

        // 対象軸のサイズを取得（静的のみ）
        let target_size = shape[target_axis].as_const()? as usize;
        let padding_amount = self.calculate_padding_static(target_size, alignment);

        if padding_amount == 0 {
            return None;
        }

        // 各入力をパディング
        let padded_inputs: Vec<GraphNode> = node
            .src
            .iter()
            .map(|src| {
                let src_shape = src.view.shape();
                // スカラーはパディング不要
                if src_shape.is_empty() {
                    return src.clone();
                }

                // パディング対象軸が入力にも存在する場合のみパディング
                if target_axis < src_shape.len() {
                    // 入力の対象軸サイズを確認
                    if let Some(src_axis_size) = src_shape[target_axis].as_const() {
                        let src_axis_size = src_axis_size as usize;
                        // 入力軸サイズがsrcと異なる場合（ブロードキャストなど）はスキップ
                        if src_axis_size != target_size {
                            return src.clone();
                        }
                    }

                    let padding: Vec<(usize, usize)> = (0..src_shape.len())
                        .map(|i| {
                            if i == target_axis {
                                (0, padding_amount)
                            } else {
                                (0, 0)
                            }
                        })
                        .collect();
                    src.pad(padding, 0.0)
                } else {
                    src.clone()
                }
            })
            .collect();

        // パディング後の出力shapeを計算
        let padded_output_shape: Vec<Expr> = shape
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == target_axis {
                    Expr::from((target_size + padding_amount) as i64)
                } else {
                    s.clone()
                }
            })
            .collect();

        // パディングされた入力で演算ノードを再作成
        let padded_node = GraphNode::new(
            node.dtype.clone(),
            node.op.clone(),
            padded_inputs,
            View::contiguous(padded_output_shape),
        );

        // スライスで元のサイズに戻す
        let slice_ranges: Vec<(usize, usize)> = shape
            .iter()
            .map(|s| {
                let size = s.as_const().unwrap_or(0) as usize;
                (0, size)
            })
            .collect();

        // 全ての軸が静的の場合のみスライスを適用
        let all_static = shape.iter().all(|s| s.as_const().is_some());
        if !all_static {
            return None;
        }

        Some(padded_node.slice(slice_ranges))
    }

    /// Reduce演算に対するパディング・スライス変換を生成
    fn transform_reduce(
        &self,
        node: &GraphNode,
        reduce_op: &ReduceOp,
        reduce_axis: usize,
        target_axis: usize,
        alignment: usize,
    ) -> Option<GraphNode> {
        // 入力のshapeを取得（Reduceの入力は1つ）
        let input = node.src.first()?;
        let input_shape = input.view.shape();
        let input_ndim = input_shape.len();

        // 対象軸のサイズを取得（静的のみ）
        let target_size = input_shape[target_axis].as_const()? as usize;
        let padding_amount = self.calculate_padding_static(target_size, alignment);

        if padding_amount == 0 {
            return None;
        }

        // Reduce軸をパディングする場合は中性値を使用
        let pad_value = if target_axis == reduce_axis {
            Self::get_reduce_neutral_value(&input.dtype, reduce_op)
        } else {
            0.0 // Reduce軸以外は0でOK（結果がスライスで切り捨てられる）
        };

        // 入力をパディング
        let padding: Vec<(usize, usize)> = (0..input_ndim)
            .map(|i| {
                if i == target_axis {
                    (0, padding_amount)
                } else {
                    (0, 0)
                }
            })
            .collect();

        let padded_input = input.pad(padding, pad_value);

        // パディング後の出力shapeを計算
        // Reduce軸は削除されるため、出力shapeの計算が必要
        let padded_output_shape: Vec<Expr> = if target_axis == reduce_axis {
            // Reduce軸自体をパディングする場合、出力shapeは変わらない
            node.view.shape().to_vec()
        } else {
            // Reduce軸以外をパディングする場合
            let output_shape = node.view.shape();

            // 出力での対象軸位置を計算（Reduce軸より前か後かで変わる）
            let output_axis = if target_axis > reduce_axis {
                target_axis - 1
            } else {
                target_axis
            };

            output_shape
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if i == output_axis {
                        Expr::from((target_size + padding_amount) as i64)
                    } else {
                        s.clone()
                    }
                })
                .collect()
        };

        // パディングされた入力でReduce演算ノードを再作成
        let padded_node = GraphNode::new(
            node.dtype.clone(),
            node.op.clone(),
            vec![padded_input],
            View::contiguous(padded_output_shape),
        );

        // Reduce軸がパディング対象の場合はスライス不要
        if target_axis == reduce_axis {
            return Some(padded_node);
        }

        // Reduce軸以外がパディング対象の場合はスライスが必要
        let output_shape = node.view.shape();
        let all_static = output_shape.iter().all(|s| s.as_const().is_some());
        if !all_static {
            return None;
        }

        let slice_ranges: Vec<(usize, usize)> = output_shape
            .iter()
            .map(|s| {
                let size = s.as_const().unwrap_or(0) as usize;
                (0, size)
            })
            .collect();

        Some(padded_node.slice(slice_ranges))
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

    /// グラフ内の特定ノードを置き換えた新しいグラフを作成
    fn replace_node_in_graph(
        &self,
        graph: &Graph,
        old_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
        node_map.insert(old_node.as_ptr(), new_node);

        let mut cache: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            cache: &mut HashMap<*const GraphNodeData, GraphNode>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // Bufferノードは常に元のノードをそのまま返す
            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            // キャッシュを確認
            if let Some(cached) = cache.get(&ptr) {
                return cached.clone();
            }

            let new_src: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, cache))
                .collect();

            let src_changed = new_src
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            let result = if !src_changed {
                node.clone()
            } else {
                GraphNode::new(
                    node.dtype.clone(),
                    node.op.clone(),
                    new_src,
                    node.view.clone(),
                )
            };

            cache.insert(ptr, result.clone());
            result
        }

        let mut new_graph = Graph::new();

        // メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // 全ての出力ノードを再構築
        for (name, output_node) in graph.outputs() {
            let rebuilt = rebuild_node(output_node, &node_map, &mut cache);
            new_graph.set_output_node(name.clone(), rebuilt);
        }

        new_graph
    }
}

impl Default for PaddingSliceSuggester {
    fn default() -> Self {
        Self::new(vec![4, 8, 16])
    }
}

impl GraphSuggester for PaddingSliceSuggester {
    fn name(&self) -> &'static str {
        "PaddingSlice"
    }

    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        // 各アライメント値についてループ
        for &alignment in &self.alignments {
            for node in &nodes {
                match &node.op {
                    GraphOp::Elementwise { .. } | GraphOp::FusedElementwise { .. } => {
                        let shape = node.view.shape();
                        let unaligned_axes = self.find_unaligned_axes(shape, alignment);

                        // 各非アライン軸に対して個別の提案を生成
                        for axis in unaligned_axes {
                            if let Some(transformed) =
                                self.transform_elementwise(node, axis, alignment)
                            {
                                let new_graph =
                                    self.replace_node_in_graph(graph, node, transformed);
                                suggestions.push(SuggestResult::with_description(
                                    new_graph,
                                    self.name(),
                                    format!(
                                        "Pad axis {} to alignment {} for elementwise op",
                                        axis, alignment
                                    ),
                                ));
                            }
                        }
                    }

                    GraphOp::Reduce { op, axis, .. } => {
                        let input = &node.src[0];
                        let input_shape = input.view.shape();
                        let unaligned_axes = self.find_unaligned_axes(input_shape, alignment);

                        for target_axis in unaligned_axes {
                            if let Some(transformed) =
                                self.transform_reduce(node, op, *axis, target_axis, alignment)
                            {
                                let new_graph =
                                    self.replace_node_in_graph(graph, node, transformed);
                                suggestions.push(SuggestResult::with_description(
                                    new_graph,
                                    self.name(),
                                    format!(
                                        "Pad axis {} to alignment {} for reduce {:?}",
                                        target_axis, alignment, op
                                    ),
                                ));
                            }
                        }
                    }

                    GraphOp::FusedElementwiseReduce {
                        reduce_op, axes, ..
                    } => {
                        let input_shape = node.src[0].view.shape();
                        let unaligned_axes = self.find_unaligned_axes(input_shape, alignment);

                        // FusedElementwiseReduceの場合、最初のreduce軸を使用
                        let Some(&first_reduce_axis) = axes.first() else {
                            continue;
                        };

                        for target_axis in unaligned_axes {
                            if let Some(transformed) = self.transform_reduce(
                                node,
                                reduce_op,
                                first_reduce_axis,
                                target_axis,
                                alignment,
                            ) {
                                let new_graph =
                                    self.replace_node_in_graph(graph, node, transformed);
                                suggestions.push(SuggestResult::with_description(
                                    new_graph,
                                    self.name(),
                                    format!(
                                        "Pad axis {} to alignment {} for fused reduce {:?}",
                                        target_axis, alignment, reduce_op
                                    ),
                                ));
                            }
                        }
                    }

                    _ => {}
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
    fn test_elementwise_padding() {
        // alignment=4で、サイズ10の軸をパディング（10 → 12）
        let suggester = PaddingSliceSuggester::new(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 1つの候補（軸0をパディング）
        assert_eq!(suggestions.len(), 1);
        assert!(suggestions[0].description.contains("axis 0"));
    }

    #[test]
    fn test_already_aligned() {
        // alignment=4で、サイズ12は既にアライン済み
        let suggester = PaddingSliceSuggester::new(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![12]);
        let b = graph.input("b", DType::F32, vec![12]);

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 候補なし（既にアライン済み）
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_reduce_sum_neutral() {
        // Reduce Sum: パディング値は0（中性値）
        let suggester = PaddingSliceSuggester::new(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let c = a.reduce_sum(0); // axis=0でreduce
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // axis=0 (size=10) のパディング候補が生成
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_reduce_max_neutral() {
        // Reduce Max: パディング値は-∞（中性値）
        let suggester = PaddingSliceSuggester::new(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);

        let c = a.reduce_max(0);
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // axis=0のパディング候補が生成
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_reduce_prod_neutral() {
        // Reduce Prod: パディング値は1（中性値）
        let suggester = PaddingSliceSuggester::new(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);

        let c = a.reduce_mul(0);
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_multi_axis_candidates() {
        // 複数軸が非アラインの場合、各軸に対して候補を生成
        let suggester = PaddingSliceSuggester::new(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 15]); // 両軸とも非アライン
        let b = graph.input("b", DType::F32, vec![10, 15]);

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 2つの候補（axis=0とaxis=1）
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_scalar_skip() {
        // スカラー入力はパディング対象外
        let suggester = PaddingSliceSuggester::new(vec![4]);

        let mut graph = Graph::new();
        let a = graph.input::<usize, _>("a", DType::F32, vec![]); // スカラー
        let b = graph.input::<usize, _>("b", DType::F32, vec![]);

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 候補なし（スカラーにはパディング不要）
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_multiple_alignments() {
        // 複数のアライメント値で候補を生成
        let suggester = PaddingSliceSuggester::new(vec![4, 8]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 2つの候補（alignment=4とalignment=8、各軸0に対して）
        assert_eq!(suggestions.len(), 2);
        assert!(
            suggestions
                .iter()
                .any(|s| s.description.contains("alignment 4"))
        );
        assert!(
            suggestions
                .iter()
                .any(|s| s.description.contains("alignment 8"))
        );
    }

    #[test]
    fn test_neutral_value_sum() {
        assert_eq!(
            PaddingSliceSuggester::get_reduce_neutral_value(&DType::F32, &ReduceOp::Sum),
            0.0
        );
    }

    #[test]
    fn test_neutral_value_prod() {
        assert_eq!(
            PaddingSliceSuggester::get_reduce_neutral_value(&DType::F32, &ReduceOp::Prod),
            1.0
        );
    }

    #[test]
    fn test_neutral_value_max() {
        assert_eq!(
            PaddingSliceSuggester::get_reduce_neutral_value(&DType::F32, &ReduceOp::Max),
            f32::NEG_INFINITY
        );
    }
}
