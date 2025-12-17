//! Lowering Suggester
//!
//! GraphOpをKernelノードに変換するSuggester。
//! 各GraphOpに対して、対応するAstNode::Functionを生成し、
//! Kernel演算として統合します。

mod elementwise;
mod fold;
mod helpers;
mod other;
mod reduce;

use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::{GraphSuggester, SuggestResult};
use std::collections::{HashMap, HashSet};

// サブモジュールから関数を再エクスポート
pub use elementwise::is_pure_const_node;

/// GraphOpをKernelノードに変換するSuggester
///
/// 各計算ノードを等価なKernelノード（AstNode::Functionを保持）に変換します。
/// これにより、すべての計算がAST関数として統一され、
/// ASTレベルの最適化が可能になります。
///
/// # 並列化について
///
/// このSuggesterは逐次実行（Sequential）のみで候補を生成します。
/// 並列化はASTレベルの最適化（Global/LocalParallelizationSuggester）で行います。
pub struct LoweringSuggester;

/// カーネル/関数の種類を表すプレフィックス
#[derive(Debug, Clone, Copy)]
enum KernelKind {
    /// Elementwise演算
    Elementwise,
    /// ElementwiseReduce演算 (FusedElementwiseReduceを含む)
    ElementwiseReduce,
    /// Cumulative演算 (FusedElementwiseCumulativeを含む)
    Cumulative,
    /// Reduce演算
    Reduce,
    /// その他の演算 (Contiguous, Cast, etc.)
    Other,
}

impl KernelKind {
    /// プレフィックス文字列を取得
    fn prefix(&self) -> &'static str {
        match self {
            KernelKind::Elementwise => "E",
            KernelKind::ElementwiseReduce => "ER",
            KernelKind::Cumulative => "C",
            KernelKind::Reduce => "R",
            KernelKind::Other => "O",
        }
    }
}

impl LoweringSuggester {
    /// 新しいLoweringSuggesterを作成
    pub fn new() -> Self {
        LoweringSuggester
    }

    /// ノードの種類とshapeからカーネル/関数名を生成
    ///
    /// 命名規則:
    /// - プレフィックス: E (Elementwise), ER (ElementwiseReduce), C (Cumulative), R (Reduce), O (Other)
    /// - 出力shape: `_`区切りで追加
    /// - 例: shape [2, 4] のElementwise演算 → `E_2_4`
    fn generate_kernel_name(
        &self,
        kind: KernelKind,
        shape: &[crate::graph::shape::Expr],
    ) -> String {
        let mut name = kind.prefix().to_string();

        for dim in shape {
            name.push('_');
            match dim {
                crate::graph::shape::Expr::Const(val) => {
                    name.push_str(&val.to_string());
                }
                crate::graph::shape::Expr::Var(var_name) => {
                    name.push_str(var_name);
                }
                _ => {
                    name.push_str("dyn");
                }
            }
        }

        name
    }

    /// ノードからカーネル種類を判定
    fn get_kernel_kind(&self, op: &GraphOp) -> KernelKind {
        match op {
            GraphOp::Elementwise { .. } | GraphOp::FusedElementwise { .. } => {
                KernelKind::Elementwise
            }
            GraphOp::FusedElementwiseReduce { .. } => KernelKind::ElementwiseReduce,
            GraphOp::Cumulative { .. } | GraphOp::FusedElementwiseCumulative { .. } => {
                KernelKind::Cumulative
            }
            GraphOp::Reduce { .. } => KernelKind::Reduce,
            _ => KernelKind::Other,
        }
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

    /// ノードをKernelノードに変換可能かチェック
    fn can_lower(&self, node: &GraphNode) -> bool {
        // 基本的なノードタイプをチェック
        if matches!(
            node.op,
            GraphOp::Buffer { .. }
                | GraphOp::Const(_)
                | GraphOp::ComplexConst { .. }
                | GraphOp::View(_)
                | GraphOp::Kernel { .. }
                // サブグラフ関連のノードは直接lowerできない
                // これらは別のグラフを呼び出すメタ演算であり、
                // 呼び出し先のグラフを個別に最適化する必要がある
                | GraphOp::SubgraphCall { .. }
                | GraphOp::SubgraphOutput { .. }
        ) {
            return false;
        }

        // Elementwise演算の場合、すべての入力が連続している必要がある
        if matches!(node.op, GraphOp::Elementwise { .. }) {
            for src in &node.src {
                if !src.view.is_contiguous() {
                    log::debug!(
                        "LoweringSuggester: skipping Elementwise due to non-contiguous input view"
                    );
                    return false;
                }
            }
        }

        // ソースがすべてlowered済み（Buffer, Kernel, View, Const）であることを確認
        // これにより、依存関係の順序でloweringが行われる
        for src in &node.src {
            if !self.is_lowered_or_passthrough(src) {
                log::trace!(
                    "LoweringSuggester: skipping {:?} because source {:?} is not yet lowered",
                    std::mem::discriminant(&node.op),
                    std::mem::discriminant(&src.op)
                );
                return false;
            }
        }
        log::debug!(
            "LoweringSuggester: can_lower {:?} = true (all sources are lowered)",
            std::mem::discriminant(&node.op)
        );

        true
    }

    /// ノードがlowered済みまたはパススルー（Buffer, Kernel, View, Const）かをチェック
    fn is_lowered_or_passthrough(&self, node: &GraphNode) -> bool {
        match &node.op {
            GraphOp::Buffer { .. }
            | GraphOp::Const(_)
            | GraphOp::ComplexConst { .. }
            | GraphOp::Kernel { .. } => true,
            GraphOp::View(_) => {
                // Viewノードはすべてのソースがlowered済みの場合のみパススルー
                let result = node.src.iter().all(|s| self.is_lowered_or_passthrough(s));
                if !result {
                    log::trace!(
                        "LoweringSuggester: View is NOT passthrough, its source is {:?}",
                        node.src.first().map(|s| std::mem::discriminant(&s.op))
                    );
                }
                result
            }
            _ => false,
        }
    }

    /// GraphOpをKernelノードに変換
    fn lower_to_custom(&self, node: &GraphNode) -> Option<GraphNode> {
        let kind = self.get_kernel_kind(&node.op);
        let name = self.generate_kernel_name(kind, node.view.shape());

        let ast = match &node.op {
            GraphOp::Elementwise { op, .. } => {
                elementwise::build_elementwise_function(node, op, &name)
            }
            GraphOp::Reduce { op, axis, .. } => {
                reduce::build_reduce_function(node, op, *axis, &name)
            }
            GraphOp::Cumulative { op, axis, .. } => {
                reduce::build_cumulative_function(node, op, *axis, &name)
            }
            GraphOp::Contiguous => other::build_contiguous_function(node, &name),
            GraphOp::FusedElementwise { expr, .. } => {
                elementwise::build_fused_elementwise_function(node, expr, &name)
            }
            GraphOp::FusedElementwiseReduce {
                expr,
                reduce_op,
                axes,
                ..
            } => {
                reduce::build_fused_elementwise_reduce_function(node, expr, reduce_op, axes, &name)
            }
            GraphOp::FusedElementwiseCumulative {
                expr,
                cumulative_op,
                axis,
                ..
            } => reduce::build_fused_elementwise_cumulative_function(
                node,
                expr,
                cumulative_op,
                *axis,
                &name,
            ),
            GraphOp::Pad { padding, value } => other::build_pad_function(node, padding, *value),
            GraphOp::Slice { ranges } => other::build_slice_function(node, ranges, &name),
            GraphOp::Concat { axis } => other::build_concat_function(node, *axis),
            GraphOp::Rand => other::build_rand_function(node, &name),
            GraphOp::Arange => other::build_arange_function(node, &name),
            GraphOp::Cast { target_dtype, .. } => {
                other::build_cast_function(node, target_dtype, &name)
            }
            GraphOp::Real => other::build_real_function(node, &name),
            GraphOp::Imag => other::build_imag_function(node, &name),
            GraphOp::ComplexFromParts => other::build_complex_from_parts_function(node, &name),
            GraphOp::Fold {
                output_size,
                kernel_size,
                stride,
                dilation,
                groups,
            } => fold::build_fold_function(
                node,
                output_size,
                kernel_size,
                stride,
                dilation,
                *groups,
                &name,
            ),
            GraphOp::FusedReduce { .. } => {
                // FusedReduceはタプル出力が必要なので後で実装
                return None;
            }
            GraphOp::SubgraphCall { .. } | GraphOp::SubgraphOutput { .. } => {
                // サブグラフ関連のノードは直接lowerできない
                // これらは別のグラフを呼び出すメタ演算
                return None;
            }
            _ => return None,
        }?;

        // Kernelノードを作成
        // srcからView経由でInputまで辿り、対応するBufferノードを収集
        let non_const_src: Vec<_> = node
            .src
            .iter()
            .filter(|s| !matches!(s.op, GraphOp::Const(_)) && !is_pure_const_node(s))
            .cloned()
            .collect();
        let mut new_src = helpers::collect_input_buffers(&non_const_src);

        // 出力バッファーを作成
        let output_buffer_name = format!("output_{}", name);
        let output_buffer = GraphNode::new(
            node.dtype.clone(),
            GraphOp::Buffer {
                name: output_buffer_name,
            },
            vec![],
            node.view.clone(),
        );
        new_src.push(output_buffer);

        Some(GraphNode::new(
            node.dtype.clone(),
            GraphOp::Kernel {
                ast,
                input_buffers: None,
            },
            new_src,
            node.view.clone(),
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
        node_map.insert(old_node.as_ptr(), new_node);

        // キャッシュをHashMapに変更（HashSetではなく）
        // 再構築済みノードをキャッシュして、共有ノードの正しい参照を保持
        let mut cache: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            cache: &mut HashMap<*const GraphNodeData, GraphNode>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            // キャッシュをチェック（再構築済みノードを返す）
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

            // 再構築結果をキャッシュ
            cache.insert(ptr, result.clone());
            result
        }

        let mut new_graph = Graph::new();

        // 入力・出力メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // 全ての出力ノードを再構築
        for (name, output_node) in graph.outputs() {
            let rebuilt = rebuild_node(output_node, &node_map, &mut cache);
            new_graph.set_output_node(name.clone(), rebuilt);
        }

        // shape変数のデフォルト値をコピー
        for (name, value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(name.clone(), *value);
        }

        new_graph
    }
}

impl Default for LoweringSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for LoweringSuggester {
    fn name(&self) -> &'static str {
        "Lowering"
    }

    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        let mut lowerable_count = 0;
        let mut already_custom = 0;
        let mut lowered_count = 0;

        for node in &nodes {
            if matches!(node.op, GraphOp::Kernel { .. }) {
                already_custom += 1;
                continue;
            }

            if !self.can_lower(node) {
                continue;
            }

            lowerable_count += 1;

            if let Some(custom_node) = self.lower_to_custom(node) {
                let new_graph = self.replace_node_in_graph(graph, node, custom_node);
                suggestions.push(SuggestResult::new(new_graph, self.name()));
                lowered_count += 1;
            } else {
                log::debug!(
                    "LoweringSuggester: failed to lower {:?}",
                    std::mem::discriminant(&node.op)
                );
            }
        }

        log::debug!(
            "LoweringSuggester: {} nodes total, {} already custom, {} lowerable, {} lowered candidates",
            nodes.len(),
            already_custom,
            lowerable_count,
            lowered_count
        );

        suggestions
    }
}

#[cfg(test)]
mod tests;
