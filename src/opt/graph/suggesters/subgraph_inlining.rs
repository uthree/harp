//! SubGraph Inlining Suggester
//!
//! サブグラフ呼び出しをインライン展開するSuggester。
//! SubgraphCallノードを対応するサブグラフの計算ノードで置き換えます。

use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::{GraphSuggester, SuggestResult};
use std::collections::{HashMap, HashSet};

/// サブグラフ呼び出しをインライン展開するSuggester
///
/// SubgraphCallノードを見つけ、対応するサブグラフの計算を
/// 呼び出し元グラフに直接埋め込みます。
///
/// # インライン展開の流れ
///
/// 1. SubgraphCallノードを検出
/// 2. 対応するサブグラフを取得
/// 3. サブグラフの入力を実際の引数で置き換え
/// 4. サブグラフの出力を計算結果に接続
///
/// # 対応するケース
///
/// - 単一出力のサブグラフ: SubgraphCallノードを直接置き換え
/// - 複数出力のサブグラフ: SubgraphOutputノードも含めて置き換え
pub struct SubgraphInliningSuggester;

impl SubgraphInliningSuggester {
    /// 新しいSubgraphInliningSuggesterを作成
    pub fn new() -> Self {
        SubgraphInliningSuggester
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

    /// サブグラフをインライン展開する
    ///
    /// SubgraphCallノードの入力を使って、サブグラフの計算を再構築します。
    fn inline_subgraph(
        &self,
        call_node: &GraphNode,
        subgraph: &Graph,
    ) -> Option<HashMap<String, GraphNode>> {
        let GraphOp::SubgraphCall { name } = &call_node.op else {
            log::debug!("inline_subgraph: node is not SubgraphCall");
            return None;
        };

        log::debug!(
            "inline_subgraph: starting inline of '{}', call_node has {} src",
            name,
            call_node.src.len()
        );
        for (i, src) in call_node.src.iter().enumerate() {
            log::trace!("inline_subgraph: src[{}] = {:?}", i, src.op);
        }

        // サブグラフの入力と呼び出し時の引数をマッピング
        let input_metas = subgraph.input_metas();

        log::debug!(
            "inline_subgraph: subgraph input_metas = {:?}",
            input_metas.iter().map(|m| &m.name).collect::<Vec<_>>()
        );

        if input_metas.len() != call_node.src.len() {
            log::warn!(
                "SubgraphInlining: argument count mismatch. Expected {}, got {}",
                input_metas.len(),
                call_node.src.len()
            );
            return None;
        }

        // 入力Bufferノードから実際の引数へのマッピングを作成
        // input_metasの順序とcall_node.srcの順序は対応している
        let mut input_mapping: HashMap<String, GraphNode> = HashMap::new();
        for (i, meta) in input_metas.iter().enumerate() {
            input_mapping.insert(meta.name.clone(), call_node.src[i].clone());
        }

        // サブグラフの出力ノードを再構築
        let mut node_cache: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
        let mut output_nodes: HashMap<String, GraphNode> = HashMap::new();

        for (output_name, output_node) in subgraph.outputs() {
            log::trace!(
                "inline_subgraph: rebuilding output '{}' from {:?}",
                output_name,
                output_node.op
            );
            let rebuilt =
                self.rebuild_node_with_inputs(output_node, &input_mapping, &mut node_cache);
            log::trace!(
                "inline_subgraph: rebuilt output '{}' is {:?}, has {} srcs",
                output_name,
                rebuilt.op,
                rebuilt.src.len()
            );
            // デバッグ: 依存グラフを出力
            fn count_nodes(
                node: &GraphNode,
                visited: &mut std::collections::HashSet<*const GraphNodeData>,
            ) -> usize {
                let ptr = node.as_ptr();
                if visited.contains(&ptr) {
                    return 0;
                }
                visited.insert(ptr);
                let mut count = 1;
                for src in &node.src {
                    count += count_nodes(src, visited);
                }
                count
            }
            let mut visited = std::collections::HashSet::new();
            let node_count = count_nodes(&rebuilt, &mut visited);
            log::debug!(
                "inline_subgraph: inlined '{}' has {} nodes in dependency graph",
                output_name,
                node_count
            );
            output_nodes.insert(output_name.clone(), rebuilt);
        }

        Some(output_nodes)
    }

    /// 入力マッピングを使ってノードを再構築する
    fn rebuild_node_with_inputs(
        &self,
        node: &GraphNode,
        input_mapping: &HashMap<String, GraphNode>,
        cache: &mut HashMap<*const GraphNodeData, GraphNode>,
    ) -> GraphNode {
        let ptr = node.as_ptr();

        // キャッシュをチェック
        if let Some(cached) = cache.get(&ptr) {
            return cached.clone();
        }

        // Bufferノードの場合は入力マッピングをチェック
        if let GraphOp::Buffer { name } = &node.op
            && let Some(actual_input) = input_mapping.get(name)
        {
            cache.insert(ptr, actual_input.clone());
            return actual_input.clone();
        }

        // srcを再構築
        let new_src: Vec<GraphNode> = node
            .src
            .iter()
            .map(|src| self.rebuild_node_with_inputs(src, input_mapping, cache))
            .collect();

        // srcが変更されたかチェック
        let src_changed = new_src
            .iter()
            .zip(&node.src)
            .any(|(a, b)| a.as_ptr() != b.as_ptr());

        let new_node = if src_changed {
            GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
            )
        } else {
            node.clone()
        };

        cache.insert(ptr, new_node.clone());
        new_node
    }

    /// グラフ内のSubgraphCallノードを置き換えた新しいグラフを作成
    fn replace_subgraph_call(
        &self,
        graph: &Graph,
        call_node: &GraphNode,
        inlined_outputs: HashMap<String, GraphNode>,
    ) -> Graph {
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        // SubgraphCallノードの参照先を特定
        let call_ptr = call_node.as_ptr();

        // 単一出力の場合: SubgraphCallノードを直接出力ノードで置き換え
        // 複数出力の場合: SubgraphOutputノードで個別に置き換えるので、node_mapには入れない
        let single_output = if inlined_outputs.len() == 1 {
            let output_node = inlined_outputs.values().next().unwrap().clone();
            node_map.insert(call_ptr, output_node);
            true
        } else {
            false
        };

        // SubgraphOutputノードも処理
        // SubgraphOutputノードはSubgraphCallを参照しているので、
        // 対応する出力ノードで置き換える

        let mut cache: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            inlined_outputs: &HashMap<String, GraphNode>,
            call_ptr: *const GraphNodeData,
            cache: &mut HashMap<*const GraphNodeData, GraphNode>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // キャッシュをチェック（再構築済みノードを返す）
            if let Some(cached) = cache.get(&ptr) {
                log::trace!("rebuild_node: cache hit for {:?}", node.op);
                return cached.clone();
            }

            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                log::trace!(
                    "rebuild_node: replacing SubgraphCall with inlined output: {:?} -> {:?}",
                    node.op,
                    new_node.op
                );
                return new_node.clone();
            }

            // SubgraphOutputノードの場合
            if let GraphOp::SubgraphOutput { output_name, .. } = &node.op {
                // このSubgraphOutputが参照しているSubgraphCallを確認
                if !node.src.is_empty()
                    && node.src[0].as_ptr() == call_ptr
                    && let Some(inlined) = inlined_outputs.get(output_name)
                {
                    return inlined.clone();
                }
            }

            let new_src: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, inlined_outputs, call_ptr, cache))
                .collect();

            let src_changed = new_src
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            let result = if !src_changed {
                log::trace!("rebuild_node: no src change for {:?}", node.op);
                node.clone()
            } else {
                log::trace!(
                    "rebuild_node: src changed for {:?}, creating new node",
                    node.op
                );
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

        // メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);
        new_graph.copy_subgraphs_from(graph);

        // 複数出力の場合のみinlined_outputsを使用
        let outputs_ref = if single_output {
            &HashMap::new()
        } else {
            &inlined_outputs
        };

        // 出力ノードを再構築
        for (name, output_node) in graph.outputs() {
            let rebuilt = rebuild_node(output_node, &node_map, outputs_ref, call_ptr, &mut cache);
            new_graph.set_output_node(name.clone(), rebuilt);
        }

        new_graph
    }
}

impl Default for SubgraphInliningSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for SubgraphInliningSuggester {
    fn name(&self) -> &'static str {
        "SubgraphInlining"
    }

    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        log::debug!(
            "SubgraphInlining: collected {} nodes, graph has {} subgraphs",
            nodes.len(),
            graph.subgraphs().len()
        );

        // SubgraphCallノードを探す
        for node in &nodes {
            if let GraphOp::SubgraphCall { name } = &node.op {
                log::debug!("SubgraphInlining: found SubgraphCall '{}'", name);

                // 対応するサブグラフを取得
                let Some(subgraph) = graph.subgraph(name) else {
                    log::warn!("SubgraphInlining: subgraph '{}' not found in graph", name);
                    continue;
                };

                log::debug!(
                    "SubgraphInlining: found subgraph '{}' with {} inputs, {} outputs",
                    name,
                    subgraph.input_metas().len(),
                    subgraph.outputs().len()
                );

                // サブグラフをインライン展開
                if let Some(inlined_outputs) = self.inline_subgraph(node, subgraph) {
                    log::debug!(
                        "SubgraphInlining: successfully inlined '{}' with {} outputs",
                        name,
                        inlined_outputs.len()
                    );
                    let new_graph = self.replace_subgraph_call(graph, node, inlined_outputs);
                    suggestions.push(SuggestResult::new(new_graph, self.name()));
                    // 一度に1つのSubgraphCallのみを展開
                    // 複数のSubgraphCallがある場合は、次のイテレーションで処理される
                    break;
                } else {
                    log::warn!("SubgraphInlining: failed to inline '{}'", name);
                }
            }
        }

        log::debug!(
            "SubgraphInlining: returning {} suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    #[test]
    fn test_inline_simple_subgraph() {
        let suggester = SubgraphInliningSuggester::new();

        // サブグラフを作成: y = x * 2
        let mut subgraph = Graph::new();
        let x = subgraph.input("x", DType::F32, vec![4]);
        let two = GraphNode::constant(2.0f32);
        let y = x * two;
        subgraph.output("y", y);

        // メイングラフを作成
        let mut main_graph = Graph::new();
        let a = main_graph.input("a", DType::F32, vec![4]);

        // サブグラフを追加
        main_graph.add_subgraph("double", subgraph);

        // SubgraphCallノードを作成
        let call_node = GraphNode::subgraph_call(
            "double",
            vec![a.clone()],
            DType::F32,
            crate::graph::View::contiguous(vec![4]),
        );
        main_graph.output("result", call_node);

        // サジェストを取得
        let suggestions = suggester.suggest(&main_graph);

        // 1つの候補が生成されるはず
        assert_eq!(suggestions.len(), 1);

        // インライン展開後のグラフにSubgraphCallがないことを確認
        let inlined = &suggestions[0].graph;
        let nodes = suggester.collect_all_nodes(inlined);
        let has_subgraph_call = nodes
            .iter()
            .any(|n| matches!(n.op, GraphOp::SubgraphCall { .. }));
        assert!(!has_subgraph_call, "SubgraphCall should be inlined");
    }

    #[test]
    fn test_no_suggestion_without_subgraph_call() {
        let suggester = SubgraphInliningSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4]);
        let b = graph.input("b", DType::F32, vec![4]);
        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // SubgraphCallがないので候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_collect_nodes_finds_subgraph_calls() {
        let suggester = SubgraphInliningSuggester::new();

        // サブグラフを作成: y = x * 2
        let mut subgraph = Graph::new();
        let x = subgraph.input("x", DType::F32, vec![4]);
        let two = GraphNode::constant(2.0f32);
        let y = x * two;
        subgraph.output("y", y);

        // メイングラフを作成
        let mut main_graph = Graph::new();
        let a = main_graph.input("a", DType::F32, vec![4]);

        // サブグラフを追加
        main_graph.add_subgraph("double", subgraph);

        // SubgraphCallノードを作成
        let call_node = GraphNode::subgraph_call(
            "double",
            vec![a.clone()],
            DType::F32,
            crate::graph::View::contiguous(vec![4]),
        );
        main_graph.output("result", call_node);

        // ノードを収集
        let nodes = suggester.collect_all_nodes(&main_graph);

        // SubgraphCallノードが見つかることを確認
        let subgraph_call_count = nodes
            .iter()
            .filter(|n| matches!(n.op, GraphOp::SubgraphCall { .. }))
            .count();
        assert!(
            subgraph_call_count > 0,
            "Should find SubgraphCall nodes in the graph"
        );
    }

    #[test]
    fn test_inline_subgraph_method() {
        let suggester = SubgraphInliningSuggester::new();

        // サブグラフを作成: y = x * 2
        let mut subgraph = Graph::new();
        let x = subgraph.input("x", DType::F32, vec![4]);
        let two = GraphNode::constant(2.0f32);
        let y = x * two;
        subgraph.output("y", y);

        // 入力ノード
        let a = GraphNode::new(
            DType::F32,
            GraphOp::Buffer {
                name: "a".to_string(),
            },
            vec![],
            crate::graph::View::contiguous(vec![4]),
        );

        // SubgraphCallノードを作成
        let call_node = GraphNode::subgraph_call(
            "double",
            vec![a.clone()],
            DType::F32,
            crate::graph::View::contiguous(vec![4]),
        );

        // インライン展開を実行
        let result = suggester.inline_subgraph(&call_node, &subgraph);

        // 結果があることを確認
        assert!(result.is_some(), "inline_subgraph should return Some");

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1, "Should have 1 output");
        assert!(outputs.contains_key("y"), "Output should be named 'y'");

        // インライン展開された出力はElementwise(Mul)であるべき
        let output_node = &outputs["y"];
        assert!(
            matches!(output_node.op, GraphOp::Elementwise { .. }),
            "Inlined output should be Elementwise, got {:?}",
            output_node.op
        );
    }
}
