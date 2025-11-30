//! Buffer Absorption Suggester
//!
//! CustomノードのsrcにあるBufferノードを取り込み、
//! `input_buffers`フィールドに設定するSuggester。
//!
//! # 処理フロー
//! 1. srcにBufferノードを持つCustomノードを検出
//! 2. Bufferノードの情報を`input_buffers`に取り込む
//! 3. srcからBufferノードを削除（srcは空になる）
//!
//! # ProgramRootAbsorptionSuggesterとの連携
//! このSuggesterの適用後、CustomノードはsrcにBufferを持たなくなるため、
//! ProgramRootAbsorptionSuggesterはシンプルにCustomを吸収するだけでよくなる。

use crate::graph::ops::InputBufferMeta;
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// BufferノードをCustomノードに取り込むSuggester
pub struct BufferAbsorptionSuggester;

impl BufferAbsorptionSuggester {
    pub fn new() -> Self {
        Self
    }

    /// srcにBufferを持つCustomノードを検出
    fn find_custom_with_buffers(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        if let Some(sink) = graph.sink() {
            self.find_customs_recursive(sink, &mut result, &mut visited);
        }

        result
    }

    fn find_customs_recursive(
        &self,
        node: &GraphNode,
        result: &mut Vec<GraphNode>,
        visited: &mut HashSet<*const GraphNodeData>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // 子ノードを先に探索
        for src in &node.src {
            self.find_customs_recursive(src, result, visited);
        }

        // input_buffersがNoneで、srcにBufferを持つCustomノードを検出
        if let GraphOp::Kernel {
            input_buffers: None,
            ..
        } = &node.op
        {
            let has_buffer_in_src = node
                .src
                .iter()
                .any(|s| matches!(s.op, GraphOp::Buffer { .. }));
            if has_buffer_in_src {
                result.push(node.clone());
            }
        }
    }

    /// CustomノードにBufferを取り込む
    fn absorb_buffers(&self, graph: &Graph, custom_node: &GraphNode) -> Option<Graph> {
        // CustomノードのASTを取得
        let ast = match &custom_node.op {
            GraphOp::Kernel {
                ast,
                input_buffers: None,
            } => ast.clone(),
            _ => return None,
        };

        // srcからBufferノードの情報を収集
        let mut input_buffers = Vec::new();
        let mut new_src = Vec::new();

        for src in &custom_node.src {
            match &src.op {
                GraphOp::Buffer { name } => {
                    // 出力バッファ（output_で始まる）はsrcに残す
                    if name.starts_with("output_") {
                        new_src.push(src.clone());
                    } else {
                        input_buffers.push(InputBufferMeta {
                            name: name.clone(),
                            dtype: src.dtype.clone(),
                            shape: src.view.shape().to_vec(),
                        });
                    }
                }
                _ => {
                    // Buffer以外のノード（他のCustomノードなど）はsrcに残す
                    new_src.push(src.clone());
                }
            }
        }

        // input_buffersが空なら変換不要
        if input_buffers.is_empty() {
            return None;
        }

        // 新しいCustomノードを作成
        let new_custom = GraphNode::new(
            custom_node.dtype.clone(),
            GraphOp::Kernel {
                ast,
                input_buffers: Some(input_buffers),
            },
            new_src,
            custom_node.view.clone(),
        );

        // グラフを再構築
        Some(self.replace_node_in_graph(graph, custom_node, new_custom))
    }

    /// グラフ内の特定ノードを置き換えた新しいグラフを作成
    fn replace_node_in_graph(
        &self,
        graph: &Graph,
        old_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut new_graph = Graph::new();
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        // メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // ノードを再構築
        fn rebuild_node(
            node: &GraphNode,
            old_ptr: *const GraphNodeData,
            new_node: &GraphNode,
            node_map: &mut HashMap<*const GraphNodeData, GraphNode>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            if let Some(mapped) = node_map.get(&ptr) {
                return mapped.clone();
            }

            if ptr == old_ptr {
                node_map.insert(ptr, new_node.clone());
                return new_node.clone();
            }

            let new_children: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, old_ptr, new_node, node_map))
                .collect();

            let children_changed = new_children
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            let result = if children_changed {
                GraphNode::new(
                    node.dtype.clone(),
                    node.op.clone(),
                    new_children,
                    node.view.clone(),
                )
            } else {
                node.clone()
            };

            node_map.insert(ptr, result.clone());
            result
        }

        let old_ptr = old_node.as_ptr();

        // Sinkノードを再構築
        if let Some(sink) = graph.sink() {
            let new_sink = rebuild_node(sink, old_ptr, &new_node, &mut node_map);
            new_graph.set_sink(new_sink);
        }

        new_graph
    }
}

impl Default for BufferAbsorptionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for BufferAbsorptionSuggester {
    fn name(&self) -> &'static str {
        "BufferAbsorptionSuggester"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let custom_nodes = self.find_custom_with_buffers(graph);

        log::debug!(
            "BufferAbsorption: found {} Custom nodes with buffers to absorb",
            custom_nodes.len()
        );

        let mut suggestions = Vec::new();

        // 各Customノードに対して1つの提案を生成
        for custom_node in &custom_nodes {
            if let Some(new_graph) = self.absorb_buffers(graph, custom_node) {
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
    fn test_buffer_absorption_basic() {
        use crate::opt::graph::suggesters::LoweringSuggester;

        let lowering = LoweringSuggester::new();
        let buffer_absorber = BufferAbsorptionSuggester::new();

        // シンプルなElementwise演算グラフ
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = a + b;
        graph.output("c", c);

        eprintln!("=== Initial Graph ===");
        eprintln!("Sink exists: {:?}", graph.sink().is_some());

        // Loweringを適用
        let lowered = lowering.suggest(&graph);
        assert!(!lowered.is_empty());
        let lowered_graph = &lowered[0];

        eprintln!("\n=== After Lowering ===");
        if let Some(ref sink) = lowered_graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let has_buffer = src
                    .src
                    .iter()
                    .any(|s| matches!(s.op, GraphOp::Buffer { .. }));
                eprintln!("  src[{}]: has_buffer={}", i, has_buffer);
            }
        }

        // BufferAbsorptionを適用
        let absorbed = buffer_absorber.suggest(lowered_graph);
        eprintln!("\n=== After BufferAbsorption ===");
        eprintln!("Got {} suggestions", absorbed.len());

        assert!(!absorbed.is_empty());
        let absorbed_graph = &absorbed[0];

        // 検証: Customノードのinput_buffersが設定されている
        if let Some(ref sink) = absorbed_graph.sink() {
            for src in &sink.src {
                if let GraphOp::Kernel { input_buffers, .. } = &src.op {
                    eprintln!("Custom node input_buffers: {:?}", input_buffers);
                    assert!(input_buffers.is_some());
                    let buffers = input_buffers.as_ref().unwrap();
                    assert_eq!(buffers.len(), 2); // a と b
                    assert!(buffers.iter().any(|b| b.name == "a"));
                    assert!(buffers.iter().any(|b| b.name == "b"));
                }
            }
        }
    }

    #[test]
    fn test_buffer_absorption_preserves_input_metas() {
        use crate::opt::graph::suggesters::LoweringSuggester;

        let lowering = LoweringSuggester::new();
        let buffer_absorber = BufferAbsorptionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = a + b;
        graph.output("c", c);

        let lowered = lowering.suggest(&graph);
        let lowered_graph = &lowered[0];

        let absorbed = buffer_absorber.suggest(lowered_graph);
        let absorbed_graph = &absorbed[0];

        // 入力メタデータが保持されていることを確認
        assert_eq!(
            absorbed_graph.input_metas().len(),
            2,
            "Input metas should be preserved"
        );
        assert!(absorbed_graph.input_metas().iter().any(|m| m.name == "a"));
        assert!(absorbed_graph.input_metas().iter().any(|m| m.name == "b"));
    }
}
