use crate::graph::ops::{FusedElementwiseOp, FusedInput};
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// ノード融合を提案するSuggester
pub struct FusionSuggester;

impl FusionSuggester {
    /// 新しいFusionSuggesterを作成
    pub fn new() -> Self {
        Self
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

    /// Elementwise演算チェーンを検出して融合可能な場合にパターンを返す
    ///
    /// 連続する2つのElementwise演算を融合する
    /// 例: (a + b) * c -> FusedElementwise([a, b, c], [Add, Mul])
    #[allow(dead_code)]
    fn detect_elementwise_chain(
        &self,
        node: &GraphNode,
    ) -> Option<(Vec<GraphNode>, Vec<FusedElementwiseOp>)> {
        // このノードがElementwiseでない場合はNone
        let current_op = match &node.op {
            GraphOp::Elementwise { op, .. } => op.clone(),
            _ => return None,
        };

        // 入力のうち、少なくとも1つがElementwiseの場合に融合可能
        let mut found_elementwise_input = false;
        for src in &node.src {
            if matches!(src.op, GraphOp::Elementwise { .. }) {
                found_elementwise_input = true;
                break;
            }
        }

        if !found_elementwise_input {
            return None;
        }

        // 融合するノードチェーンを構築
        let mut graph_inputs = Vec::new();
        let mut ops = Vec::new();
        let mut input_mapping: HashMap<*const GraphNodeData, usize> = HashMap::new();

        // 入力ノードを処理（Elementwise入力は展開、それ以外はgraph_inputsに追加）
        let mut intermediate_inputs = Vec::new();

        for src in &node.src {
            match &src.op {
                GraphOp::Elementwise { op, .. } => {
                    // Elementwiseノードの場合、その入力を展開
                    for sub_src in &src.src {
                        let ptr = sub_src.as_ptr();
                        let idx = if let Some(&existing_idx) = input_mapping.get(&ptr) {
                            existing_idx
                        } else {
                            let new_idx = graph_inputs.len();
                            input_mapping.insert(ptr, new_idx);
                            graph_inputs.push(sub_src.clone());
                            new_idx
                        };
                        intermediate_inputs.push(FusedInput::GraphInput(idx));
                    }

                    // 中間演算を追加
                    let num_inputs = src.src.len();
                    let op_inputs =
                        intermediate_inputs.split_off(intermediate_inputs.len() - num_inputs);
                    ops.push(FusedElementwiseOp {
                        op: op.clone(),
                        inputs: op_inputs,
                    });
                }
                _ => {
                    // 通常の入力の場合
                    let ptr = src.as_ptr();
                    let idx = if let Some(&existing_idx) = input_mapping.get(&ptr) {
                        existing_idx
                    } else {
                        let new_idx = graph_inputs.len();
                        input_mapping.insert(ptr, new_idx);
                        graph_inputs.push(src.clone());
                        new_idx
                    };
                    intermediate_inputs.push(FusedInput::GraphInput(idx));
                }
            }
        }

        // 現在のノードの演算を追加
        ops.push(FusedElementwiseOp {
            op: current_op,
            inputs: intermediate_inputs,
        });

        // 2つ以上の演算がある場合のみ融合
        if ops.len() < 2 {
            return None;
        }

        Some((graph_inputs, ops))
    }

    /// Elementwise -> Reduceパターンを検出
    fn detect_elementwise_reduce_pattern(
        &self,
        node: &GraphNode,
    ) -> Option<(
        Vec<GraphNode>,
        Vec<FusedElementwiseOp>,
        crate::graph::ops::ReduceOp,
        usize,
    )> {
        // このノードがReduceでない場合はNone
        let (reduce_op, axis) = match &node.op {
            GraphOp::Reduce { op, axis, .. } => (op.clone(), *axis),
            _ => return None,
        };

        // 入力がElementwiseの場合、融合可能
        if node.src.len() != 1 {
            return None;
        }

        let input = &node.src[0];
        let elementwise_op = match &input.op {
            GraphOp::Elementwise { op, .. } => op.clone(),
            _ => return None,
        };

        // Elementwise入力のさらに入力をgraph_inputsとして収集
        let graph_inputs = input.src.clone();

        // FusedElementwiseOpを作成
        let inputs: Vec<FusedInput> = (0..graph_inputs.len())
            .map(FusedInput::GraphInput)
            .collect();

        let ops = vec![FusedElementwiseOp {
            op: elementwise_op,
            inputs,
        }];

        Some((graph_inputs, ops, reduce_op, axis))
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

        // 出力ノードを再構築
        for (name, output_node) in graph.outputs() {
            let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
            new_graph.output(name, rebuilt);
        }

        new_graph
    }
}

impl Default for FusionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for FusionSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        for node in &nodes {
            // Elementwise -> Reduceパターンを検出して融合
            if let Some((graph_inputs, ops, reduce_op, axis)) =
                self.detect_elementwise_reduce_pattern(node)
            {
                // FusedElementwiseReduceノードを作成
                let fused_node =
                    crate::graph::ops::fused_elementwise_reduce(graph_inputs, ops, reduce_op, axis);

                let new_graph = self.replace_node_in_graph(graph, node, fused_node);
                suggestions.push(new_graph);
            }

            // Elementwise チェーンを検出して融合
            if let Some((graph_inputs, ops)) = self.detect_elementwise_chain(node) {
                // FusedElementwiseノードを作成
                let fused_node = crate::graph::ops::fused_elementwise(graph_inputs, ops);

                let new_graph = self.replace_node_in_graph(graph, node, fused_node);
                suggestions.push(new_graph);
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
    fn test_fusion_suggester() {
        let suggester = FusionSuggester::new();

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

        // (a + b).reduce_sum(0) -> FusedElementwiseReduce
        let c = (a + b).reduce_sum(0);
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // Elementwise -> Reduceパターンが検出され、1つの候補が生成されるはず
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_fusion_suggester_no_pattern() {
        let suggester = FusionSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        // 単純な入力のみ、融合パターンなし
        graph.output("a", a);

        let suggestions = suggester.suggest(&graph);

        // パターンが検出されないので、候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_fusion_elementwise_chain() {
        let suggester = FusionSuggester::new();

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

        let c = graph
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // (a + b) * c -> Elementwiseチェーンを融合可能
        let add = a + b;
        let mul = add * c;
        graph.output("result", mul);

        let suggestions = suggester.suggest(&graph);

        // Elementwiseチェーンが検出され、融合候補が生成されるはず
        assert!(
            !suggestions.is_empty(),
            "Expected fusion suggestions for elementwise chain"
        );

        // 融合された結果がFusedElementwiseになっているか確認
        let fused_graph = &suggestions[0];
        let output_node = fused_graph.outputs().get("result").unwrap();

        // 出力ノードがFusedElementwiseであることを確認
        assert!(
            matches!(output_node.op, GraphOp::FusedElementwise { .. }),
            "Expected FusedElementwise operation"
        );
    }

    #[test]
    fn test_fusion_complex_chain() {
        let suggester = FusionSuggester::new();

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

        let c = graph
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let d = graph
            .input("d")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // ((a + b) * c) - d -> 3つのElementwise演算チェーン
        let add = a + b;
        let mul = add * c;
        let sub = mul - d;
        graph.output("result", sub);

        let suggestions = suggester.suggest(&graph);

        // 複数の融合候補が生成される可能性がある
        assert!(
            !suggestions.is_empty(),
            "Expected fusion suggestions for complex chain"
        );
    }
}
