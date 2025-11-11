//! グラフの可視化機能
//!
//! Graphviz DOT形式でのグラフ可視化をサポートします。

use super::{Graph, GraphNode, GraphNodeData};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

impl Graph {
    /// Graphviz DOT形式でグラフを出力
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph G {\n");
        dot.push_str("  rankdir=LR;\n"); // 左から右へのレイアウト
        dot.push_str("  node [shape=box];\n\n");

        let mut visited = HashSet::new();
        let mut node_counter = 0;
        let mut node_ids = HashMap::new();

        // 全ノードを収集してDOT形式に変換
        fn traverse_and_collect(
            node: &GraphNode,
            visited: &mut HashSet<*const GraphNodeData>,
            dot: &mut String,
            node_ids: &mut HashMap<*const GraphNodeData, usize>,
            counter: &mut usize,
        ) -> usize {
            let node_ptr = Rc::as_ptr(&node.0);

            if visited.contains(&node_ptr) {
                return *node_ids.get(&node_ptr).unwrap();
            }
            visited.insert(node_ptr);

            // 入力ノードを先に処理
            for input in &node.src {
                traverse_and_collect(input, visited, dot, node_ids, counter);
            }

            // このノードのIDを取得
            let node_id = if let Some(&id) = node_ids.get(&node_ptr) {
                id
            } else {
                let id = *counter;
                *counter += 1;
                node_ids.insert(node_ptr, id);
                id
            };

            // ノードのラベルを作成
            let op_str = format!("{:?}", node.op);
            let op_str = if op_str.len() > 50 {
                format!("{}...", &op_str[..50])
            } else {
                op_str
            };
            let dtype_str = format!("{:?}", node.dtype);
            let shape_str = node
                .view
                .shape()
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join(", ");

            let label = format!(
                "Node {}\\n{}\\nDType: {}\\nShape: [{}]",
                node_id, op_str, dtype_str, shape_str
            );

            // ノード定義を追加
            dot.push_str(&format!("  n{} [label=\"{}\"];\n", node_id, label));

            // エッジを追加
            for (i, input) in node.src.iter().enumerate() {
                let input_id = *node_ids.get(&Rc::as_ptr(&input.0)).unwrap();
                dot.push_str(&format!(
                    "  n{} -> n{} [label=\"input {}\"];\n",
                    input_id, node_id, i
                ));
            }

            node_id
        }

        // 出力ノードを名前順でソートして処理（重複除去のため順序を固定）
        let mut outputs: Vec<_> = self.outputs.iter().collect();
        outputs.sort_by_key(|(name, _)| name.as_str());

        for (output_name, output_node) in outputs {
            let output_id = traverse_and_collect(
                output_node,
                &mut visited,
                &mut dot,
                &mut node_ids,
                &mut node_counter,
            );

            // 出力ノードにマーク
            dot.push_str(&format!(
                "  output_{} [label=\"Output: {}\", shape=ellipse, style=filled, fillcolor=lightblue];\n",
                output_name, output_name
            ));
            dot.push_str(&format!("  n{} -> output_{};\n", output_id, output_name));
        }

        dot.push_str("}\n");
        dot
    }

    /// DOT形式でファイルに保存
    pub fn save_dot(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_dot())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    #[test]
    fn test_to_dot_output_order_independence() {
        // 出力ノードの順序が異なっても同じDOT文字列が生成されることを確認

        // グラフ1: 出力順序 "a", "b"
        let mut graph1 = Graph::new();
        let input1 = graph1
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        graph1.output("a", input1.clone());
        graph1.output("b", input1.clone());

        // グラフ2: 出力順序 "b", "a"
        let mut graph2 = Graph::new();
        let input2 = graph2
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        graph2.output("b", input2.clone());
        graph2.output("a", input2.clone());

        // 両方のグラフは同じDOT文字列を生成すべき
        let dot1 = graph1.to_dot();
        let dot2 = graph2.to_dot();

        assert_eq!(
            dot1, dot2,
            "Graphs with different output order should produce the same DOT string"
        );
    }

    #[test]
    fn test_to_dot_multiple_outputs_sorted() {
        // 複数の出力ノードが名前順でソートされることを確認
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        // 意図的に非アルファベット順で追加
        graph.output("zebra", input.clone());
        graph.output("alpha", input.clone());
        graph.output("middle", input.clone());

        let dot = graph.to_dot();

        // "alpha"が"middle"より前に、"middle"が"zebra"より前に現れることを確認
        let alpha_pos = dot.find("output_alpha").expect("alpha output not found");
        let middle_pos = dot.find("output_middle").expect("middle output not found");
        let zebra_pos = dot.find("output_zebra").expect("zebra output not found");

        assert!(
            alpha_pos < middle_pos && middle_pos < zebra_pos,
            "Outputs should appear in alphabetical order in DOT string"
        );
    }
}
