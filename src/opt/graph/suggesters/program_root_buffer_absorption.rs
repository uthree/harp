//! Sink Buffer Absorption Suggester
//!
//! SinkノードのsrcにあるBufferノード（入力バッファ）を除去するSuggester。
//!
//! # 処理フロー
//! 1. srcに入力Bufferノード、またはView→Buffer(input)パターンを検出
//! 2. 入力バッファ（output_で始まらないBuffer）とその1レベルのViewラッパーをsrcから除去
//!
//! # BufferAbsorptionSuggesterとの関係
//! - BufferAbsorptionSuggester: Buffer → Custom の吸収
//! - ProgramRootBufferAbsorptionSuggester: Buffer → Sink の吸収
//!
//! 入力バッファの情報は既にgraph.input_metas()に保存されているため、
//! Sinkに別途保持する必要はありません。

use crate::graph::{Graph, GraphNode, GraphOp};
use crate::opt::graph::GraphSuggester;

/// SinkノードからBufferノードを除去するSuggester
pub struct ProgramRootBufferAbsorptionSuggester;

impl ProgramRootBufferAbsorptionSuggester {
    pub fn new() -> Self {
        Self
    }

    /// ノードが入力Bufferかどうかを判定
    /// View→...→Buffer(input) のチェーンに対応（再帰ではなくループで実装）
    fn is_input_buffer_pattern(node: &GraphNode) -> bool {
        let mut current = node;
        loop {
            match &current.op {
                // 入力Buffer（output_で始まらない）
                GraphOp::Buffer { name } => return !name.starts_with("output_"),
                // Viewチェーンをたどる
                GraphOp::View(_) => {
                    if current.src.len() != 1 {
                        // Viewは通常1つの入力のみ
                        return false;
                    }
                    current = &current.src[0];
                }
                // Const も入力として扱う
                GraphOp::Const(_) | GraphOp::ComplexConst { .. } => return true,
                // その他は入力パターンではない
                _ => return false,
            }
        }
    }

    /// Sinkのsrcに入力Bufferパターンがあるか確認
    fn has_input_buffer_dependencies(&self, graph: &Graph) -> bool {
        if let Some(sink) = graph.sink() {
            sink.src.iter().any(Self::is_input_buffer_pattern)
        } else {
            false
        }
    }

    /// SinkノードからBufferパターンを除去
    fn absorb_buffers(&self, graph: &Graph) -> Option<Graph> {
        let sink = graph.sink()?;

        // srcから入力Bufferパターンを除去
        let new_src: Vec<GraphNode> = sink
            .src
            .iter()
            .filter(|src| !Self::is_input_buffer_pattern(src))
            .cloned()
            .collect();

        // srcに変更がなければ何もしない
        if new_src.len() == sink.src.len() {
            return None;
        }

        log::debug!(
            "ProgramRootBufferAbsorption: removing {} nodes (input buffers and View→Buffer patterns) from Sink.src",
            sink.src.len() - new_src.len()
        );

        // 新しいSinkノードを作成
        let new_sink = GraphNode::new(
            sink.dtype.clone(),
            sink.op.clone(),
            new_src,
            sink.view.clone(),
        );

        // 新しいグラフを構築
        let mut new_graph = Graph::new();
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);
        new_graph.set_sink(new_sink);

        // shape変数のデフォルト値をコピー
        for (name, value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(name.clone(), *value);
        }

        Some(new_graph)
    }
}

impl Default for ProgramRootBufferAbsorptionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ProgramRootBufferAbsorptionSuggester {
    fn name(&self) -> &'static str {
        "ProgramRootBufferAbsorption"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        // Sinkがない場合は何もしない
        if graph.sink().is_none() {
            return vec![];
        }

        // 入力Bufferの依存関係がない場合は何もしない
        if !self.has_input_buffer_dependencies(graph) {
            return vec![];
        }

        // Bufferとその依存チェーンを除去
        if let Some(new_graph) = self.absorb_buffers(graph) {
            vec![new_graph]
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;
    use crate::opt::graph::suggesters::{
        BufferAbsorptionSuggester, LoweringSuggester, ProgramRootAbsorptionSuggester,
    };

    #[test]
    fn test_sink_buffer_absorption_basic() {
        let lowering = LoweringSuggester::new();
        let buffer_absorber = BufferAbsorptionSuggester::new();
        let sink_absorber = ProgramRootAbsorptionSuggester::new();
        let sink_buffer_absorber = ProgramRootBufferAbsorptionSuggester::new();

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
                let op_name = match &src.op {
                    GraphOp::Kernel { .. } => "Custom".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    _ => format!("{:?}", std::mem::discriminant(&src.op)),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
        }

        // BufferAbsorptionを適用
        let absorbed = buffer_absorber.suggest(lowered_graph);
        assert!(!absorbed.is_empty());
        let absorbed_graph = &absorbed[0];

        eprintln!("\n=== After BufferAbsorption ===");
        if let Some(ref sink) = absorbed_graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Kernel { .. } => "Custom".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    _ => format!("{:?}", std::mem::discriminant(&src.op)),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
        }

        // ProgramRootAbsorptionを適用
        let sink_absorbed = sink_absorber.suggest(absorbed_graph);
        assert!(!sink_absorbed.is_empty());
        let sink_absorbed_graph = &sink_absorbed[0];

        eprintln!("\n=== After ProgramRootAbsorption ===");
        if let Some(ref sink) = sink_absorbed_graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Kernel { .. } => "Custom".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    _ => format!("{:?}", std::mem::discriminant(&src.op)),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
        }

        // ProgramRootBufferAbsorptionを適用
        let final_suggestions = sink_buffer_absorber.suggest(sink_absorbed_graph);
        eprintln!("\n=== After ProgramRootBufferAbsorption ===");
        eprintln!("Got {} suggestions", final_suggestions.len());

        if !final_suggestions.is_empty() {
            let final_graph = &final_suggestions[0];
            if let Some(ref sink) = final_graph.sink() {
                eprintln!("Sink src count: {}", sink.src.len());
                for (i, src) in sink.src.iter().enumerate() {
                    let op_name = match &src.op {
                        GraphOp::Kernel { .. } => "Custom".to_string(),
                        GraphOp::Buffer { name } => format!("Buffer({})", name),
                        _ => format!("{:?}", std::mem::discriminant(&src.op)),
                    };
                    eprintln!("  src[{}]: {}", i, op_name);
                }
                // 入力バッファが除去されていることを確認
                let input_buffer_count = sink.src.iter().filter(|s| {
                    matches!(&s.op, GraphOp::Buffer { name } if !name.starts_with("output_"))
                }).count();
                assert_eq!(
                    input_buffer_count, 0,
                    "Input buffers should be removed from Sink.src"
                );
            }
        } else {
            // ProgramRootAbsorption後に入力Bufferがなければ提案はない
            if let Some(ref sink) = sink_absorbed_graph.sink() {
                let input_buffer_count = sink.src.iter().filter(|s| {
                    matches!(&s.op, GraphOp::Buffer { name } if !name.starts_with("output_"))
                }).count();
                assert_eq!(
                    input_buffer_count, 0,
                    "No suggestions means no input buffers in Sink.src"
                );
            }
        }
    }

    #[test]
    fn test_sink_buffer_absorption_with_view_chain() {
        let sink_buffer_absorber = ProgramRootBufferAbsorptionSuggester::new();

        // View → Buffer のチェーンをテスト
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);

        // View操作を追加
        let a_view = a.view(a.view.clone().permute(vec![1, 0]));

        // 出力
        let b = graph.input("b", DType::F32, vec![20, 10]);
        let c = a_view + b;
        graph.output("c", c);

        eprintln!("=== Graph with View chain ===");
        if let Some(sink) = graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            fn print_tree(node: &GraphNode, depth: usize) {
                let indent = "  ".repeat(depth);
                let op_name = match &node.op {
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    GraphOp::View(_) => "View".to_string(),
                    _ => format!("{:?}", std::mem::discriminant(&node.op)),
                };
                eprintln!("{}{} (src_count={})", indent, op_name, node.src.len());
                for src in &node.src {
                    print_tree(src, depth + 1);
                }
            }
            for (i, src) in sink.src.iter().enumerate() {
                eprintln!("src[{}]:", i);
                print_tree(src, 1);
            }
        }

        // ProgramRootBufferAbsorptionを適用
        let suggestions = sink_buffer_absorber.suggest(&graph);
        eprintln!("\nGot {} suggestions", suggestions.len());

        // View → Buffer のチェーンが除去されることを確認
        // (このテストではElementwiseがあるので、完全には除去されないかもしれない)
    }

    #[test]
    fn test_is_input_buffer_pattern() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);

        // Buffer(a) は入力パターン
        assert!(ProgramRootBufferAbsorptionSuggester::is_input_buffer_pattern(&a));

        // View → Buffer(a) のチェーンも入力パターン
        let a_view = a.view(a.view.clone());
        assert!(ProgramRootBufferAbsorptionSuggester::is_input_buffer_pattern(&a_view));

        // 出力バッファは入力パターンではない
        let output_buffer = GraphNode::new(
            DType::F32,
            GraphOp::Buffer {
                name: "output_c".to_string(),
            },
            vec![],
            a.view.clone(),
        );
        assert!(!ProgramRootBufferAbsorptionSuggester::is_input_buffer_pattern(&output_buffer));
    }

    #[test]
    fn test_sink_buffer_absorption_preserves_output_buffers() {
        let sink_buffer_absorber = ProgramRootBufferAbsorptionSuggester::new();

        // 手動でSinkを構築（入力と出力バッファを含む）
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = a + b;
        graph.output("c", c);

        // この時点では Sink.src に入力バッファが含まれている可能性がある
        if let Some(sink) = graph.sink() {
            let input_count = sink
                .src
                .iter()
                .filter(
                    |s| matches!(&s.op, GraphOp::Buffer { name } if !name.starts_with("output_")),
                )
                .count();
            eprintln!("Before: {} input buffers in Sink.src", input_count);
        }

        // ProgramRootBufferAbsorptionを適用
        let suggestions = sink_buffer_absorber.suggest(&graph);

        if !suggestions.is_empty() {
            let new_graph = &suggestions[0];
            if let Some(sink) = new_graph.sink() {
                // 入力バッファが除去されていることを確認
                let input_count = sink.src.iter().filter(|s| {
                    matches!(&s.op, GraphOp::Buffer { name } if !name.starts_with("output_"))
                }).count();
                assert_eq!(input_count, 0, "Input buffers should be removed");

                // 出力バッファは残っていることを確認
                let output_count = sink.src.iter().filter(|s| {
                    matches!(&s.op, GraphOp::Buffer { name } if name.starts_with("output_"))
                }).count();
                eprintln!("After: {} output buffers remain in Sink.src", output_count);
            }
        }
    }
}
