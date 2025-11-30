//! Sink Buffer Absorption Suggester
//!
//! SinkノードのsrcにあるBufferノード（入力バッファ）を除去するSuggester。
//!
//! # 処理フロー
//! 1. srcにBufferノードを持つSinkノードを検出
//! 2. 入力バッファ（output_で始まらないBuffer）をsrcから除去
//!
//! # BufferAbsorptionSuggesterとの関係
//! - BufferAbsorptionSuggester: Buffer → Custom の吸収
//! - SinkBufferAbsorptionSuggester: Buffer → Sink の吸収
//!
//! 入力バッファの情報は既にgraph.input_metas()に保存されているため、
//! Sinkに別途保持する必要はありません。

use crate::graph::{Graph, GraphNode, GraphOp};
use crate::opt::graph::GraphSuggester;

/// SinkノードからBufferノードを除去するSuggester
pub struct SinkBufferAbsorptionSuggester;

impl SinkBufferAbsorptionSuggester {
    pub fn new() -> Self {
        Self
    }

    /// Sinkのsrcに入力Bufferがあるか確認
    fn has_input_buffers_in_sink(&self, graph: &Graph) -> bool {
        if let Some(sink) = graph.sink() {
            sink.src.iter().any(
                |src| matches!(&src.op, GraphOp::Buffer { name } if !name.starts_with("output_")),
            )
        } else {
            false
        }
    }

    /// SinkノードからBufferを除去
    fn absorb_buffers(&self, graph: &Graph) -> Option<Graph> {
        let sink = graph.sink()?;

        // srcから入力Bufferを除去
        let new_src: Vec<GraphNode> = sink
            .src
            .iter()
            .filter(|src| {
                // 出力バッファ（output_で始まる）は残す
                // 入力バッファ（それ以外のBuffer）は除去
                !matches!(&src.op, GraphOp::Buffer { name } if !name.starts_with("output_"))
            })
            .cloned()
            .collect();

        // srcに変更がなければ何もしない
        if new_src.len() == sink.src.len() {
            return None;
        }

        log::debug!(
            "SinkBufferAbsorption: removing {} input buffers from Sink.src",
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

impl Default for SinkBufferAbsorptionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for SinkBufferAbsorptionSuggester {
    fn name(&self) -> &'static str {
        "SinkBufferAbsorption"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        // Sinkがない場合は何もしない
        if graph.sink().is_none() {
            return vec![];
        }

        // 入力Bufferがない場合は何もしない
        if !self.has_input_buffers_in_sink(graph) {
            return vec![];
        }

        // Bufferを除去
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
        BufferAbsorptionSuggester, LoweringSuggester, SinkAbsorptionSuggester,
    };

    #[test]
    fn test_sink_buffer_absorption_basic() {
        let lowering = LoweringSuggester::new();
        let buffer_absorber = BufferAbsorptionSuggester::new();
        let sink_absorber = SinkAbsorptionSuggester::new();
        let sink_buffer_absorber = SinkBufferAbsorptionSuggester::new();

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
                    GraphOp::Custom { .. } => "Custom".to_string(),
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
                    GraphOp::Custom { .. } => "Custom".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    _ => format!("{:?}", std::mem::discriminant(&src.op)),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
        }

        // SinkAbsorptionを適用
        let sink_absorbed = sink_absorber.suggest(absorbed_graph);
        assert!(!sink_absorbed.is_empty());
        let sink_absorbed_graph = &sink_absorbed[0];

        eprintln!("\n=== After SinkAbsorption ===");
        if let Some(ref sink) = sink_absorbed_graph.sink() {
            eprintln!("Sink src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Custom { .. } => "Custom".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    _ => format!("{:?}", std::mem::discriminant(&src.op)),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
        }

        // SinkBufferAbsorptionを適用
        let final_suggestions = sink_buffer_absorber.suggest(sink_absorbed_graph);
        eprintln!("\n=== After SinkBufferAbsorption ===");
        eprintln!("Got {} suggestions", final_suggestions.len());

        if !final_suggestions.is_empty() {
            let final_graph = &final_suggestions[0];
            if let Some(ref sink) = final_graph.sink() {
                eprintln!("Sink src count: {}", sink.src.len());
                for (i, src) in sink.src.iter().enumerate() {
                    let op_name = match &src.op {
                        GraphOp::Custom { .. } => "Custom".to_string(),
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
            // SinkAbsorption後に入力Bufferがなければ提案はない
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
    fn test_sink_buffer_absorption_preserves_output_buffers() {
        let sink_buffer_absorber = SinkBufferAbsorptionSuggester::new();

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

        // SinkBufferAbsorptionを適用
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
