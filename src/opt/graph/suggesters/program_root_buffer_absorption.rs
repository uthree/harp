//! Sink Buffer Absorption Suggester
//!
//! SinkノードのsrcにあるBufferノード（入力バッファ）とConstノードを除去するSuggester。
//!
//! # 処理フロー
//! 1. srcにある直接の入力Bufferノード（output_で始まらない）またはConstを検出
//! 2. 検出したノードをsrcから除去
//!
//! # 設計方針
//! - Viewノードは処理しない（ViewMergeSuggesterに委譲）
//! - GraphNodeのviewフィールドにView情報が保持されているため、Viewノードをたどる必要はない
//!
//! # BufferAbsorptionSuggesterとの関係
//! - BufferAbsorptionSuggester: Buffer → Custom の吸収
//! - ProgramRootBufferAbsorptionSuggester: Buffer/Const → Sink の除去
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

    /// ノードがSinkから除去すべき入力パターンかどうかを判定
    ///
    /// 直接のBuffer(入力)またはConstのみを検出します。
    /// Viewノードはたどりません（ViewMergeSuggesterに委譲）。
    fn is_removable_input(node: &GraphNode) -> bool {
        match &node.op {
            // 入力Buffer（output_で始まらない）
            GraphOp::Buffer { name } => !name.starts_with("output_"),
            // Const も除去対象
            GraphOp::Const(_) | GraphOp::ComplexConst { .. } => true,
            // その他（Viewを含む）は除去しない
            _ => false,
        }
    }

    /// Sinkのsrcに除去可能な入力があるか確認
    fn has_removable_inputs(&self, graph: &Graph) -> bool {
        if let Some(sink) = graph.sink() {
            sink.src.iter().any(Self::is_removable_input)
        } else {
            false
        }
    }

    /// Sinkノードから入力Buffer/Constを除去
    fn remove_inputs(&self, graph: &Graph) -> Option<Graph> {
        let sink = graph.sink()?;

        // srcから入力Buffer/Constを除去
        let new_src: Vec<GraphNode> = sink
            .src
            .iter()
            .filter(|src| !Self::is_removable_input(src))
            .cloned()
            .collect();

        // srcに変更がなければ何もしない
        if new_src.len() == sink.src.len() {
            return None;
        }

        log::debug!(
            "ProgramRootBufferAbsorption: removing {} input nodes (Buffer/Const) from Sink.src",
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

        // 除去可能な入力がない場合は何もしない
        if !self.has_removable_inputs(graph) {
            return vec![];
        }

        // 入力Buffer/Constを除去
        if let Some(new_graph) = self.remove_inputs(graph) {
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
    fn test_view_nodes_not_removed() {
        let sink_buffer_absorber = ProgramRootBufferAbsorptionSuggester::new();

        // View → Buffer のチェーンをテスト
        // Viewノードは除去されず、ViewMergeSuggesterに委譲される
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

        // Viewノードは除去されない（ViewMergeSuggesterに委譲）
        // このテストではElementwiseがsinkのsrcにあるので、除去対象がない
    }

    #[test]
    fn test_is_removable_input() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);

        // Buffer(a) は除去対象
        assert!(ProgramRootBufferAbsorptionSuggester::is_removable_input(&a));

        // View → Buffer(a) のViewノードは除去対象ではない
        let a_view = a.view(a.view.clone());
        assert!(!ProgramRootBufferAbsorptionSuggester::is_removable_input(
            &a_view
        ));

        // 出力バッファは除去対象ではない
        let output_buffer = GraphNode::new(
            DType::F32,
            GraphOp::Buffer {
                name: "output_c".to_string(),
            },
            vec![],
            a.view.clone(),
        );
        assert!(!ProgramRootBufferAbsorptionSuggester::is_removable_input(
            &output_buffer
        ));
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
