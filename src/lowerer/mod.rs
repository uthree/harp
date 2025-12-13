//! Lowerer - グラフからASTへの変換
//!
//! グラフ最適化（マルチフェーズ最適化）により、すべてのノードが単一のProgramに
//! 融合されます。Lowererはグラフ最適化を実行し、結果のProgramを返します。

use crate::graph::{Graph, ops::GraphOp};

// モジュール宣言
mod utils; // 共通ユーティリティ（create_signature等）

/// Lowerer構造体
///
/// グラフからKernelSignatureを生成するためのユーティリティを提供します。
/// 実際のlowering処理はグラフ最適化（LoweringSuggester, ProgramRootAbsorptionSuggester）
/// によって行われます。
pub struct Lowerer;

impl Lowerer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

/// グラフ内にCustom(Program)ノードがあれば、そのProgramを返す
/// KernelMergeSuggesterの出力を検出するために使用
fn find_custom_program(graph: &Graph) -> Option<crate::ast::AstNode> {
    for output in graph.outputs().values() {
        if let GraphOp::Kernel { ast, .. } = &output.op
            && matches!(ast, crate::ast::AstNode::Program { .. })
        {
            return Some(ast.clone());
        }
    }
    None
}

/// SinkノードからProgramを取得する
/// ProgramRootAbsorptionSuggesterの出力を検出するために使用
fn find_sink_program(graph: &Graph) -> Option<crate::ast::AstNode> {
    if let Some(sink) = graph.sink()
        && let GraphOp::ProgramRoot { ast, .. } = &sink.op
        && matches!(ast, crate::ast::AstNode::Program { .. })
    {
        // ProgramRootがProgramを持っていれば返す（空のProgramも許可）
        // 入力をそのまま出力するケースではfunctionsが空になる
        return Some(ast.clone());
    }
    None
}

/// グラフ最適化を実行する
///
/// マルチフェーズ最適化を使用して、グラフを単一のProgramに収束させます。
/// - Phase 1 (Preparation): グラフ構造の最適化（View挿入、融合など）
/// - Phase 2 (Lowering): Custom変換、ProgramRoot集約
fn optimize_graph_for_lowering(graph: Graph) -> Graph {
    use crate::backend::pipeline::{MultiPhaseConfig, optimize_graph_multi_phase};

    let config = MultiPhaseConfig::new()
        .with_beam_width(4)
        .with_max_steps(5000)
        .with_progress(false);

    let (optimized_graph, _history) = optimize_graph_multi_phase(graph, config);

    optimized_graph
}

/// GraphをProgramに変換する公開関数
///
/// マルチフェーズ最適化を実行し、単一のProgramに収束させます。
/// グラフ最適化が収束しなかった場合はパニックします。
///
/// # Panics
/// グラフ最適化が単一のProgramに収束しなかった場合にパニックします。
/// これは通常、サポートされていないノードタイプがある場合に発生します。
pub(crate) fn lower(graph: Graph) -> crate::ast::AstNode {
    // グラフ最適化を実行（マルチフェーズ最適化）
    let optimized_graph = optimize_graph_for_lowering(graph);

    // Sink(Program)ノードがあればそれを直接返す（ProgramRootAbsorptionSuggesterの出力）
    if let Some(program) = find_sink_program(&optimized_graph) {
        log::debug!("Found Sink(Program) node, returning directly");
        return program;
    }

    // Custom(Program)ノードがあればそれを直接返す（KernelMergeSuggesterの出力）
    if let Some(program) = find_custom_program(&optimized_graph) {
        log::debug!("Found Custom(Program) node, returning directly");
        return program;
    }

    // グラフ最適化が収束しなかった場合はエラー
    // 残っているノードの情報を収集してエラーメッセージに含める
    let remaining_ops: Vec<String> = optimized_graph
        .outputs()
        .values()
        .map(|n| format!("{:?}", std::mem::discriminant(&n.op)))
        .collect();

    panic!(
        "Graph optimization did not converge to a single Program. \
         Remaining output operations: {:?}. \
         This may indicate unsupported node types or a bug in the optimization passes.",
        remaining_ops
    );
}

#[cfg(test)]
mod tests;
