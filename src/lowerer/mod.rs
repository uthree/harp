//! Lowerer - グラフからASTへの変換
//!
//! グラフ最適化により、すべてのノードが単一のProgramに融合されます。
//!
//! # Optimizer作成関数
//!
//! - **create_lowering_optimizer**: マルチフェーズ最適化（ビームサーチ、複数の並列化戦略）
//! - **create_simple_lowering_optimizer**: 貪欲法で高速（ビーム幅=1、Sequential戦略のみ）
//!
//! # 使用例
//!
//! ```ignore
//! use harp::lowerer::{create_lowering_optimizer, create_simple_lowering_optimizer};
//! use harp::opt::graph::GraphOptimizer;
//!
//! // 通常のOptimizer
//! let optimizer = create_lowering_optimizer(4, 5000);
//! let (optimized, history) = optimizer.optimize_with_history(graph);
//!
//! // 高速なOptimizer（実測用）
//! let optimizer = create_simple_lowering_optimizer(5000);
//! let (optimized, history) = optimizer.optimize_with_history(graph);
//! ```

use crate::backend::pipeline::{
    MultiPhaseConfig, create_greedy_optimizer, create_multi_phase_optimizer,
};
use crate::graph::{Graph, ops::GraphOp};
use crate::opt::graph::ChainedGraphOptimizer;

// モジュール宣言
mod subgraph_lowering;
mod utils;

// 公開エクスポート
pub use self::extract_program_from_graph as extract_program;
pub use subgraph_lowering::SubgraphLoweringOptimizer;
pub use utils::create_signature;

/// Lowering用のGraphOptimizerを作成
///
/// マルチフェーズ最適化を使用してグラフをProgramに変換するOptimizerを返します。
/// ビームサーチと複数の並列化戦略により、最適なコードを生成します。
///
/// # Arguments
/// * `beam_width` - ビームサーチの幅
/// * `max_steps` - 各フェーズの最大ステップ数
///
/// # Returns
/// `ChainedGraphOptimizer`（Preparation → Loweringの2フェーズ）
pub fn create_lowering_optimizer(beam_width: usize, max_steps: usize) -> ChainedGraphOptimizer {
    let config = MultiPhaseConfig::new()
        .with_beam_width(beam_width)
        .with_max_steps(max_steps)
        .with_progress(false);
    create_multi_phase_optimizer(config)
}

/// 貪欲法Lowering用のGraphOptimizerを作成
///
/// ビーム幅=1、Sequential戦略のみで高速にloweringを行うOptimizerを返します。
/// 実行時間の実測など、軽量なloweringが必要な場合に使用します。
///
/// # Arguments
/// * `max_steps` - 各フェーズの最大ステップ数
///
/// # Returns
/// `ChainedGraphOptimizer`（貪欲法版）
///
/// # 用途
/// - 実行時間の実測によるコスト評価
/// - 最適化の初期候補生成
/// - デバッグ・テスト用途
pub fn create_simple_lowering_optimizer(max_steps: usize) -> ChainedGraphOptimizer {
    let config = MultiPhaseConfig::new()
        .with_beam_width(1)
        .with_max_steps(max_steps)
        .with_progress(false);
    create_greedy_optimizer(config)
}

// =============================================================================
// 互換性のための関数
// =============================================================================

/// GraphをProgramに変換する公開関数
///
/// 既存コードとの互換性のために提供されています。
/// 内部で`create_lowering_optimizer`を使用してGraphをProgramに変換します。
///
/// # Panics
/// グラフ最適化が単一のProgramに収束しなかった場合にパニックします。
pub(crate) fn lower(graph: Graph) -> crate::ast::AstNode {
    use crate::opt::graph::GraphOptimizer;

    let optimizer = create_lowering_optimizer(4, 5000);
    let (optimized_graph, _history) = optimizer.optimize_with_history(graph);
    extract_program_from_graph(optimized_graph)
}

// =============================================================================
// Program抽出関数
// =============================================================================

/// グラフ内のOutputにKernel(Program)ノードがあれば、そのProgramを返す
pub fn find_custom_program(graph: &Graph) -> Option<crate::ast::AstNode> {
    for output in graph.outputs().values() {
        if let GraphOp::Kernel { ast, .. } = &output.op
            && matches!(ast, crate::ast::AstNode::Program { .. })
        {
            return Some(ast.clone());
        }
    }
    None
}

/// ProgramRootノードからProgramを取得する
pub fn find_program_root_program(graph: &Graph) -> Option<crate::ast::AstNode> {
    if let Some(root) = graph.program_root()
        && let GraphOp::ProgramRoot { ast, .. } = &root.op
        && matches!(ast, crate::ast::AstNode::Program { .. })
    {
        return Some(ast.clone());
    }
    None
}

/// 最適化済みグラフからProgramを抽出する
///
/// グラフ最適化後、ProgramRootまたはKernel(Program)ノードからASTを取得します。
/// どちらも見つからない場合はパニックします。
///
/// # Panics
/// グラフ内にProgramRoot(Program)またはKernel(Program)が存在しない場合
pub fn extract_program_from_graph(optimized_graph: Graph) -> crate::ast::AstNode {
    // SubgraphCallノードが残っていないかチェック
    check_for_unlowered_subgraph_calls(&optimized_graph);

    if let Some(program) = find_program_root_program(&optimized_graph) {
        return program;
    }

    if let Some(program) = find_custom_program(&optimized_graph) {
        return program;
    }

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

/// SubgraphCallノードが残っていないかチェックし、警告を出す
fn check_for_unlowered_subgraph_calls(graph: &Graph) {
    use std::collections::HashSet;

    fn collect_subgraph_calls(
        node: &crate::graph::GraphNode,
        visited: &mut HashSet<*const crate::graph::GraphNodeData>,
        subgraph_names: &mut Vec<String>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        if let GraphOp::SubgraphCall { name } = &node.op
            && !subgraph_names.contains(name)
        {
            subgraph_names.push(name.clone());
        }

        for src in &node.src {
            collect_subgraph_calls(src, visited, subgraph_names);
        }
    }

    let mut visited = HashSet::new();
    let mut subgraph_names = Vec::new();

    for output in graph.outputs().values() {
        collect_subgraph_calls(output, &mut visited, &mut subgraph_names);
    }

    if let Some(root) = graph.program_root() {
        collect_subgraph_calls(root, &mut visited, &mut subgraph_names);
    }

    if !subgraph_names.is_empty() {
        log::warn!(
            "Graph contains unlowered SubgraphCall nodes: {:?}. \
             SubgraphCall nodes cannot be directly lowered to kernels. \
             Consider using the SubgraphInliningSuggester to inline subgraph calls, \
             or manually expand subgraph calls before optimization.",
            subgraph_names
        );
    }
}

#[cfg(test)]
mod tests;
