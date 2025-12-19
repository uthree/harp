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
//! ```
//! use harp::lowerer::{create_lowering_optimizer, create_simple_lowering_optimizer};
//! use harp::opt::graph::GraphOptimizer;
//! use harp::graph::{Graph, DType};
//!
//! // テスト用グラフを作成
//! let mut graph = Graph::new();
//! let a = graph.input("a", DType::F32, vec![4]);
//! let b = graph.input("b", DType::F32, vec![4]);
//! let c = a + b;
//! graph.output("c", c);
//!
//! // 通常のOptimizer
//! let optimizer = create_lowering_optimizer(4, 5000);
//! let (optimized, _history) = optimizer.optimize_with_history(graph.clone());
//!
//! // 高速なOptimizer（実測用）
//! let optimizer = create_simple_lowering_optimizer(5000);
//! let (optimized, _history) = optimizer.optimize_with_history(graph);
//! ```

use crate::backend::{MultiPhaseConfig, create_greedy_optimizer, create_multi_phase_optimizer};
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

use std::collections::HashSet;

/// グラフ内のKernelノードを収集してProgramを作成する
///
/// グラフ全体を走査し、全てのKernel(Function)またはKernel(Kernel)ノードを
/// 収集してProgramとして返します。
/// execution_waves情報も収集・マージします。
///
/// 現在の実装では各カーネルを順次実行（各waveに1つのカーネル）しますが、
/// 将来的にはデータフロー解析により並列実行可能なカーネルを同じwaveにグループ化します。
pub fn collect_kernels_as_program(graph: &Graph) -> Option<crate::ast::AstNode> {
    use crate::ast::{AstKernelCallInfo, AstNode};
    use crate::graph::GraphNode;

    let mut kernels: Vec<AstNode> = Vec::new();
    let mut execution_waves: Vec<Vec<AstKernelCallInfo>> = Vec::new();
    let mut visited: HashSet<*const crate::graph::GraphNodeData> = HashSet::new();

    fn collect_kernels(
        node: &GraphNode,
        kernels: &mut Vec<AstNode>,
        execution_waves: &mut Vec<Vec<AstKernelCallInfo>>,
        visited: &mut HashSet<*const crate::graph::GraphNodeData>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // Kernel(Function) または Kernel(Kernel) を収集
        if let GraphOp::Kernel { ast, .. } = &node.op {
            match ast {
                AstNode::Function { .. } | AstNode::Kernel { .. } => {
                    // 重複チェック（同名カーネルは1回だけ追加）
                    let name = match ast {
                        AstNode::Function { name, .. } => name.clone(),
                        AstNode::Kernel { name, .. } => name.clone(),
                        _ => None,
                    };
                    let already_exists = kernels.iter().any(|k| {
                        let existing_name = match k {
                            AstNode::Function { name, .. } => name.clone(),
                            AstNode::Kernel { name, .. } => name.clone(),
                            _ => None,
                        };
                        existing_name == name && name.is_some()
                    });
                    if !already_exists {
                        kernels.push(ast.clone());
                        // カーネル呼び出し情報を生成（各カーネルを別のwaveに配置）
                        if let Some(kernel_name) = name {
                            // Kernelノードからdispatchサイズを取得（AstNodeからExprに変換）
                            use crate::graph::shape::Expr;
                            let (grid_size, local_size) = match ast {
                                AstNode::Kernel {
                                    default_grid_size,
                                    default_thread_group_size,
                                    ..
                                } => {
                                    let gs: [Expr; 3] = [
                                        Expr::try_from(default_grid_size[0].as_ref())
                                            .unwrap_or(Expr::Const(1)),
                                        Expr::try_from(default_grid_size[1].as_ref())
                                            .unwrap_or(Expr::Const(1)),
                                        Expr::try_from(default_grid_size[2].as_ref())
                                            .unwrap_or(Expr::Const(1)),
                                    ];
                                    let ls: [Expr; 3] = [
                                        Expr::try_from(default_thread_group_size[0].as_ref())
                                            .unwrap_or(Expr::Const(1)),
                                        Expr::try_from(default_thread_group_size[1].as_ref())
                                            .unwrap_or(Expr::Const(1)),
                                        Expr::try_from(default_thread_group_size[2].as_ref())
                                            .unwrap_or(Expr::Const(1)),
                                    ];
                                    (gs, ls)
                                }
                                _ => (
                                    [Expr::Const(1), Expr::Const(1), Expr::Const(1)],
                                    [Expr::Const(1), Expr::Const(1), Expr::Const(1)],
                                ),
                            };
                            let call_info = AstKernelCallInfo::new(
                                kernel_name,
                                vec![], // TODO: 入出力情報を解析
                                vec![],
                                grid_size,
                                local_size,
                            );
                            // 各カーネルを別のwaveに配置（順次実行）
                            execution_waves.push(vec![call_info]);
                        }
                    }
                }
                AstNode::Program {
                    functions,
                    execution_waves: inner_waves,
                } => {
                    // Kernel(Program)の場合は中の関数を展開
                    for func in functions {
                        kernels.push(func.clone());
                    }
                    // 内側のexecution_wavesを追加
                    for wave in inner_waves {
                        execution_waves.push(wave.clone());
                    }
                }
                _ => {}
            }
        }

        // 子ノードも走査
        for src in &node.src {
            collect_kernels(src, kernels, execution_waves, visited);
        }
    }

    // 全出力からKernelを収集
    for output in graph.outputs().values() {
        collect_kernels(output, &mut kernels, &mut execution_waves, &mut visited);
    }

    if kernels.is_empty() {
        None
    } else {
        Some(AstNode::Program {
            functions: kernels,
            execution_waves,
        })
    }
}

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

/// 最適化済みグラフからProgramを抽出する
///
/// グラフ最適化後、以下の優先順位でASTを取得します：
/// 1. グラフ内のKernelノードを収集してProgram化
/// 2. Kernel(Program)ノード
///
/// # Panics
/// グラフ内にKernelノードが存在しない場合
pub fn extract_program_from_graph(optimized_graph: Graph) -> crate::ast::AstNode {
    // SubgraphCallノードが残っていないかチェック
    check_for_unlowered_subgraph_calls(&optimized_graph);

    // まずKernelノードを収集してProgram化を試みる
    if let Some(program) = collect_kernels_as_program(&optimized_graph) {
        return program;
    }

    // Kernel(Program)を探す
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
