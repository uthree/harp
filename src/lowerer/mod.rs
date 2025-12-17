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

use std::collections::HashSet;

/// グラフ内のKernelノードを収集してProgramを作成する
///
/// グラフ全体を走査し、全てのKernel(Function)またはKernel(Kernel)ノードを
/// 収集してProgramとして返します。
/// execution_order情報も収集・マージします。
pub fn collect_kernels_as_program(graph: &Graph) -> Option<crate::ast::AstNode> {
    use crate::ast::{AstNode, KernelExecutionInfo};
    use crate::graph::GraphNode;

    let mut kernels: Vec<AstNode> = Vec::new();
    let mut execution_infos: Vec<KernelExecutionInfo> = Vec::new();
    let mut visited: HashSet<*const crate::graph::GraphNodeData> = HashSet::new();

    fn collect_kernels(
        node: &GraphNode,
        kernels: &mut Vec<AstNode>,
        execution_infos: &mut Vec<KernelExecutionInfo>,
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
                        // 単一カーネルの場合、execution_infoを生成
                        if let Some(kernel_name) = name {
                            let current_wave =
                                execution_infos.iter().map(|i| i.wave_id).max().unwrap_or(0);
                            let new_wave = if execution_infos.is_empty() {
                                0
                            } else {
                                current_wave + 1
                            };
                            execution_infos.push(KernelExecutionInfo::new(
                                kernel_name,
                                vec![], // 入出力情報は後で埋める
                                vec![],
                                new_wave,
                            ));
                        }
                    }
                }
                AstNode::Program {
                    functions,
                    execution_order,
                } => {
                    // Kernel(Program)の場合は中の関数を展開
                    // execution_orderがあれば、wave_idをオフセットしてマージ
                    let base_wave = execution_infos
                        .iter()
                        .map(|i| i.wave_id)
                        .max()
                        .map_or(0, |m| m + 1);

                    for func in functions {
                        kernels.push(func.clone());
                    }

                    if let Some(order) = execution_order {
                        for info in order {
                            execution_infos.push(KernelExecutionInfo::new(
                                info.kernel_name.clone(),
                                info.inputs.clone(),
                                info.outputs.clone(),
                                info.wave_id + base_wave,
                            ));
                        }
                    } else {
                        // execution_orderがない場合、各カーネルに連番のwave_idを割り当て
                        for (i, func) in functions.iter().enumerate() {
                            let kernel_name = match func {
                                AstNode::Kernel { name, .. } => {
                                    name.clone().unwrap_or_else(|| format!("kernel_{}", i))
                                }
                                AstNode::Function { name, .. } => {
                                    name.clone().unwrap_or_else(|| format!("func_{}", i))
                                }
                                _ => format!("unknown_{}", i),
                            };
                            execution_infos.push(KernelExecutionInfo::new(
                                kernel_name,
                                vec![],
                                vec![],
                                base_wave + i,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        // 子ノードも走査
        for src in &node.src {
            collect_kernels(src, kernels, execution_infos, visited);
        }
    }

    // 全出力からKernelを収集
    for output in graph.outputs().values() {
        collect_kernels(output, &mut kernels, &mut execution_infos, &mut visited);
    }

    if kernels.is_empty() {
        None
    } else {
        // execution_infosが空でなければSomeとして設定
        let execution_order = if execution_infos.is_empty() {
            None
        } else {
            Some(execution_infos)
        };
        Some(AstNode::Program {
            functions: kernels,
            execution_order,
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
