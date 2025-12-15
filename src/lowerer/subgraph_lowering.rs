//! サブグラフを個別カーネルに変換するモジュール
//!
//! SubgraphCallノードを検出し、各サブグラフを独立したカーネル関数として
//! 低レベル化します。実行順序はKernelCallメタデータとして保持されます。

use std::collections::HashSet;

use crate::ast::AstNode;
use crate::ast::program::KernelCall;
use crate::graph::shape::Expr;
use crate::graph::{Graph, GraphNode, ops::GraphOp};
use crate::opt::graph::{GraphOptimizer, OptimizationHistory};

/// サブグラフを個別カーネルに変換するOptimizer
///
/// # 処理フロー
/// 1. グラフ内のSubgraphCallノードを検出
/// 2. 各サブグラフを独立してlower
/// 3. カーネル関数としてProgramに追加
/// 4. 実行順序をKernelCallとして記録
#[derive(Debug, Clone, Default)]
pub struct SubgraphLoweringOptimizer {
    /// 最大再帰深度（サブグラフがネストしている場合）
    max_depth: usize,
}

impl SubgraphLoweringOptimizer {
    /// 新しいSubgraphLoweringOptimizerを作成
    pub fn new() -> Self {
        Self { max_depth: 10 }
    }

    /// 最大再帰深度を設定
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// グラフからSubgraphCallノードを収集
    fn collect_subgraph_calls(graph: &Graph) -> Vec<(String, GraphNode)> {
        let mut visited = HashSet::new();
        let mut calls = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const crate::graph::GraphNodeData>,
            calls: &mut Vec<(String, GraphNode)>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            if let GraphOp::SubgraphCall { name } = &node.op {
                // 同名のサブグラフは1回だけ収集
                if !calls.iter().any(|(n, _)| n == name) {
                    calls.push((name.clone(), node.clone()));
                }
            }

            for src in &node.src {
                visit(src, visited, calls);
            }
        }

        // 全出力からSubgraphCallを収集
        for output in graph.outputs().values() {
            visit(output, &mut visited, &mut calls);
        }

        calls
    }

    /// サブグラフを個別にlowerしてカーネル関数を生成
    fn lower_subgraph(
        &self,
        name: &str,
        subgraph: &Graph,
        depth: usize,
    ) -> Result<(Vec<AstNode>, Vec<KernelCall>), String> {
        if depth > self.max_depth {
            return Err(format!(
                "Subgraph nesting depth exceeded maximum ({}) for '{}'",
                self.max_depth, name
            ));
        }

        log::debug!("Lowering subgraph '{}' at depth {}", name, depth);

        // サブグラフ内にさらにSubgraphCallがあるか確認
        let nested_calls = Self::collect_subgraph_calls(subgraph);

        let mut all_kernels = Vec::new();
        let mut all_execution_order = Vec::new();

        // ネストしたサブグラフを先に処理
        for (nested_name, _) in &nested_calls {
            if let Some(nested_subgraph) = subgraph.subgraph(nested_name) {
                let (kernels, exec_order) =
                    self.lower_subgraph(nested_name, nested_subgraph, depth + 1)?;
                all_kernels.extend(kernels);
                all_execution_order.extend(exec_order);
            }
        }

        // 現在のサブグラフをlower
        let lowered = super::lower(subgraph.clone());

        // Programから関数を抽出
        if let AstNode::Program { functions } = lowered {
            // カーネル名をサブグラフ名でプレフィックス
            let prefixed_kernels: Vec<AstNode> = functions
                .into_iter()
                .map(|f| Self::prefix_kernel_name(f, name))
                .collect();

            all_kernels.extend(prefixed_kernels.clone());

            // カーネル実行順序情報を生成（CompiledProgram.execution_wavesで使用）
            // Note: 実行順序はCompiledProgramレベルで管理されるが、
            //       ここでもKernelCallを生成しておく
            if let Some(AstNode::Kernel {
                name: Some(kernel_name),
                params,
                default_grid_size,
                default_thread_group_size,
                ..
            }) = prefixed_kernels.first()
            {
                let inputs: Vec<String> = params
                    .iter()
                    .filter(|p| p.name.starts_with("input") || p.name.starts_with("param"))
                    .map(|p| p.name.clone())
                    .collect();

                let outputs: Vec<String> = params
                    .iter()
                    .filter(|p| p.name.starts_with("output"))
                    .map(|p| p.name.clone())
                    .collect();

                let grid_size: Vec<Expr> = default_grid_size
                    .iter()
                    .map(|g| Self::ast_to_expr(g))
                    .collect();

                let thread_group_size: Vec<Expr> = default_thread_group_size
                    .iter()
                    .map(|t| Self::ast_to_expr(t))
                    .collect();

                all_execution_order.push(KernelCall::new(
                    kernel_name.clone(),
                    inputs,
                    outputs,
                    grid_size,
                    thread_group_size,
                ));
            }
        }

        Ok((all_kernels, all_execution_order))
    }

    /// カーネル名にサブグラフ名をプレフィックスとして追加
    fn prefix_kernel_name(kernel: AstNode, prefix: &str) -> AstNode {
        match kernel {
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => {
                let new_name = name.map(|n| format!("{}_{}", prefix, n));
                AstNode::Kernel {
                    name: new_name,
                    params,
                    return_type,
                    body,
                    default_grid_size,
                    default_thread_group_size,
                }
            }
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                let new_name = name.map(|n| format!("{}_{}", prefix, n));
                AstNode::Function {
                    name: new_name,
                    params,
                    return_type,
                    body,
                }
            }
            other => other,
        }
    }

    /// AstNodeを式(Expr)に変換
    fn ast_to_expr(ast: &AstNode) -> Expr {
        match ast {
            AstNode::Const(lit) => match lit {
                crate::ast::Literal::Int(i) => Expr::Const(*i),
                crate::ast::Literal::F32(f) => Expr::Const(*f as isize),
                crate::ast::Literal::Bool(_) => Expr::Const(1),
            },
            AstNode::Var(name) => Expr::Var(name.clone()),
            AstNode::Mul(a, b) => Expr::Mul(
                Box::new(Self::ast_to_expr(a)),
                Box::new(Self::ast_to_expr(b)),
            ),
            AstNode::Add(a, b) => Expr::Add(
                Box::new(Self::ast_to_expr(a)),
                Box::new(Self::ast_to_expr(b)),
            ),
            _ => Expr::Const(1),
        }
    }

    /// メイングラフとサブグラフのカーネルを統合したProgramを生成
    pub fn process_graph(&self, graph: &Graph) -> Result<Graph, String> {
        let subgraph_calls = Self::collect_subgraph_calls(graph);

        if subgraph_calls.is_empty() {
            // SubgraphCallがない場合はそのまま返す
            return Ok(graph.clone());
        }

        log::info!(
            "Processing {} subgraph call(s): {:?}",
            subgraph_calls.len(),
            subgraph_calls.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );

        let mut all_kernels = Vec::new();
        let mut all_execution_order = Vec::new();

        // 各サブグラフを処理
        for (name, _call_node) in &subgraph_calls {
            if let Some(subgraph) = graph.subgraph(name) {
                let (kernels, exec_order) = self.lower_subgraph(name, subgraph, 0)?;
                all_kernels.extend(kernels);
                all_execution_order.extend(exec_order);
            } else {
                return Err(format!("Subgraph '{}' not found", name));
            }
        }

        // 新しいグラフを作成（サブグラフを展開済み）
        // TODO: SubgraphCallノードをインライン展開するか、
        //       中間バッファを通じた呼び出しに変換する
        //       現時点では警告を出してそのまま返す
        log::warn!(
            "SubgraphLowering: {} subgraph(s) processed, {} kernel(s) generated. \
             Note: SubgraphCall nodes are not yet replaced in the graph.",
            subgraph_calls.len(),
            all_kernels.len()
        );

        Ok(graph.clone())
    }
}

impl GraphOptimizer for SubgraphLoweringOptimizer {
    fn name(&self) -> Option<&str> {
        Some("SubgraphLowering")
    }

    fn optimize(&self, graph: Graph) -> Graph {
        match self.process_graph(&graph) {
            Ok(optimized) => optimized,
            Err(e) => {
                log::error!("SubgraphLowering failed: {}", e);
                graph
            }
        }
    }

    fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        let optimized = self.optimize(graph);
        (optimized, OptimizationHistory::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_subgraph_calls_empty() {
        let graph = Graph::new();
        let calls = SubgraphLoweringOptimizer::collect_subgraph_calls(&graph);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_subgraph_lowering_optimizer_creation() {
        let optimizer = SubgraphLoweringOptimizer::new();
        assert_eq!(optimizer.max_depth, 10);

        let optimizer = optimizer.with_max_depth(5);
        assert_eq!(optimizer.max_depth, 5);
    }
}
