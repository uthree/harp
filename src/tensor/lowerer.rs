//! TensorLowerer - TensorNodeから直接ASTへの変換
//!
//! TensorNodeツリーをASTに変換する。
//! 現在の実装では内部でGraphOptimizerを使用するが、
//! 将来的には直接Loweringに移行する予定。
//!
//! # 使用例
//!
//! ```ignore
//! use harp::tensor::{Tensor, Dim2};
//! use harp::tensor::lowerer::TensorLowerer;
//!
//! let a = Tensor::<Dim2>::input("a", [2, 3]);
//! let b = Tensor::<Dim2>::input("b", [2, 3]);
//! let c = &a + &b;
//!
//! let mut lowerer = TensorLowerer::new();
//! let ast = lowerer.lower(&c.clone().into_dyn());
//! ```

use std::collections::{HashMap, HashSet};

use crate::ast::AstNode;
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::lowerer::{create_simple_lowering_optimizer, extract_program};
use crate::opt::graph::GraphOptimizer;
use crate::tensor::{DimDyn, Tensor, TensorNode};

/// TensorをASTに変換するLowerer
///
/// TensorNodeツリーをトラバースし、ASTプログラムを生成する。
/// 現在はGraphOptimizer経由で変換を行うが、将来的には直接変換に移行予定。
pub struct TensorLowerer {
    /// 変換済みノードのキャッシュ
    node_cache: HashMap<*const TensorNode, GraphNode>,
}

impl TensorLowerer {
    /// 新しいTensorLowererを作成
    pub fn new() -> Self {
        Self {
            node_cache: HashMap::new(),
        }
    }

    /// TensorをASTに変換
    ///
    /// # Arguments
    /// * `tensor` - 変換するテンソル
    ///
    /// # Returns
    /// ASTプログラム
    pub fn lower(&mut self, tensor: &Tensor<DimDyn>) -> AstNode {
        // 1. TensorNode → Graph変換
        let graph = self.build_graph(tensor);

        // 2. Graph → AST変換（既存のパイプライン使用）
        let optimizer = create_simple_lowering_optimizer(5000);
        let (optimized_graph, _history) = optimizer.optimize_with_history(graph);

        // 3. ProgramをGraphから抽出
        extract_program(optimized_graph)
    }

    /// TensorからGraphを構築
    fn build_graph(&mut self, tensor: &Tensor<DimDyn>) -> Graph {
        let mut graph = Graph::new();

        // テンソルのGraphNodeを取得（既存のnode fieldを使用）
        let output_node = tensor.node.clone();

        // 入力バッファを収集
        self.collect_inputs(&output_node, &mut graph);

        // 出力を設定
        graph.output("output", output_node);

        graph
    }

    /// 入力バッファを収集してGraphに登録
    fn collect_inputs(&self, node: &GraphNode, graph: &mut Graph) {
        let mut visited: HashSet<*const GraphNodeData> = HashSet::new();

        fn visit(node: &GraphNode, graph: &mut Graph, visited: &mut HashSet<*const GraphNodeData>) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            // 入力バッファを登録
            if let GraphOp::Buffer { name } = &node.op {
                // 入力メタデータを登録
                let shape = node.view.shape().to_vec();
                graph.register_input_meta(name.clone(), node.dtype.clone(), shape);
            }

            // 再帰的に子ノードを処理
            for src in &node.src {
                visit(src, graph, visited);
            }
        }

        visit(node, graph, &mut visited);
    }
}

impl Default for TensorLowerer {
    fn default() -> Self {
        Self::new()
    }
}

/// TensorをASTに変換する簡易関数
///
/// TensorLowererのインスタンスを作成せずに変換できる便利関数。
pub fn lower_tensor(tensor: &Tensor<DimDyn>) -> AstNode {
    let mut lowerer = TensorLowerer::new();
    lowerer.lower(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_lower_simple_add() {
        let a = Tensor::<Dim2>::input("a", [2, 3]);
        let b = Tensor::<Dim2>::input("b", [2, 3]);
        let c = &a + &b;

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&c.clone().into_dyn());

        // ASTがProgramであることを確認
        match ast {
            AstNode::Program { functions, .. } => {
                assert!(
                    !functions.is_empty(),
                    "Program should have at least one function"
                );
            }
            _ => panic!("Expected AstNode::Program"),
        }
    }

    #[test]
    fn test_lower_fused_operations() {
        let a = Tensor::<Dim2>::input("a", [4, 4]);
        let b = a.recip().sqrt(); // Fused: recip -> sqrt

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&b.clone().into_dyn());

        match ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }
    }

    #[test]
    fn test_lower_reduce() {
        let a = Tensor::<Dim2>::input("a", [4, 4]);
        let b = a.reduce_sum(&[1], false);

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&b);

        match ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }
    }
}
