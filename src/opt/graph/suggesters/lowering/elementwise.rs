//! Elementwise演算のLowering
//!
//! Elementwise演算の式変換と定数処理を担当します。
//! 実際のAST関数生成はreduce.rs (FusedElementwiseReduce) で統一して行います。

use crate::ast::AstNode;
use crate::graph::{ElementwiseOp, GraphNode, GraphNodeData, GraphOp};
use std::collections::HashSet;

/// ElementwiseOpから演算式を構築
pub fn build_elementwise_expr(op: &ElementwiseOp) -> AstNode {
    use crate::ast::helper::*;

    match op {
        ElementwiseOp::Add => wildcard("0") + wildcard("1"),
        ElementwiseOp::Mul => wildcard("0") * wildcard("1"),
        ElementwiseOp::Neg => const_f32(-1.0) * wildcard("0"),
        ElementwiseOp::Max => max(wildcard("0"), wildcard("1")),
        ElementwiseOp::Rem => wildcard("0") % wildcard("1"),
        ElementwiseOp::Idiv => idiv(wildcard("0"), wildcard("1")),
        ElementwiseOp::Recip => recip(wildcard("0")),
        ElementwiseOp::Log2 => log2(wildcard("0")),
        ElementwiseOp::Exp2 => exp2(wildcard("0")),
        ElementwiseOp::Sin => sin(wildcard("0")),
        ElementwiseOp::Sqrt => sqrt(wildcard("0")),
        ElementwiseOp::Floor => AstNode::Floor(Box::new(wildcard("0"))),
    }
}

// ============================================================
// 定数処理関連
// ============================================================

/// ノードが「純粋な定数」かどうかをチェック
///
/// 純粋な定数ノードとは、再帰的にConstノードのみに依存するノードのこと。
/// 例: `2.0 * 3.0` は純粋な定数（結果は6.0）
pub fn is_pure_const_node(node: &GraphNode) -> bool {
    let mut visited = HashSet::new();
    is_pure_const_impl(node, &mut visited)
}

/// ノードが「純粋な定数」かどうかをチェック（内部実装）
fn is_pure_const_impl(node: &GraphNode, visited: &mut HashSet<*const GraphNodeData>) -> bool {
    let ptr = node.as_ptr();
    if visited.contains(&ptr) {
        return true; // 循環参照を避ける
    }
    visited.insert(ptr);

    match &node.op {
        GraphOp::Const(_) => true,
        GraphOp::Elementwise { .. } => {
            // すべてのsrcが純粋な定数なら、このノードも純粋な定数
            node.src.iter().all(|s| is_pure_const_impl(s, visited))
        }
        _ => false,
    }
}
