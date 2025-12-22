//! Elementwise演算のLowering
//!
//! Elementwise演算の式変換と定数処理を担当します。
//! 実際のAST関数生成はreduce.rs (FusedElementwiseReduce) で統一して行います。

use crate::ast::{AstNode, Literal, helper::wildcard};
use crate::graph::{ElementwiseOp, GraphNode, GraphNodeData, GraphOp};
use std::collections::{HashMap, HashSet};

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

/// 純粋な定数ノードを評価してLiteralを取得
///
/// 注: 現時点ではスカラー演算のみサポート
pub fn evaluate_pure_const(node: &GraphNode) -> Option<Literal> {
    match &node.op {
        GraphOp::Const(lit) => Some(lit.clone()),
        GraphOp::Elementwise { op } => {
            match node.src.len() {
                1 => {
                    // 単項演算
                    let val = evaluate_pure_const(&node.src[0])?;
                    match (op, val) {
                        (ElementwiseOp::Neg, Literal::F32(v)) => Some(Literal::F32(-v)),
                        (ElementwiseOp::Neg, Literal::I64(v)) => Some(Literal::I64(-v)),
                        (ElementwiseOp::Recip, Literal::F32(v)) => Some(Literal::F32(1.0 / v)),
                        (ElementwiseOp::Sqrt, Literal::F32(v)) => Some(Literal::F32(v.sqrt())),
                        (ElementwiseOp::Exp2, Literal::F32(v)) => Some(Literal::F32(v.exp2())),
                        (ElementwiseOp::Log2, Literal::F32(v)) => Some(Literal::F32(v.log2())),
                        (ElementwiseOp::Sin, Literal::F32(v)) => Some(Literal::F32(v.sin())),
                        _ => None,
                    }
                }
                2 => {
                    // 二項演算
                    let left = evaluate_pure_const(&node.src[0])?;
                    let right = evaluate_pure_const(&node.src[1])?;
                    match (op, left, right) {
                        (ElementwiseOp::Add, Literal::F32(l), Literal::F32(r)) => {
                            Some(Literal::F32(l + r))
                        }
                        (ElementwiseOp::Add, Literal::I64(l), Literal::I64(r)) => {
                            Some(Literal::I64(l + r))
                        }
                        (ElementwiseOp::Mul, Literal::F32(l), Literal::F32(r)) => {
                            Some(Literal::F32(l * r))
                        }
                        (ElementwiseOp::Mul, Literal::I64(l), Literal::I64(r)) => {
                            Some(Literal::I64(l * r))
                        }
                        (ElementwiseOp::Idiv, Literal::I64(l), Literal::I64(r)) => {
                            Some(Literal::I64(l / r))
                        }
                        (ElementwiseOp::Rem, Literal::I64(l), Literal::I64(r)) => {
                            Some(Literal::I64(l % r))
                        }
                        (ElementwiseOp::Max, Literal::F32(l), Literal::F32(r)) => {
                            Some(Literal::F32(l.max(r)))
                        }
                        (ElementwiseOp::Max, Literal::I64(l), Literal::I64(r)) => {
                            Some(Literal::I64(l.max(r)))
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// 定数ノードを式に埋め込む
///
/// - 直接のConstノードはそのまま埋め込む
/// - 純粋な定数ノード（Constのみに依存するElementwise等）は評価して埋め込む
pub fn embed_constants(expr: &AstNode, srcs: &[GraphNode]) -> AstNode {
    let mut mappings = HashMap::new();
    let mut non_const_idx = 0;

    for (i, src) in srcs.iter().enumerate() {
        if let GraphOp::Const(lit) = &src.op {
            // 直接のConstノード
            mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
        } else {
            // 純粋な定数ノードかチェックして埋め込む
            let mut visited = HashSet::new();
            if is_pure_const_impl(src, &mut visited)
                && let Some(lit) = evaluate_pure_const(src)
            {
                mappings.insert(i.to_string(), AstNode::Const(lit));
                continue;
            }

            // 非Constノードは元のインデックスを維持
            if non_const_idx != i {
                mappings.insert(i.to_string(), wildcard(non_const_idx.to_string()));
            }
            non_const_idx += 1;
        }
    }

    if mappings.is_empty() {
        expr.clone()
    } else {
        expr.substitute(&mappings)
    }
}
