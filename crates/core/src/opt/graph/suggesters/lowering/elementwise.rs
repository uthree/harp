//! Elementwise演算のLowering
//!
//! Elementwise、FusedElementwise演算のAST関数生成を担当します。

use crate::ast::{AstNode, DType as AstDType, Literal, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{DType as GraphDType, ElementwiseOp, GraphNode, GraphNodeData, GraphOp};
use std::collections::{HashMap, HashSet};

use crate::graph::shape::Expr;

use super::helpers::{
    build_contiguous_offset_with_shape, graph_dtype_to_ast, wrap_with_loops_with_shape,
};

/// ElementwiseOpから演算式を構築
pub fn build_elementwise_expr(op: &ElementwiseOp) -> AstNode {
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

/// Elementwise演算の関数を生成
pub fn build_elementwise_function(
    node: &GraphNode,
    op: &ElementwiseOp,
    name: &str,
) -> Option<AstNode> {
    let shape = node.view.shape();
    let ndim = shape.len();

    // 演算式を構築
    let expr = build_elementwise_expr(op);

    // 入力数を計算（Constノードおよび純粋な定数ノードを除く）
    let num_inputs = node
        .src
        .iter()
        .filter(|s| !matches!(s.op, GraphOp::Const(_)) && !is_pure_const_node(s))
        .count();

    // 定数を埋め込んだ式を構築
    let expr_with_consts = embed_constants(&expr, &node.src);

    Some(build_elementwise_function_impl_with_shape(
        ndim,
        num_inputs,
        expr_with_consts,
        &node.dtype,
        name,
        Some(shape),
    ))
}

/// FusedElementwise演算の関数を生成
pub fn build_fused_elementwise_function(
    node: &GraphNode,
    expr: &AstNode,
    name: &str,
) -> Option<AstNode> {
    let shape = node.view.shape();
    let ndim = shape.len();
    let num_inputs = node
        .src
        .iter()
        .filter(|s| !matches!(s.op, GraphOp::Const(_)) && !is_pure_const_node(s))
        .count();

    let expr_with_consts = embed_constants(expr, &node.src);

    Some(build_elementwise_function_impl_with_shape(
        ndim,
        num_inputs,
        expr_with_consts,
        &node.dtype,
        name,
        Some(shape),
    ))
}

/// Elementwise関数の実装（具体的なshapeを使用）
pub fn build_elementwise_function_impl_with_shape(
    ndim: usize,
    num_inputs: usize,
    expr: AstNode,
    output_dtype: &GraphDType,
    name: &str,
    shape: Option<&[Expr]>,
) -> AstNode {
    let offset = build_contiguous_offset_with_shape(ndim, shape);
    let load_dtype = graph_dtype_to_ast(output_dtype);

    // 入力のロードを含む式を構築
    let mut mappings = HashMap::new();
    for i in 0..num_inputs {
        let load_node = load(var(ph::input(i)), offset.clone(), load_dtype.clone());
        mappings.insert(i.to_string(), load_node);
    }
    let final_expr = expr.substitute(&mappings);

    let store_stmt = store(var(ph::OUTPUT), offset, final_expr);
    let body = wrap_with_loops_with_shape(ndim, vec![store_stmt], shape);

    function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    )
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
                        (ElementwiseOp::Neg, Literal::Int(v)) => Some(Literal::Int(-v)),
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
                        (ElementwiseOp::Add, Literal::Int(l), Literal::Int(r)) => {
                            Some(Literal::Int(l + r))
                        }
                        (ElementwiseOp::Mul, Literal::F32(l), Literal::F32(r)) => {
                            Some(Literal::F32(l * r))
                        }
                        (ElementwiseOp::Mul, Literal::Int(l), Literal::Int(r)) => {
                            Some(Literal::Int(l * r))
                        }
                        (ElementwiseOp::Idiv, Literal::Int(l), Literal::Int(r)) => {
                            Some(Literal::Int(l / r))
                        }
                        (ElementwiseOp::Rem, Literal::Int(l), Literal::Int(r)) => {
                            Some(Literal::Int(l % r))
                        }
                        (ElementwiseOp::Max, Literal::F32(l), Literal::F32(r)) => {
                            Some(Literal::F32(l.max(r)))
                        }
                        (ElementwiseOp::Max, Literal::Int(l), Literal::Int(r)) => {
                            Some(Literal::Int(l.max(r)))
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
