//! Eager Fusion - 演算呼び出し時の融合判定
//!
//! 演算呼び出し時に即座に融合を試み、可能であれば融合演算を返す。
//!
//! ## 融合パターン
//!
//! | 親演算 | 子演算 | 結果 |
//! |--------|--------|------|
//! | Elementwise | Elementwise | FusedElementwise |
//! | FusedElementwise | Elementwise | FusedElementwise（拡張） |
//! | Elementwise | Reduce | FusedElementwiseReduce |
//! | FusedElementwise | Reduce | FusedElementwiseReduce |
//!
//! ## 融合条件
//!
//! 1. **View互換**: 同一または互換性のあるView
//! 2. **dtype一致**: 同一データ型
//! 3. **単一消費**: 親テンソルが子演算のみに使用される（所有権ベース設計で保証）

use crate::ast::{AstNode, Literal, helper::*};
use crate::graph::DType;
use crate::graph::shape::View;
#[allow(unused_imports)]
use crate::tensor::ops::{ElementwiseOp, ReduceOp, TensorOp};
use crate::tensor::{DimDyn, Tensor, TensorNode};

/// ElementwiseOpをAstNodeに変換する
///
/// # Arguments
/// * `op` - 変換するElementwiseOp
/// * `input_indices` - 入力のプレースホルダーインデックス
///
/// # Returns
/// 対応するAstNode式
pub fn elementwise_to_ast(op: &ElementwiseOp, input_indices: &[usize]) -> AstNode {
    match op {
        // 二項演算
        ElementwiseOp::Add => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            lhs + rhs
        }
        ElementwiseOp::Mul => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            lhs * rhs
        }
        ElementwiseOp::Max => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            max(lhs, rhs)
        }
        ElementwiseOp::Rem => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            rem(lhs, rhs)
        }
        ElementwiseOp::Idiv => {
            let lhs = wildcard(input_indices[0].to_string());
            let rhs = wildcard(input_indices[1].to_string());
            idiv(lhs, rhs)
        }

        // 単項演算
        ElementwiseOp::Neg => {
            // Neg(x) = -1.0 * x
            let input = wildcard(input_indices[0].to_string());
            AstNode::Const(Literal::F32(-1.0)) * input
        }
        ElementwiseOp::Recip => {
            let input = wildcard(input_indices[0].to_string());
            recip(input)
        }
        ElementwiseOp::Log2 => {
            let input = wildcard(input_indices[0].to_string());
            log2(input)
        }
        ElementwiseOp::Exp2 => {
            let input = wildcard(input_indices[0].to_string());
            exp2(input)
        }
        ElementwiseOp::Sin => {
            let input = wildcard(input_indices[0].to_string());
            sin(input)
        }
        ElementwiseOp::Sqrt => {
            let input = wildcard(input_indices[0].to_string());
            sqrt(input)
        }
        ElementwiseOp::Floor => {
            let input = wildcard(input_indices[0].to_string());
            AstNode::Floor(Box::new(input))
        }
    }
}

/// 2つのElementwise演算を融合してAstNodeを生成
///
/// parent -> child の順で演算される場合、childの入力をparentの結果に置換
fn compose_elementwise(parent_op: &ElementwiseOp, child_op: &ElementwiseOp) -> AstNode {
    // parentは入力0, 1を使用（二項演算の場合）
    let parent_expr = if parent_op.is_binary() {
        elementwise_to_ast(parent_op, &[0, 1])
    } else {
        elementwise_to_ast(parent_op, &[0])
    };

    // childはparentの結果を入力として使用
    // childが単項演算なら、parentの結果だけを使用
    // childが二項演算なら、parentの結果と新しい入力を使用
    if child_op.is_unary() {
        // parent_exprをchildの入力として使用
        substitute_wildcard(&elementwise_to_ast(child_op, &[0]), "0", &parent_expr)
    } else {
        // parent_exprをchildの最初の入力として使用
        // 2番目の入力は新しい入力（インデックスは親の入力数に依存）
        let next_input_idx = if parent_op.is_binary() { 2 } else { 1 };
        let child_expr = elementwise_to_ast(child_op, &[0, next_input_idx]);
        substitute_wildcard(&child_expr, "0", &parent_expr)
    }
}

/// FusedElementwiseを拡張してElementwiseを追加
fn extend_fused(parent_expr: &AstNode, child_op: &ElementwiseOp, next_input_idx: usize) -> AstNode {
    if child_op.is_unary() {
        // parent_exprをchildの入力として使用
        substitute_wildcard(&elementwise_to_ast(child_op, &[0]), "0", parent_expr)
    } else {
        // parent_exprをchildの最初の入力、next_input_idxを2番目の入力として使用
        let child_expr = elementwise_to_ast(child_op, &[0, next_input_idx]);
        substitute_wildcard(&child_expr, "0", parent_expr)
    }
}

/// Wildcard名を別の式で置換
fn substitute_wildcard(expr: &AstNode, name: &str, replacement: &AstNode) -> AstNode {
    match expr {
        AstNode::Wildcard(n) if n == name => replacement.clone(),
        AstNode::Add(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Add(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Mul(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Mul(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Recip(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Recip(Box::new(new_inner))
        }
        AstNode::Log2(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Log2(Box::new(new_inner))
        }
        AstNode::Exp2(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Exp2(Box::new(new_inner))
        }
        AstNode::Sin(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Sin(Box::new(new_inner))
        }
        AstNode::Sqrt(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Sqrt(Box::new(new_inner))
        }
        AstNode::Floor(inner) => {
            let new_inner = substitute_wildcard(inner, name, replacement);
            AstNode::Floor(Box::new(new_inner))
        }
        AstNode::Max(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Max(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Rem(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Rem(Box::new(new_lhs), Box::new(new_rhs))
        }
        AstNode::Idiv(lhs, rhs) => {
            let new_lhs = substitute_wildcard(lhs, name, replacement);
            let new_rhs = substitute_wildcard(rhs, name, replacement);
            AstNode::Idiv(Box::new(new_lhs), Box::new(new_rhs))
        }
        // その他のノードはそのまま返す
        _ => expr.clone(),
    }
}

/// Try to fuse and create a TensorNode with eager fusion
///
/// If the source has exactly one input and the parent op can be fused with the child op,
/// returns a fused TensorNode. Otherwise, returns a regular TensorNode.
///
/// # Arguments
/// * `op` - The operation to apply
/// * `src` - Source tensors
/// * `view` - Result view
/// * `dtype` - Result data type
///
/// # Returns
/// A TensorNode (possibly fused)
pub(crate) fn try_fuse_and_create(
    op: TensorOp,
    src: Vec<Tensor<DimDyn>>,
    view: View,
    dtype: DType,
) -> TensorNode {
    // Only try fusion for single-input operations
    if src.len() == 1 {
        let parent_op = &src[0].inner.op;

        if let Some(fused_op) = try_fuse(parent_op, &op) {
            // Fusion successful - inherit parent's sources
            let parent_sources = src[0].inner.src.clone();
            return TensorNode {
                op: fused_op,
                src: parent_sources,
                view,
                dtype,
                name: None,
            };
        }
    }

    // No fusion - create regular TensorNode
    TensorNode::new(op, src, view, dtype)
}

/// 2つのTensorOpの融合を試みる
///
/// # Arguments
/// * `parent` - 親（先に実行される）演算
/// * `child` - 子（後に実行される）演算
///
/// # Returns
/// 融合可能な場合は融合結果のTensorOp、不可能な場合はNone
pub fn try_fuse(parent: &TensorOp, child: &TensorOp) -> Option<TensorOp> {
    match (parent, child) {
        // Elementwise + Elementwise -> FusedElementwise
        (TensorOp::Elementwise { op: p }, TensorOp::Elementwise { op: c }) => {
            let expr = compose_elementwise(p, c);
            Some(TensorOp::FusedElementwise { expr })
        }

        // FusedElementwise + Elementwise -> FusedElementwise（拡張）
        (TensorOp::FusedElementwise { expr: p }, TensorOp::Elementwise { op: c }) => {
            // 既存のexprに含まれる最大のWildcard番号を見つける
            let max_idx = find_max_wildcard_index(p);
            let extended_expr = extend_fused(p, c, max_idx + 1);
            Some(TensorOp::FusedElementwise {
                expr: extended_expr,
            })
        }

        // Elementwise + Reduce -> FusedElementwiseReduce
        (
            TensorOp::Elementwise { op },
            TensorOp::Reduce {
                op: reduce_op,
                axes,
                keepdim,
            },
        ) => {
            let input_indices: Vec<usize> = if op.is_binary() { vec![0, 1] } else { vec![0] };
            let expr = elementwise_to_ast(op, &input_indices);
            Some(TensorOp::FusedElementwiseReduce {
                expr,
                reduce_op: *reduce_op,
                axes: axes.clone(),
                keepdim: *keepdim,
            })
        }

        // FusedElementwise + Reduce -> FusedElementwiseReduce
        (
            TensorOp::FusedElementwise { expr },
            TensorOp::Reduce {
                op: reduce_op,
                axes,
                keepdim,
            },
        ) => Some(TensorOp::FusedElementwiseReduce {
            expr: expr.clone(),
            reduce_op: *reduce_op,
            axes: axes.clone(),
            keepdim: *keepdim,
        }),

        // その他の組み合わせは融合不可
        _ => None,
    }
}

/// AstNode内の最大Wildcardインデックスを見つける
fn find_max_wildcard_index(expr: &AstNode) -> usize {
    match expr {
        AstNode::Wildcard(name) => name.parse::<usize>().unwrap_or(0),
        AstNode::Add(lhs, rhs)
        | AstNode::Mul(lhs, rhs)
        | AstNode::Max(lhs, rhs)
        | AstNode::Rem(lhs, rhs)
        | AstNode::Idiv(lhs, rhs) => find_max_wildcard_index(lhs).max(find_max_wildcard_index(rhs)),
        AstNode::Recip(inner)
        | AstNode::Log2(inner)
        | AstNode::Exp2(inner)
        | AstNode::Sin(inner)
        | AstNode::Sqrt(inner)
        | AstNode::Floor(inner) => find_max_wildcard_index(inner),
        _ => 0,
    }
}

/// 融合が可能かどうかをチェック（View互換性、dtype一致）
///
/// 現在はシンプルな実装。将来的にはViewの互換性チェックを強化する。
pub fn can_fuse(parent: &TensorOp, child: &TensorOp) -> bool {
    // 基本的な融合可能性チェック
    match (parent, child) {
        // Elementwise系 + Elementwise系は常に融合可能
        (TensorOp::Elementwise { .. }, TensorOp::Elementwise { .. }) => true,
        (TensorOp::FusedElementwise { .. }, TensorOp::Elementwise { .. }) => true,

        // Elementwise系 + Reduce系も融合可能
        (TensorOp::Elementwise { .. }, TensorOp::Reduce { .. }) => true,
        (TensorOp::FusedElementwise { .. }, TensorOp::Reduce { .. }) => true,

        // その他は融合不可
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_to_ast_unary() {
        let expr = elementwise_to_ast(&ElementwiseOp::Recip, &[0]);
        // Recip(Wildcard("0"))
        match expr {
            AstNode::Recip(inner) => match *inner {
                AstNode::Wildcard(name) => assert_eq!(name, "0"),
                _ => panic!("Expected Wildcard"),
            },
            _ => panic!("Expected Recip"),
        }
    }

    #[test]
    fn test_elementwise_to_ast_neg() {
        let expr = elementwise_to_ast(&ElementwiseOp::Neg, &[0]);
        // Mul(Const(-1.0), Wildcard("0"))
        match expr {
            AstNode::Mul(lhs, rhs) => {
                match *lhs {
                    AstNode::Const(Literal::F32(v)) => assert_eq!(v, -1.0),
                    _ => panic!("Expected Const(-1.0) for lhs"),
                }
                match *rhs {
                    AstNode::Wildcard(name) => assert_eq!(name, "0"),
                    _ => panic!("Expected Wildcard for rhs"),
                }
            }
            _ => panic!("Expected Mul"),
        }
    }

    #[test]
    fn test_elementwise_to_ast_binary() {
        let expr = elementwise_to_ast(&ElementwiseOp::Add, &[0, 1]);
        // Add(Wildcard("0"), Wildcard("1"))
        match expr {
            AstNode::Add(lhs, rhs) => {
                match *lhs {
                    AstNode::Wildcard(name) => assert_eq!(name, "0"),
                    _ => panic!("Expected Wildcard for lhs"),
                }
                match *rhs {
                    AstNode::Wildcard(name) => assert_eq!(name, "1"),
                    _ => panic!("Expected Wildcard for rhs"),
                }
            }
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_try_fuse_elementwise_elementwise() {
        let parent = TensorOp::Elementwise {
            op: ElementwiseOp::Add,
        };
        let child = TensorOp::Elementwise {
            op: ElementwiseOp::Recip,
        };
        let result = try_fuse(&parent, &child);
        assert!(result.is_some());
        match result.unwrap() {
            TensorOp::FusedElementwise { expr } => {
                // recip(add(w0, w1))
                match expr {
                    AstNode::Recip(_) => {} // Expected
                    _ => panic!("Expected Recip at top level"),
                }
            }
            _ => panic!("Expected FusedElementwise"),
        }
    }

    #[test]
    fn test_try_fuse_elementwise_reduce() {
        let parent = TensorOp::Elementwise {
            op: ElementwiseOp::Mul,
        };
        let child = TensorOp::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keepdim: false,
        };
        let result = try_fuse(&parent, &child);
        assert!(result.is_some());
        match result.unwrap() {
            TensorOp::FusedElementwiseReduce {
                reduce_op,
                axes,
                keepdim,
                ..
            } => {
                assert_eq!(reduce_op, ReduceOp::Sum);
                assert_eq!(axes, vec![0]);
                assert!(!keepdim);
            }
            _ => panic!("Expected FusedElementwiseReduce"),
        }
    }

    #[test]
    fn test_can_fuse() {
        let elem = TensorOp::Elementwise {
            op: ElementwiseOp::Add,
        };
        let reduce = TensorOp::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keepdim: false,
        };

        assert!(can_fuse(&elem, &elem));
        assert!(can_fuse(&elem, &reduce));
        assert!(!can_fuse(&reduce, &elem)); // Reduce + Elementwise は融合不可
    }

    #[test]
    fn test_eager_fusion_unary_chain() {
        use crate::tensor::Dim2;

        // Create a chain: input -> recip -> sqrt
        // Should be fused into a single FusedElementwise
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.recip(); // First elementwise op
        let c = b.sqrt(); // Second elementwise op - should fuse

        // Check that the op is FusedElementwise, not plain Elementwise
        match &c.inner.op {
            TensorOp::FusedElementwise { expr: _ } => {
                // Expected - fusion worked
            }
            TensorOp::Elementwise { .. } => {
                panic!("Expected FusedElementwise, but got Elementwise (fusion failed)");
            }
            other => {
                panic!("Unexpected op type: {:?}", other);
            }
        }
    }

    #[test]
    fn test_eager_fusion_elementwise_reduce() {
        use crate::tensor::Dim2;

        // Create: input -> recip -> reduce_sum
        // Should be fused into FusedElementwiseReduce
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = a.recip(); // Elementwise op
        let c = b.reduce_sum(&[1], false); // Reduce - should fuse with parent

        // Check that the op is FusedElementwiseReduce
        match &c.inner.op {
            TensorOp::FusedElementwiseReduce { reduce_op, .. } => {
                assert_eq!(*reduce_op, ReduceOp::Sum);
            }
            TensorOp::Reduce { .. } => {
                panic!("Expected FusedElementwiseReduce, but got Reduce (fusion failed)");
            }
            other => {
                panic!("Unexpected op type: {:?}", other);
            }
        }
    }
}
