//! TensorLowerer用ヘルパー関数
//!
//! オフセット計算、ループ生成、Reduce用ユーティリティを提供します。

use crate::ast::DType;
use crate::ast::{AstNode, DType as AstDType, Literal, Scope, helper::*};
use crate::tensor::ReduceOp;
use crate::tensor::shape::{Expr, View};
use std::collections::HashSet;

// ============================================================================
// プレースホルダー
// ============================================================================

/// カスタムプレースホルダー名
pub mod ph {
    /// 入力バッファのプレースホルダー名を生成
    pub fn input(index: usize) -> String {
        format!("input{}", index)
    }

    /// 出力バッファのプレースホルダー名
    pub const OUTPUT: &str = "output";

    /// Shape変数のプレースホルダー名を生成
    pub fn shape(axis: usize) -> String {
        format!("shape{}", axis)
    }

    /// ループインデックス変数のプレースホルダー名を生成
    pub fn ridx(axis: usize) -> String {
        format!("ridx{}", axis)
    }
}

// ============================================================================
// 型変換
// ============================================================================

/// DTypeをAstDTypeに変換
///
/// Note: DTypeとAstDTypeは同じ型になったため、この関数は主にフォールバック処理を行う
pub fn dtype_to_ast(dtype: &DType) -> AstDType {
    match dtype {
        DType::Bool => AstDType::Bool,
        DType::I8 => AstDType::I8,
        DType::I16 => AstDType::I16,
        DType::I32 => AstDType::I32,
        DType::I64 => AstDType::I64,
        DType::U8 => AstDType::U8,
        DType::U16 => AstDType::U16,
        DType::U32 => AstDType::U32,
        DType::U64 => AstDType::U64,
        DType::F32 => AstDType::F32,
        DType::F64 => AstDType::F64,
        DType::Int => AstDType::Int,
        DType::Unknown => AstDType::F32, // デフォルトでF32
        // Tensor領域では通常使用しないが、そのまま通す
        DType::Ptr(inner) => AstDType::Ptr(inner.clone()),
        DType::Vec(inner, size) => AstDType::Vec(inner.clone(), *size),
        DType::Tuple(types) => AstDType::Tuple(types.clone()),
    }
}

// ============================================================================
// Shape/Expr変換
// ============================================================================

/// Shape式をAstNodeに変換
pub fn shape_expr_to_ast(expr: &Expr) -> AstNode {
    expr.clone().into()
}

/// 指定軸のShape式をAstNodeに変換
pub fn shape_dim_to_ast(shape: Option<&[Expr]>, axis: usize) -> AstNode {
    if let Some(s) = shape
        && axis < s.len()
    {
        return shape_expr_to_ast(&s[axis]);
    }
    // フォールバック: プレースホルダー変数を使用
    var(ph::shape(axis))
}

// ============================================================================
// オフセット計算
// ============================================================================

/// 連続メモリのオフセット計算式を構築
pub fn build_contiguous_offset(ndim: usize) -> AstNode {
    build_contiguous_offset_with_shape(ndim, None)
}

/// 連続メモリのオフセット計算式を構築（具体的なshapeを使用）
pub fn build_contiguous_offset_with_shape(ndim: usize, shape: Option<&[Expr]>) -> AstNode {
    if ndim == 0 {
        return const_int(0);
    }

    let mut offset = var(ph::ridx(ndim - 1));

    for axis in (0..ndim - 1).rev() {
        let mut stride = shape_dim_to_ast(shape, axis + 1);
        for inner_axis in (axis + 2)..ndim {
            stride = stride * shape_dim_to_ast(shape, inner_axis);
        }
        offset = var(ph::ridx(axis)) * stride + offset;
    }

    offset
}

/// 複数の軸を除いた連続メモリのオフセット計算式を構築（複数軸Reduce用、具体的なshapeを使用）
pub fn build_contiguous_offset_excluding_axes_with_shape(
    ndim: usize,
    exclude_axes: &[usize],
    shape: Option<&[Expr]>,
) -> AstNode {
    let exclude_set: HashSet<usize> = exclude_axes.iter().copied().collect();

    let mut output_axes = Vec::new();
    for axis in 0..ndim {
        if !exclude_set.contains(&axis) {
            output_axes.push(axis);
        }
    }

    let output_ndim = output_axes.len();
    if output_ndim == 0 {
        return const_int(0);
    }

    let mut offset = var(ph::ridx(output_axes[output_ndim - 1]));

    for (out_axis, &in_axis) in output_axes.iter().enumerate().take(output_ndim - 1).rev() {
        let stride = if out_axis + 1 < output_axes.len() {
            let next_in_axis = output_axes[out_axis + 1];
            let mut s = shape_dim_to_ast(shape, next_in_axis);
            for &inner_in_axis in &output_axes[out_axis + 2..] {
                s = s * shape_dim_to_ast(shape, inner_in_axis);
            }
            s
        } else {
            const_int(1)
        };

        offset = var(ph::ridx(in_axis)) * stride + offset;
    }

    offset
}

/// 線形インデックスのオフセット計算式を構築
///
/// ndim次元のループインデックスから単一の線形インデックスを計算
/// 計算: ridx0 * (s1 * s2 * ...) + ridx1 * (s2 * s3 * ...) + ... + ridxN
/// reshapeで形状が変わっても要素数が同じなら同じ線形インデックスになる
pub fn build_linear_offset_with_shape(ndim: usize, shape: &[Expr]) -> AstNode {
    if ndim == 0 {
        return const_int(0);
    }
    // 線形インデックス = 各ridxの連続オフセット
    build_contiguous_offset_with_shape(ndim, Some(shape))
}

/// Viewを考慮したストライドベースのオフセット計算式を構築
pub fn build_strided_offset(view: &View, ndim: usize) -> AstNode {
    if ndim == 0 {
        return const_int(0);
    }

    match view {
        View::Linear {
            strides, offset, ..
        } => {
            let mut result: AstNode = offset.clone().into();

            for (axis, stride_expr) in strides.iter().enumerate().take(ndim) {
                let stride: AstNode = stride_expr.clone().into();
                result = result + var(ph::ridx(axis)) * stride;
            }

            result
        }
        View::IndexExpr { index_expr, .. } => index_expr.clone().into(),
        View::Masked { inner, .. } => {
            // Maskedの場合は内側のViewのオフセット計算を使用
            // 注意: 条件チェックはLowerer側で別途処理される
            build_strided_offset(inner, ndim)
        }
    }
}

// ============================================================================
// ループ生成
// ============================================================================

/// ネストされたループで本体をラップ
pub fn wrap_with_loops(ndim: usize, inner_body: Vec<AstNode>) -> AstNode {
    wrap_with_loops_with_shape(ndim, inner_body, None)
}

/// ネストされたループで本体をラップ（具体的なshapeを使用）
pub fn wrap_with_loops_with_shape(
    ndim: usize,
    inner_body: Vec<AstNode>,
    shape: Option<&[Expr]>,
) -> AstNode {
    if ndim == 0 {
        return block(inner_body, Scope::new());
    }

    let mut body = block(inner_body, Scope::new());

    for axis in (0..ndim).rev() {
        body = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            shape_dim_to_ast(shape, axis),
            body,
        );
    }

    block(vec![body], Scope::new())
}

/// 複数軸を除いたネストされたループで本体をラップ（スコープ付き、複数軸対応、具体的なshapeを使用）
pub fn wrap_with_loops_excluding_axes_with_scope_and_shape(
    ndim: usize,
    exclude_axes: &[usize],
    inner_body: Vec<AstNode>,
    scope: Scope,
    shape: Option<&[Expr]>,
) -> AstNode {
    let exclude_set: HashSet<usize> = exclude_axes.iter().copied().collect();

    if ndim == 0 {
        return block(inner_body, scope);
    }

    let mut body = block(inner_body, scope);

    for axis in (0..ndim).rev() {
        if exclude_set.contains(&axis) {
            continue;
        }
        body = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            shape_dim_to_ast(shape, axis),
            body,
        );
    }

    block(vec![body], Scope::new())
}

// ============================================================================
// Reduce用ユーティリティ
// ============================================================================

/// Reduce演算の初期値を取得
pub fn get_reduce_init(dtype: &DType, op: &ReduceOp) -> AstNode {
    match op {
        ReduceOp::Sum => match dtype {
            DType::Bool => AstNode::Const(Literal::Bool(false)),
            DType::I32 => const_int(0),
            _ => const_f32(0.0),
        },
        ReduceOp::Prod => match dtype {
            DType::Bool => AstNode::Const(Literal::Bool(true)),
            DType::I32 => const_int(1),
            _ => const_f32(1.0),
        },
        ReduceOp::Max => match dtype {
            DType::Bool => AstNode::Const(Literal::Bool(false)),
            DType::I32 => const_int(i32::MIN as i64),
            _ => const_f32(f32::NEG_INFINITY),
        },
    }
}

/// アキュムレータの型エイリアス
pub type AccumulateFn = Box<dyn Fn(AstNode, AstNode) -> AstNode>;

/// Reduce演算のアキュムレータ（初期値と更新関数）を生成
pub fn build_reduce_accumulator(op: &ReduceOp, dtype: &DType) -> (AstNode, AccumulateFn) {
    match op {
        ReduceOp::Sum => (get_reduce_init(dtype, op), Box::new(|acc, val| acc + val)),
        ReduceOp::Prod => (get_reduce_init(dtype, op), Box::new(|acc, val| acc * val)),
        ReduceOp::Max => (get_reduce_init(dtype, op), Box::new(max)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ph_input() {
        assert_eq!(ph::input(0), "input0");
        assert_eq!(ph::input(1), "input1");
    }

    #[test]
    fn test_ph_ridx() {
        assert_eq!(ph::ridx(0), "ridx0");
        assert_eq!(ph::ridx(1), "ridx1");
    }

    #[test]
    fn test_contiguous_offset_0d() {
        let offset = build_contiguous_offset(0);
        match offset {
            AstNode::Const(Literal::I64(0)) => {}
            _ => panic!("Expected const 0"),
        }
    }

    #[test]
    fn test_reduce_init() {
        let init = get_reduce_init(&DType::F32, &ReduceOp::Sum);
        match init {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 0.0),
            _ => panic!("Expected F32 0.0"),
        }

        let init = get_reduce_init(&DType::F32, &ReduceOp::Max);
        match init {
            AstNode::Const(Literal::F32(v)) => assert!(v.is_infinite() && v.is_sign_negative()),
            _ => panic!("Expected F32 -inf"),
        }
    }
}
