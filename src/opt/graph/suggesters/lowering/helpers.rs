//! LoweringSuggester用のヘルパー関数
//!
//! オフセット計算、ループ生成、型変換など共通のユーティリティを提供します。

use crate::ast::{AstNode, DType as AstDType, Scope, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{DType as GraphDType, ReduceOp, View};

/// GraphのDTypeをAstのDTypeに変換
pub fn graph_dtype_to_ast(dtype: &GraphDType) -> AstDType {
    match dtype {
        GraphDType::Bool => AstDType::Bool,
        GraphDType::I32 => AstDType::Int,
        GraphDType::F32 => AstDType::F32,
        GraphDType::Complex => AstDType::F32, // 複素数は2つのf32として扱う
        GraphDType::Unknown => AstDType::F32,
    }
}

/// Reduce演算の初期値を取得
pub fn get_reduce_init(dtype: &GraphDType, op: &ReduceOp) -> AstNode {
    match op {
        ReduceOp::Sum => match dtype {
            GraphDType::Bool => AstNode::Const(false.into()),
            GraphDType::I32 => const_int(0),
            _ => const_f32(0.0),
        },
        ReduceOp::Prod => match dtype {
            GraphDType::Bool => AstNode::Const(true.into()),
            GraphDType::I32 => const_int(1),
            _ => const_f32(1.0),
        },
        ReduceOp::Max => match dtype {
            GraphDType::Bool => AstNode::Const(false.into()),
            GraphDType::I32 => const_int(i32::MIN as isize),
            _ => const_f32(f32::NEG_INFINITY),
        },
    }
}

/// 連続メモリのオフセット計算式を構築
pub fn build_contiguous_offset(ndim: usize) -> AstNode {
    if ndim == 0 {
        return const_int(0);
    }

    let mut offset = var(ph::ridx(ndim - 1));

    for axis in (0..ndim - 1).rev() {
        let mut stride = var(ph::shape(axis + 1));
        for inner_axis in (axis + 2)..ndim {
            stride = stride * var(ph::shape(inner_axis));
        }
        offset = var(ph::ridx(axis)) * stride + offset;
    }

    offset
}

/// 特定軸を除いた連続メモリのオフセット計算式を構築（Reduce用）
pub fn build_contiguous_offset_excluding_axis(ndim: usize, exclude_axis: usize) -> AstNode {
    if ndim <= 1 {
        return const_int(0);
    }

    let output_ndim = ndim - 1;
    if output_ndim == 0 {
        return const_int(0);
    }

    let mut output_axes = Vec::new();
    for axis in 0..ndim {
        if axis != exclude_axis {
            output_axes.push(axis);
        }
    }

    let mut offset = var(ph::ridx(output_axes[output_ndim - 1]));

    for (out_axis, &in_axis) in output_axes.iter().enumerate().take(output_ndim - 1).rev() {
        let stride = if out_axis + 1 < output_axes.len() {
            let next_in_axis = output_axes[out_axis + 1];
            let mut s = var(ph::shape(next_in_axis));
            for &inner_in_axis in &output_axes[out_axis + 2..] {
                s = s * var(ph::shape(inner_in_axis));
            }
            s
        } else {
            const_int(1)
        };

        offset = var(ph::ridx(in_axis)) * stride + offset;
    }

    offset
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
    }
}

/// ネストされたループで本体をラップ
pub fn wrap_with_loops(ndim: usize, inner_body: Vec<AstNode>) -> AstNode {
    if ndim == 0 {
        return block(inner_body, Scope::new());
    }

    let mut body = block(inner_body, Scope::new());

    for axis in (0..ndim).rev() {
        body = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            var(ph::shape(axis)),
            body,
        );
    }

    block(vec![body], Scope::new())
}

/// 特定軸を除いたネストされたループで本体をラップ（スコープ付き）
pub fn wrap_with_loops_excluding_axis_with_scope(
    ndim: usize,
    exclude_axis: usize,
    inner_body: Vec<AstNode>,
    scope: Scope,
) -> AstNode {
    if ndim == 0 {
        return block(inner_body, scope);
    }

    let mut body = block(inner_body, scope);

    for axis in (0..ndim).rev() {
        if axis == exclude_axis {
            continue;
        }
        body = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            var(ph::shape(axis)),
            body,
        );
    }

    block(vec![body], Scope::new())
}
