//! カスタム関数ビルダー
//!
//! GraphOp::Custom 用の AstNode::Function を構築するヘルパー

use crate::ast::{AstNode, DType, Mutability, Scope, helper::*};
use crate::graph::ops::custom_placeholders as ph;

/// Elementwise演算の関数テンプレートを生成
///
/// 生成される関数は以下の構造を持ちます:
/// ```text
/// function kernel() {
///     for ridx0 in 0..shape0 {
///         for ridx1 in 0..shape1 {
///             ...
///             output[offset] = expr(input0[offset], input1[offset], ...)
///         }
///     }
/// }
/// ```
pub fn build_elementwise_function(ndim: usize, num_inputs: usize, expr: AstNode) -> AstNode {
    // 最内側の本体を構築: output[offset] = expr
    let offset = build_contiguous_offset(ndim);

    // 入力のロードを含む式を構築（Wildcardをloadに置換）
    let mut mappings = std::collections::HashMap::new();
    for i in 0..num_inputs {
        let load_node = load(var(ph::input(i)), offset.clone(), DType::F32);
        mappings.insert(i.to_string(), load_node);
    }
    let final_expr = expr.substitute(&mappings);

    // Store文
    let store_stmt = store(var(ph::OUTPUT), offset, final_expr);

    // ネストしたループを構築
    let body = wrap_with_loops(ndim, vec![store_stmt]);

    // 関数を作成（パラメータはlowering時に設定）
    function(
        None::<String>, // 名前はlowering時に設定
        vec![],         // パラメータはlowering時に設定
        DType::Tuple(vec![]),
        body,
    )
}

/// Reduce演算の関数テンプレートを生成
///
/// 生成される関数は以下の構造を持ちます:
/// ```text
/// function kernel() {
///     for outer_ridx... {
///         acc = initial_value
///         for reduce_ridx in 0..reduce_shape {
///             acc = reduce_op(acc, expr(inputs...))
///         }
///         output[outer_offset] = acc
///     }
/// }
/// ```
pub fn build_reduce_function(
    ndim: usize,
    num_inputs: usize,
    reduce_axis: usize,
    reduce_op: &crate::graph::ReduceOp,
    expr: AstNode,
) -> AstNode {
    use crate::graph::ReduceOp;

    // 初期値とaccumulate演算を決定
    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
        match reduce_op {
            ReduceOp::Sum => (const_f32(0.0), Box::new(|acc, val| acc + val)),
            ReduceOp::Prod => (const_f32(1.0), Box::new(|acc, val| acc * val)),
            ReduceOp::Max => (const_f32(f32::NEG_INFINITY), Box::new(max)),
        };

    // 入力のロードを含む式を構築
    let input_offset = build_contiguous_offset(ndim);
    let mut mappings = std::collections::HashMap::new();
    for i in 0..num_inputs {
        let load_node = load(var(ph::input(i)), input_offset.clone(), DType::F32);
        mappings.insert(i.to_string(), load_node);
    }
    let value_expr = expr.substitute(&mappings);

    // 出力オフセット（reduce軸を除いた次元）
    let output_offset = build_contiguous_offset_excluding_axis(ndim, reduce_axis);

    // Reduce内部ループの本体: acc = reduce_op(acc, value)
    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

    // Reduce軸のループ
    let reduce_loop = range(
        ph::ridx(reduce_axis),
        const_int(0),
        const_int(1),
        var(ph::shape(reduce_axis)),
        block(vec![acc_update], Scope::new()),
    );

    // acc初期化 + reduceループ + store
    // スコープに変数を宣言
    let mut scope = Scope::new();
    let _ = scope.declare(acc_var.to_string(), DType::F32, Mutability::Mutable);
    let acc_init = assign(acc_var, init_value);
    let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

    // 外側ループ（reduce軸以外）
    let inner_body = vec![acc_init, reduce_loop, store_stmt];
    let body = wrap_with_loops_excluding_axis_with_scope(ndim, reduce_axis, inner_body, scope);

    function(None::<String>, vec![], DType::Tuple(vec![]), body)
}

/// Cumulative演算の関数テンプレートを生成
pub fn build_cumulative_function(
    ndim: usize,
    num_inputs: usize,
    cum_axis: usize,
    cum_op: &crate::graph::CumulativeOp,
    expr: AstNode,
) -> AstNode {
    use crate::graph::CumulativeOp;

    // 初期値とaccumulate演算を決定
    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
        match cum_op {
            CumulativeOp::Sum => (const_f32(0.0), Box::new(|acc, val| acc + val)),
            CumulativeOp::Prod => (const_f32(1.0), Box::new(|acc, val| acc * val)),
        };

    // 入力のロードを含む式を構築
    let offset = build_contiguous_offset(ndim);
    let mut mappings = std::collections::HashMap::new();
    for i in 0..num_inputs {
        let load_node = load(var(ph::input(i)), offset.clone(), DType::F32);
        mappings.insert(i.to_string(), load_node);
    }
    let value_expr = expr.substitute(&mappings);

    // Cumulative内部: acc = acc op value; output[offset] = acc
    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
    let store_stmt = store(var(ph::OUTPUT), offset, var(acc_var));

    // Cumulative軸のループ
    let cum_loop = range(
        ph::ridx(cum_axis),
        const_int(0),
        const_int(1),
        var(ph::shape(cum_axis)),
        block(vec![acc_update, store_stmt], Scope::new()),
    );

    // acc初期化 + cumulativeループ
    // スコープに変数を宣言
    let mut scope = Scope::new();
    let _ = scope.declare(acc_var.to_string(), DType::F32, Mutability::Mutable);
    let acc_init = assign(acc_var, init_value);

    // 外側ループ（cumulative軸以外、累積軸より後の軸は内側）
    let inner_body = vec![acc_init, cum_loop];
    let body = wrap_with_loops_excluding_axis_with_scope(ndim, cum_axis, inner_body, scope);

    function(None::<String>, vec![], DType::Tuple(vec![]), body)
}

// ============================================================================
// ヘルパー関数
// ============================================================================

/// Contiguousレイアウトのオフセット計算式を生成
///
/// offset = ridx0 * (shape1 * shape2 * ...) + ridx1 * (shape2 * ...) + ... + ridxN
fn build_contiguous_offset(ndim: usize) -> AstNode {
    if ndim == 0 {
        return const_int(0);
    }

    let mut offset = var(ph::ridx(ndim - 1));

    for axis in (0..ndim - 1).rev() {
        // stride = shape[axis+1] * shape[axis+2] * ...
        let mut stride = var(ph::shape(axis + 1));
        for inner_axis in (axis + 2)..ndim {
            stride = stride * var(ph::shape(inner_axis));
        }
        offset = var(ph::ridx(axis)) * stride + offset;
    }

    offset
}

/// 指定軸を除いたContiguousオフセット計算
fn build_contiguous_offset_excluding_axis(ndim: usize, exclude_axis: usize) -> AstNode {
    if ndim <= 1 {
        return const_int(0);
    }

    let output_ndim = ndim - 1;
    if output_ndim == 0 {
        return const_int(0);
    }

    // 出力の軸インデックスを計算（exclude_axisをスキップ）
    let mut output_axes = Vec::new();
    for axis in 0..ndim {
        if axis != exclude_axis {
            output_axes.push(axis);
        }
    }

    // 出力形状を使ってオフセットを計算
    let mut offset = var(ph::ridx(output_axes[output_ndim - 1]));

    for (out_axis, &in_axis) in output_axes.iter().enumerate().take(output_ndim - 1).rev() {
        // 出力のstride計算
        let stride = if out_axis + 1 < output_axes.len() {
            // 次の出力軸の形状
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

/// ネストしたループで文をラップ
fn wrap_with_loops(ndim: usize, inner_body: Vec<AstNode>) -> AstNode {
    if ndim == 0 {
        return block(inner_body, Scope::new());
    }

    let mut body = block(inner_body, Scope::new());

    // 内側から外側へループを構築
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

/// 指定軸を除いたネストループ（スコープ指定あり）
fn wrap_with_loops_excluding_axis_with_scope(
    ndim: usize,
    exclude_axis: usize,
    inner_body: Vec<AstNode>,
    scope: Scope,
) -> AstNode {
    if ndim == 0 {
        return block(inner_body, scope);
    }

    let mut body = block(inner_body, scope);

    // 内側から外側へループを構築（exclude_axisをスキップ）
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::wildcard;

    #[test]
    fn test_build_elementwise_function() {
        // x + y の関数を生成
        let expr = wildcard("0") + wildcard("1");
        let func = build_elementwise_function(2, 2, expr);

        // 関数であることを確認
        assert!(matches!(func, AstNode::Function { .. }));
    }

    #[test]
    fn test_build_reduce_function() {
        use crate::graph::ReduceOp;

        // sum(x * y, axis=1) の関数を生成
        let expr = wildcard("0") * wildcard("1");
        let func = build_reduce_function(2, 2, 1, &ReduceOp::Sum, expr);

        assert!(matches!(func, AstNode::Function { .. }));
    }
}
