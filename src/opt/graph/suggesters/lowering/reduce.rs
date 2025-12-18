//! Reduce/Cumulative演算のLowering
//!
//! Reduce、Cumulative、FusedElementwiseReduce、FusedElementwiseCumulative演算の
//! AST関数生成を担当します。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{CumulativeOp, GraphNode, GraphOp, ReduceOp};
use std::collections::HashMap;

use crate::graph::shape::Expr;

use super::helpers::{
    build_contiguous_offset_excluding_axes_with_shape,
    build_contiguous_offset_excluding_axis_with_shape, build_contiguous_offset_with_shape,
    build_cumulative_accumulator, build_reduce_accumulator, build_reduce_axis_stride,
    build_unrolled_reduce_loop, graph_dtype_to_ast, shape_dim_to_ast,
    wrap_with_loops_excluding_axes_with_scope_and_shape,
    wrap_with_loops_excluding_axis_with_scope_and_shape, wrap_with_simd_innermost_loop,
};

/// Reduce演算の関数を生成
pub fn build_reduce_function(
    node: &GraphNode,
    op: &ReduceOp,
    axis: usize,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    let (init_value, accumulate_fn) = build_reduce_accumulator(op, &node.dtype);

    let input_offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let value_expr = load(var(ph::input(0)), input_offset, load_dtype);

    let output_offset =
        build_contiguous_offset_excluding_axis_with_shape(ndim, axis, Some(input_shape));

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

    let reduce_loop = range(
        ph::ridx(axis),
        const_int(0),
        const_int(1),
        shape_dim_to_ast(Some(input_shape), axis),
        block(vec![acc_update], Scope::new()),
    );

    let mut scope = Scope::new();
    let _ = scope.declare(
        acc_var.to_string(),
        graph_dtype_to_ast(&node.dtype),
        Mutability::Mutable,
    );
    let acc_init = assign(acc_var, init_value);
    let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

    let inner_body = vec![acc_init, reduce_loop, store_stmt];
    let body = wrap_with_loops_excluding_axis_with_scope_and_shape(
        ndim,
        axis,
        inner_body,
        scope,
        Some(input_shape),
    );

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// Cumulative演算の関数を生成
pub fn build_cumulative_function(
    node: &GraphNode,
    op: &CumulativeOp,
    axis: usize,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    let (init_value, accumulate_fn) = build_cumulative_accumulator(op, &node.dtype);

    let offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let value_expr = load(var(ph::input(0)), offset.clone(), load_dtype);

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
    let store_stmt = store(var(ph::OUTPUT), offset, var(acc_var));

    let cum_loop = range(
        ph::ridx(axis),
        const_int(0),
        const_int(1),
        shape_dim_to_ast(Some(input_shape), axis),
        block(vec![acc_update, store_stmt], Scope::new()),
    );

    let mut scope = Scope::new();
    let _ = scope.declare(
        acc_var.to_string(),
        graph_dtype_to_ast(&node.dtype),
        Mutability::Mutable,
    );
    let acc_init = assign(acc_var, init_value);

    let inner_body = vec![acc_init, cum_loop];
    let body = wrap_with_loops_excluding_axis_with_scope_and_shape(
        ndim,
        axis,
        inner_body,
        scope,
        Some(input_shape),
    );

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// FusedElementwiseReduce演算の関数を生成（複数軸対応）
pub fn build_fused_elementwise_reduce_function(
    node: &GraphNode,
    expr: &AstNode,
    reduce_op: &ReduceOp,
    axes: &[usize],
    name: &str,
) -> Option<AstNode> {
    if axes.is_empty() {
        return None;
    }

    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();
    let (init_value, accumulate_fn) = build_reduce_accumulator(reduce_op, &node.dtype);

    // 入力のロードを含む式を構築
    let input_offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);

    let mut mappings = HashMap::new();
    let mut non_const_idx = 0;
    for (i, src) in node.src.iter().enumerate() {
        if let GraphOp::Const(lit) = &src.op {
            mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
        } else {
            let load_node = load(
                var(ph::input(non_const_idx)),
                input_offset.clone(),
                load_dtype.clone(),
            );
            mappings.insert(i.to_string(), load_node);
            non_const_idx += 1;
        }
    }
    let value_expr = expr.substitute(&mappings);

    let output_offset =
        build_contiguous_offset_excluding_axes_with_shape(ndim, axes, Some(input_shape));

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

    // 複数軸の縮約ループを生成（内側から外側へネスト）
    let mut reduce_loops = block(vec![acc_update], Scope::new());
    for &axis in axes.iter().rev() {
        reduce_loops = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            shape_dim_to_ast(Some(input_shape), axis),
            reduce_loops,
        );
    }

    let mut scope = Scope::new();
    let _ = scope.declare(
        acc_var.to_string(),
        graph_dtype_to_ast(&node.dtype),
        Mutability::Mutable,
    );
    let acc_init = assign(acc_var, init_value);
    let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

    let inner_body = vec![acc_init, reduce_loops, store_stmt];
    let body = wrap_with_loops_excluding_axes_with_scope_and_shape(
        ndim,
        axes,
        inner_body,
        scope,
        Some(input_shape),
    );

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// FusedElementwiseCumulative演算の関数を生成
pub fn build_fused_elementwise_cumulative_function(
    node: &GraphNode,
    expr: &AstNode,
    cum_op: &CumulativeOp,
    axis: usize,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    let (init_value, accumulate_fn) = build_cumulative_accumulator(cum_op, &node.dtype);

    let offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);

    let mut mappings = HashMap::new();
    let mut non_const_idx = 0;
    for (i, src) in node.src.iter().enumerate() {
        if let GraphOp::Const(lit) = &src.op {
            mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
        } else {
            let load_node = load(
                var(ph::input(non_const_idx)),
                offset.clone(),
                load_dtype.clone(),
            );
            mappings.insert(i.to_string(), load_node);
            non_const_idx += 1;
        }
    }
    let value_expr = expr.substitute(&mappings);

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
    let store_stmt = store(var(ph::OUTPUT), offset, var(acc_var));

    let cum_loop = range(
        ph::ridx(axis),
        const_int(0),
        const_int(1),
        shape_dim_to_ast(Some(input_shape), axis),
        block(vec![acc_update, store_stmt], Scope::new()),
    );

    let mut scope = Scope::new();
    let _ = scope.declare(
        acc_var.to_string(),
        graph_dtype_to_ast(&node.dtype),
        Mutability::Mutable,
    );
    let acc_init = assign(acc_var, init_value);

    let inner_body = vec![acc_init, cum_loop];
    let body = wrap_with_loops_excluding_axis_with_scope_and_shape(
        ndim,
        axis,
        inner_body,
        scope,
        Some(input_shape),
    );

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

// ============================================================
// SIMD版関数生成
// ============================================================

/// FusedElementwiseReduce演算のSIMD版関数を生成
///
/// 縮約軸が最内軸を含まない場合のみSIMD化が可能です。
/// 出力テンソルの最内軸をSIMD化します。
pub fn build_fused_elementwise_reduce_function_simd(
    node: &GraphNode,
    expr: &AstNode,
    reduce_op: &ReduceOp,
    axes: &[usize],
    name: &str,
    simd_width: usize,
) -> Option<AstNode> {
    if axes.is_empty() {
        return None;
    }

    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    // 縮約軸が最内軸を含む場合はSIMD化しない（水平加算が必要になるため）
    let innermost_axis = ndim.saturating_sub(1);
    if axes.contains(&innermost_axis) {
        return None;
    }

    // 出力の最内軸サイズをチェック
    let output_shape = node.view.shape();
    if output_shape.is_empty() {
        return None;
    }
    if let Some(Expr::Const(size)) = output_shape.last()
        && (*size as usize) < simd_width
    {
        return None;
    }

    let (init_value, accumulate_fn) = build_reduce_accumulator(reduce_op, &node.dtype);

    let input_offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let scalar_dtype = graph_dtype_to_ast(&node.dtype);
    let vec_dtype = scalar_dtype.clone().to_vec(simd_width);

    // SIMD版の縮約ループを構築
    let simd_reduce_body = {
        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;
        for (i, src) in node.src.iter().enumerate() {
            if let GraphOp::Const(lit) = &src.op {
                let scalar_const = AstNode::Const(lit.clone());
                let vec_const = broadcast(scalar_const, simd_width);
                mappings.insert(i.to_string(), vec_const);
            } else {
                let load_node = load_vec(
                    var(ph::input(non_const_idx)),
                    input_offset.clone(),
                    simd_width,
                    vec_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                non_const_idx += 1;
            }
        }
        let value_expr = expr.substitute(&mappings);

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

        let mut reduce_loops = block(vec![acc_update], Scope::new());
        for &axis in axes.iter().rev() {
            reduce_loops = range(
                ph::ridx(axis),
                const_int(0),
                const_int(1),
                shape_dim_to_ast(Some(input_shape), axis),
                reduce_loops,
            );
        }

        let output_offset =
            build_contiguous_offset_excluding_axes_with_shape(ndim, axes, Some(input_shape));

        let vec_init = broadcast(init_value.clone(), simd_width);
        let acc_init = assign(acc_var, vec_init);
        let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

        vec![acc_init, reduce_loops, store_stmt]
    };

    // スカラー版の縮約ループを構築（テール処理用）
    let scalar_reduce_body = {
        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;
        for (i, src) in node.src.iter().enumerate() {
            if let GraphOp::Const(lit) = &src.op {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            } else {
                let load_node = load(
                    var(ph::input(non_const_idx)),
                    input_offset.clone(),
                    load_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                non_const_idx += 1;
            }
        }
        let value_expr = expr.substitute(&mappings);

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

        let mut reduce_loops = block(vec![acc_update], Scope::new());
        for &axis in axes.iter().rev() {
            reduce_loops = range(
                ph::ridx(axis),
                const_int(0),
                const_int(1),
                shape_dim_to_ast(Some(input_shape), axis),
                reduce_loops,
            );
        }

        let output_offset =
            build_contiguous_offset_excluding_axes_with_shape(ndim, axes, Some(input_shape));

        let acc_init = assign(acc_var, init_value);
        let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

        vec![acc_init, reduce_loops, store_stmt]
    };

    let output_ndim = output_shape.len();
    let body = wrap_with_simd_innermost_loop(
        output_ndim,
        simd_reduce_body,
        scalar_reduce_body,
        Some(output_shape),
        simd_width,
    );

    let mut outer_scope = Scope::new();
    let _ = outer_scope.declare("acc".to_string(), vec_dtype, Mutability::Mutable);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        block(vec![body], outer_scope),
    ))
}

// ============================================================
// ループアンローリング版関数生成
// ============================================================

/// Reduce演算のアンロール版関数を生成
///
/// 縮約軸のループを指定されたファクターで展開します。
/// 端数は別のテールループで処理されます。
///
/// # Arguments
/// * `node` - Reduceノード
/// * `op` - Reduce演算の種類
/// * `axis` - 縮約軸
/// * `name` - 関数名
/// * `unroll_factor` - アンロールファクター（例: 4, 8）
pub fn build_reduce_function_unrolled(
    node: &GraphNode,
    op: &ReduceOp,
    axis: usize,
    name: &str,
    unroll_factor: usize,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    // アンロールファクターが縮約軸サイズより大きい場合はスキップ
    if let Some(Expr::Const(axis_size)) = input_shape.get(axis)
        && (*axis_size as usize) < unroll_factor
    {
        return None;
    }

    let (init_value, accumulate_fn) = build_reduce_accumulator(op, &node.dtype);

    let input_offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let stride = build_reduce_axis_stride(axis, Some(input_shape));

    // 値ローダー：offset_deltaを受け取り、loadノードを返す
    let value_loader = |offset_delta: AstNode| {
        load(
            var(ph::input(0)),
            input_offset.clone() + offset_delta * stride.clone(),
            load_dtype.clone(),
        )
    };

    let reduce_loops = build_unrolled_reduce_loop(
        axis,
        unroll_factor,
        input_shape,
        &accumulate_fn,
        value_loader,
    );

    let output_offset =
        build_contiguous_offset_excluding_axis_with_shape(ndim, axis, Some(input_shape));

    let mut scope = Scope::new();
    let _ = scope.declare(
        "acc".to_string(),
        graph_dtype_to_ast(&node.dtype),
        Mutability::Mutable,
    );
    let acc_init = assign("acc", init_value);
    let store_stmt = store(var(ph::OUTPUT), output_offset, var("acc"));

    let inner_body = vec![acc_init, reduce_loops, store_stmt];
    let body = wrap_with_loops_excluding_axis_with_scope_and_shape(
        ndim,
        axis,
        inner_body,
        scope,
        Some(input_shape),
    );

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// FusedElementwiseReduce演算のアンロール版関数を生成
///
/// 縮約軸のループを指定されたファクターで展開します。
/// 複数軸の縮約には対応していません（最初の縮約軸のみアンロール）。
///
/// # Arguments
/// * `node` - FusedElementwiseReduceノード
/// * `expr` - 演算式
/// * `reduce_op` - Reduce演算の種類
/// * `axes` - 縮約軸のリスト
/// * `name` - 関数名
/// * `unroll_factor` - アンロールファクター（例: 4, 8）
pub fn build_fused_elementwise_reduce_function_unrolled(
    node: &GraphNode,
    expr: &AstNode,
    reduce_op: &ReduceOp,
    axes: &[usize],
    name: &str,
    unroll_factor: usize,
) -> Option<AstNode> {
    if axes.is_empty() {
        return None;
    }

    // 複数軸の場合は最内の縮約軸のみアンロール
    let unroll_axis = *axes.last()?;

    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    // アンロールファクターが縮約軸サイズより大きい場合はスキップ
    if let Some(Expr::Const(axis_size)) = input_shape.get(unroll_axis)
        && (*axis_size as usize) < unroll_factor
    {
        return None;
    }

    let (init_value, accumulate_fn) = build_reduce_accumulator(reduce_op, &node.dtype);

    let input_offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let stride = build_reduce_axis_stride(unroll_axis, Some(input_shape));

    // 入力のロードを含む式を構築するための関数
    let build_value_expr = |offset_delta: AstNode| {
        let adjusted_offset = input_offset.clone() + offset_delta * stride.clone();
        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;
        for (i, src) in node.src.iter().enumerate() {
            if let GraphOp::Const(lit) = &src.op {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            } else {
                let load_node = load(
                    var(ph::input(non_const_idx)),
                    adjusted_offset.clone(),
                    load_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                non_const_idx += 1;
            }
        }
        expr.substitute(&mappings)
    };

    // アンロール版縮約ループを生成
    let reduce_loops = build_unrolled_reduce_loop(
        unroll_axis,
        unroll_factor,
        input_shape,
        &accumulate_fn,
        build_value_expr,
    );

    // アンロールしない外側の縮約軸のループでラップ
    let mut outer_reduce_loops = reduce_loops;
    for &axis in axes.iter().rev() {
        if axis == unroll_axis {
            continue;
        }
        outer_reduce_loops = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            shape_dim_to_ast(Some(input_shape), axis),
            outer_reduce_loops,
        );
    }

    let output_offset =
        build_contiguous_offset_excluding_axes_with_shape(ndim, axes, Some(input_shape));

    let mut scope = Scope::new();
    let _ = scope.declare(
        "acc".to_string(),
        graph_dtype_to_ast(&node.dtype),
        Mutability::Mutable,
    );
    let acc_init = assign("acc", init_value);
    let store_stmt = store(var(ph::OUTPUT), output_offset, var("acc"));

    let inner_body = vec![acc_init, outer_reduce_loops, store_stmt];
    let body = wrap_with_loops_excluding_axes_with_scope_and_shape(
        ndim,
        axes,
        inner_body,
        scope,
        Some(input_shape),
    );

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}
