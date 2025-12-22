//! Reduce/Cumulative演算のLowering
//!
//! Reduce、Cumulative、FusedElementwiseReduce、FusedElementwiseCumulative演算の
//! AST関数生成を担当します。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{CumulativeOp, GraphNode, GraphOp, ReduceOp};
use std::collections::HashMap;

use super::helpers::{
    build_contiguous_offset_excluding_axes_with_shape,
    build_contiguous_offset_excluding_axis_with_shape, build_contiguous_offset_with_shape,
    build_cumulative_accumulator, build_reduce_accumulator, build_strided_offset,
    graph_dtype_to_ast, shape_dim_to_ast, wrap_with_loops_excluding_axes_with_scope_and_shape,
    wrap_with_loops_excluding_axis_with_scope_and_shape,
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

    // 入力のViewに基づいてオフセットを計算
    let input_offset = build_strided_offset(&input.view, ndim);
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let value_expr = load(var(ph::input(0)), input_offset, load_dtype);

    // 出力は連続メモリ配置
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

    // 入力のViewに基づいてオフセットを計算
    let input_offset = build_strided_offset(&input.view, ndim);
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let value_expr = load(var(ph::input(0)), input_offset, load_dtype);

    // 出力は連続メモリ配置
    let output_offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
    let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

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

    // 入力のロードを含む式を構築（各入力のViewを使用）
    let load_dtype = graph_dtype_to_ast(&input.dtype);

    let mut mappings = HashMap::new();
    let mut non_const_idx = 0;
    for (i, src) in node.src.iter().enumerate() {
        if let GraphOp::Const(lit) = &src.op {
            mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
        } else {
            // 各入力のViewに基づいてオフセットを計算
            let src_offset = build_strided_offset(&src.view, ndim);
            let load_node = load(
                var(ph::input(non_const_idx)),
                src_offset,
                load_dtype.clone(),
            );
            mappings.insert(i.to_string(), load_node);
            non_const_idx += 1;
        }
    }
    let value_expr = expr.substitute(&mappings);

    // 出力は連続メモリ配置
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

    // 出力は連続メモリ配置
    let output_offset = build_contiguous_offset_with_shape(ndim, Some(input_shape));
    let load_dtype = graph_dtype_to_ast(&input.dtype);

    // 入力のロードを含む式を構築（各入力のViewを使用）
    let mut mappings = HashMap::new();
    let mut non_const_idx = 0;
    for (i, src) in node.src.iter().enumerate() {
        if let GraphOp::Const(lit) = &src.op {
            mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
        } else {
            // 各入力のViewに基づいてオフセットを計算
            let src_offset = build_strided_offset(&src.view, ndim);
            let load_node = load(
                var(ph::input(non_const_idx)),
                src_offset,
                load_dtype.clone(),
            );
            mappings.insert(i.to_string(), load_node);
            non_const_idx += 1;
        }
    }
    let value_expr = expr.substitute(&mappings);

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
    let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

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
