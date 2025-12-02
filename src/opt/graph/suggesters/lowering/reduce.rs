//! Reduce/Cumulative演算のLowering
//!
//! Reduce、Cumulative、FusedElementwiseReduce、FusedElementwiseCumulative演算の
//! AST関数生成を担当します。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{CumulativeOp, GraphNode, GraphOp, ReduceOp};
use std::collections::HashMap;

use super::helpers::{
    build_contiguous_offset, build_contiguous_offset_excluding_axis, get_reduce_init,
    graph_dtype_to_ast, wrap_with_loops_excluding_axis_with_scope,
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

    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) = match op
    {
        ReduceOp::Sum => (
            get_reduce_init(&node.dtype, op),
            Box::new(|acc, val| acc + val),
        ),
        ReduceOp::Prod => (
            get_reduce_init(&node.dtype, op),
            Box::new(|acc, val| acc * val),
        ),
        ReduceOp::Max => (get_reduce_init(&node.dtype, op), Box::new(max)),
    };

    let input_offset = build_contiguous_offset(ndim);
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let value_expr = load(var(ph::input(0)), input_offset, load_dtype);

    let output_offset = build_contiguous_offset_excluding_axis(ndim, axis);

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

    let reduce_loop = range(
        ph::ridx(axis),
        const_int(0),
        const_int(1),
        var(ph::shape(axis)),
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
    let body = wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

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
    let ndim = input.view.shape().len();

    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) = match op
    {
        CumulativeOp::Sum => (const_f32(0.0), Box::new(|acc, val| acc + val)),
        CumulativeOp::Prod => (const_f32(1.0), Box::new(|acc, val| acc * val)),
    };

    let offset = build_contiguous_offset(ndim);
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let value_expr = load(var(ph::input(0)), offset.clone(), load_dtype);

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
    let store_stmt = store(var(ph::OUTPUT), offset, var(acc_var));

    let cum_loop = range(
        ph::ridx(axis),
        const_int(0),
        const_int(1),
        var(ph::shape(axis)),
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
    let body = wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// FusedElementwiseReduce演算の関数を生成
pub fn build_fused_elementwise_reduce_function(
    node: &GraphNode,
    expr: &AstNode,
    reduce_op: &ReduceOp,
    axis: usize,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();
    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
        match reduce_op {
            ReduceOp::Sum => (
                get_reduce_init(&node.dtype, reduce_op),
                Box::new(|acc, val| acc + val),
            ),
            ReduceOp::Prod => (
                get_reduce_init(&node.dtype, reduce_op),
                Box::new(|acc, val| acc * val),
            ),
            ReduceOp::Max => (get_reduce_init(&node.dtype, reduce_op), Box::new(max)),
        };

    // 入力のロードを含む式を構築
    let input_offset = build_contiguous_offset(ndim);
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

    let output_offset = build_contiguous_offset_excluding_axis(ndim, axis);

    let acc_var = "acc";
    let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

    let reduce_loop = range(
        ph::ridx(axis),
        const_int(0),
        const_int(1),
        var(ph::shape(axis)),
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
    let body = wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

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
    let ndim = input.view.shape().len();

    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
        match cum_op {
            CumulativeOp::Sum => (const_f32(0.0), Box::new(|acc, val| acc + val)),
            CumulativeOp::Prod => (const_f32(1.0), Box::new(|acc, val| acc * val)),
        };

    let offset = build_contiguous_offset(ndim);
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
        var(ph::shape(axis)),
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
    let body = wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}
