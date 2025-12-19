//! その他の演算のLowering
//!
//! Contiguous、Slice、Cast、複素数演算、Rand、Arangeなどの
//! AST関数生成を担当します。

use crate::ast::{AstNode, DType as AstDType, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{DType as GraphDType, GraphNode};

use super::helpers::{
    build_contiguous_offset, build_strided_offset, build_strided_offset_with_sources,
    graph_dtype_to_ast, wrap_with_loops,
};

/// Contiguous演算の関数を生成
///
/// LoadIndexを含むView（Gather等）にも対応しています。
pub fn build_contiguous_function(node: &GraphNode, name: &str) -> Option<AstNode> {
    let input = node.src.first()?;
    let shape = node.view.shape();
    let ndim = shape.len();
    let load_dtype = graph_dtype_to_ast(&input.dtype);

    // 入力のオフセット計算（Viewを考慮）
    // LoadIndexを含む場合は全srcのバッファ変数名を渡す
    let input_offset = if input.view.contains_load_index() {
        // srcの各バッファに対応する変数名を生成
        let src_vars: Vec<String> = (0..node.src.len()).map(ph::input).collect();
        build_strided_offset_with_sources(&input.view, ndim, &src_vars, load_dtype.clone())
    } else {
        build_strided_offset(&input.view, ndim)
    };

    let output_offset = build_contiguous_offset(ndim);

    let load_expr = load(var(ph::input(0)), input_offset, load_dtype);
    let store_stmt = store(var(ph::OUTPUT), output_offset, load_expr);

    let body = wrap_with_loops(ndim, vec![store_stmt]);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// Pad演算の関数を生成
/// 注意: 現在のASTには条件式がないため、Padのloweringは未サポート
pub fn build_pad_function(
    _node: &GraphNode,
    _padding: &[(usize, usize)],
    _value: f32,
) -> Option<AstNode> {
    // TODO: ASTに条件式を追加したら実装
    None
}

/// Slice演算の関数を生成
pub fn build_slice_function(
    node: &GraphNode,
    ranges: &[(usize, usize)],
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let shape = node.view.shape();
    let ndim = shape.len();

    // 出力のオフセット
    let output_offset = build_contiguous_offset(ndim);

    // 入力のオフセット（スライス開始位置を考慮）
    let mut input_offset_parts = Vec::new();
    for (axis, &(start, _)) in ranges.iter().enumerate().take(ndim) {
        let idx = var(ph::ridx(axis)) + const_int(start as isize);
        input_offset_parts.push(idx);
    }

    // ストライドを計算して入力オフセットを構築
    let input_shape = input.view.shape();
    let mut input_offset = input_offset_parts[ndim - 1].clone();
    for axis in (0..ndim - 1).rev() {
        let mut stride: AstNode = input_shape[axis + 1].clone().into();
        for dim in input_shape.iter().take(ndim).skip(axis + 2) {
            let s: AstNode = dim.clone().into();
            stride = stride * s;
        }
        input_offset = input_offset_parts[axis].clone() * stride + input_offset;
    }

    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let load_expr = load(var(ph::input(0)), input_offset, load_dtype);
    let store_stmt = store(var(ph::OUTPUT), output_offset, load_expr);

    let body = wrap_with_loops(ndim, vec![store_stmt]);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// Concat演算の関数を生成
/// 注意: 現在のASTには条件式がないため、Concatのloweringは未サポート
pub fn build_concat_function(_node: &GraphNode, _axis: usize) -> Option<AstNode> {
    // TODO: ASTに条件式を追加したら実装
    None
}

/// Rand演算の関数を生成
pub fn build_rand_function(node: &GraphNode, name: &str) -> Option<AstNode> {
    let shape = node.view.shape();
    let ndim = shape.len();

    let offset = build_contiguous_offset(ndim);

    // 簡易的な乱数生成（実際にはシード管理が必要）
    // ここではプレースホルダーとしてrand()呼び出しを生成
    let rand_expr = AstNode::Call {
        name: "rand_f32".to_string(),
        args: vec![],
    };

    let store_stmt = store(var(ph::OUTPUT), offset, rand_expr);
    let body = wrap_with_loops(ndim, vec![store_stmt]);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// Arange演算の関数を生成
pub fn build_arange_function(node: &GraphNode, name: &str) -> Option<AstNode> {
    let shape = node.view.shape();
    let ndim = shape.len();

    if ndim != 1 {
        return None; // Arangeは1次元のみ
    }

    let idx = var(ph::ridx(0));
    let offset = idx.clone();

    // 型に応じてキャスト
    let value = match &node.dtype {
        GraphDType::I32 => idx,
        GraphDType::F32 => cast(idx, AstDType::F32),
        _ => return None,
    };

    let store_stmt = store(var(ph::OUTPUT), offset, value);
    let body = wrap_with_loops(ndim, vec![store_stmt]);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// Cast演算の関数を生成
pub fn build_cast_function(
    node: &GraphNode,
    target_dtype: &GraphDType,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let shape = node.view.shape();
    let ndim = shape.len();

    let offset = build_contiguous_offset(ndim);
    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let target_ast_dtype = graph_dtype_to_ast(target_dtype);

    let load_expr = load(var(ph::input(0)), offset.clone(), load_dtype);
    let cast_expr = cast(load_expr, target_ast_dtype);
    let store_stmt = store(var(ph::OUTPUT), offset, cast_expr);

    let body = wrap_with_loops(ndim, vec![store_stmt]);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}
