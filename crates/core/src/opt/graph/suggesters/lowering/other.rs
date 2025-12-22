//! その他の演算のLowering
//!
//! Contiguous、Slice、Cast、複素数演算、Rand、Arangeなどの
//! AST関数生成を担当します。

use crate::ast::{AstNode, DType as AstDType, Literal, helper::*};
use crate::graph::ops::{GraphOp, custom_placeholders as ph};
use crate::graph::shape::Expr;
use crate::graph::{DType as GraphDType, GraphNode};

use super::helpers::{
    build_contiguous_offset, build_offset_from_coords_with_view, build_strided_offset,
    build_strided_offset_with_sources, graph_dtype_to_ast, wrap_with_loops,
};

/// Contiguous演算の関数を生成
///
/// LoadIndexを含むView（Gather等）にも対応しています。
/// また、入力がConstノードの場合は定数値を直接展開します。
pub fn build_contiguous_function(node: &GraphNode, name: &str) -> Option<AstNode> {
    let input = node.src.first()?;
    let shape = node.view.shape();
    let ndim = shape.len();

    // 入力がConstノードの場合、またはViewチェーンを辿ってConstが見つかる場合は定数値を直接展開
    if let Some(literal) = find_const_in_view_chain(input) {
        let output_dtype = graph_dtype_to_ast(&node.dtype);
        return build_contiguous_from_const(&literal, shape, name, output_dtype);
    }

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

/// Viewチェーンを辿ってConstノードを探す
///
/// View操作（unsqueeze, broadcast_to等）を通じてConstノードに行き着く場合、
/// そのリテラル値を返します。それ以外の場合はNoneを返します。
fn find_const_in_view_chain(node: &GraphNode) -> Option<Literal> {
    match &node.op {
        GraphOp::Const(literal) => Some(literal.clone()),
        GraphOp::View(_) => {
            // View操作は入力ノードを1つ持つ
            node.src.first().and_then(find_const_in_view_chain)
        }
        _ => None,
    }
}

/// Constノードからのcontiguous関数を生成
///
/// 定数値を全要素に展開します。
/// shape を具体的に埋め込むため、プレースホルダー変数（shape0, shape1等）は使用しません。
fn build_contiguous_from_const(
    literal: &Literal,
    shape: &[Expr],
    name: &str,
    output_dtype: AstDType,
) -> Option<AstNode> {
    use super::helpers::{build_contiguous_offset_with_shape, wrap_with_loops_with_shape};
    use crate::ast::{Mutability, VarDecl, VarKind};

    let ndim = shape.len();

    // 具体的な shape を使用してオフセットを計算
    let output_offset = build_contiguous_offset_with_shape(ndim, Some(shape));

    // リテラルをAstNodeに変換
    let const_expr = AstNode::Const(literal.clone());

    let store_stmt = store(var(ph::OUTPUT), output_offset, const_expr);

    // 具体的な shape を使用してループを生成
    let body = wrap_with_loops_with_shape(ndim, vec![store_stmt], Some(shape));

    // 出力バッファパラメータを正しい型で追加
    let output_param = VarDecl {
        name: ph::OUTPUT.to_string(),
        dtype: AstDType::Ptr(Box::new(output_dtype)),
        mutability: Mutability::Mutable,
        kind: VarKind::Normal,
    };

    Some(function(
        Some(name.to_string()),
        vec![output_param],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// Pad演算の関数を生成
///
/// パディング領域には指定値を、それ以外は入力テンソルの値をコピーします。
/// 動的shapeに対応しており、パディング量にExprを使用可能です。
/// 入力のView（転置等）を考慮してオフセットを計算します。
///
/// 生成されるカーネル:
/// ```text
/// for ridx0 in 0..output_shape[0]:
///     for ridx1 in 0..output_shape[1]:
///         src_idx0 = ridx0 - padding[0].before
///         src_idx1 = ridx1 - padding[1].before
///         if src_idx0 >= 0 && src_idx0 < input_shape[0]
///            && src_idx1 >= 0 && src_idx1 < input_shape[1]:
///             output[out_offset] = input[in_offset]  // in_offsetはView考慮
///         else:
///             output[out_offset] = pad_value
/// ```
pub fn build_pad_function(
    node: &GraphNode,
    padding: &[(Expr, Expr)],
    value: f32,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let output_shape = node.view.shape();
    let ndim = output_shape.len();
    let load_dtype = graph_dtype_to_ast(&input.dtype);

    // 出力オフセット（連続配置）
    let output_offset = build_contiguous_offset(ndim);

    // 各軸の入力座標と境界チェック条件を構築
    let mut conditions: Vec<AstNode> = Vec::new();
    let mut src_coords: Vec<AstNode> = Vec::new();

    for axis in 0..ndim {
        let ridx = var(ph::ridx(axis));
        let before: AstNode = padding[axis].0.clone().into();

        // 入力座標: src_idx = ridx - padding.before
        let src_coord = ridx - before;

        // 条件1: src_idx >= 0
        conditions.push(ge(src_coord.clone(), const_int(0)));

        // 条件2: src_idx < input_shape[axis]
        let dim_size: AstNode = input_shape[axis].clone().into();
        conditions.push(lt(src_coord.clone(), dim_size));

        src_coords.push(src_coord);
    }

    // 入力オフセットを計算（入力のViewを考慮）
    let input_offset = build_offset_from_coords_with_view(&src_coords, &input.view);

    // 境界内: 入力から値をロード
    let load_expr = load(var(ph::input(0)), input_offset, load_dtype);
    let store_load = store(var(ph::OUTPUT), output_offset.clone(), load_expr);

    // 境界外: パディング値を格納
    let store_pad = store(var(ph::OUTPUT), output_offset, const_f32(value));

    // 条件を結合（すべての条件がtrueの場合のみコピー）
    let combined_condition = conditions
        .into_iter()
        .reduce(|a, b| AstNode::BitwiseAnd(Box::new(a), Box::new(b)))
        .expect("conditions should not be empty");

    // if-then-else で分岐
    let if_stmt = if_then_else(combined_condition, store_load, store_pad);

    let body = wrap_with_loops(ndim, vec![if_stmt]);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}

/// 座標リストから入力オフセットを計算（row-major順、連続配置前提）
///
/// 注意: この関数は連続メモリ配置を前提としています。
/// 非連続Viewを考慮する場合は`build_offset_from_coords_with_view`を使用してください。
#[allow(dead_code)]
fn build_input_offset_from_coords(coords: &[AstNode], shape: &[Expr]) -> AstNode {
    let ndim = coords.len();
    if ndim == 0 {
        return const_int(0);
    }

    // Row-major: offset = coords[0] * stride[0] + coords[1] * stride[1] + ...
    // stride[i] = shape[i+1] * shape[i+2] * ... * shape[ndim-1]
    let mut offset = coords[ndim - 1].clone();

    for axis in (0..ndim - 1).rev() {
        // stride = shape[axis+1] * shape[axis+2] * ... * shape[ndim-1]
        let mut stride: AstNode = shape[axis + 1].clone().into();
        for dim in shape.iter().take(ndim).skip(axis + 2) {
            let s: AstNode = dim.clone().into();
            stride = stride * s;
        }
        offset = coords[axis].clone() * stride + offset;
    }

    offset
}

/// Slice演算の関数を生成
///
/// 入力のView（転置等）を考慮してオフセットを計算します。
pub fn build_slice_function(
    node: &GraphNode,
    ranges: &[(usize, usize)],
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let shape = node.view.shape();
    let ndim = shape.len();

    // 出力のオフセット（連続配置）
    let output_offset = build_contiguous_offset(ndim);

    // 入力座標を構築（スライス開始位置を考慮）
    let mut src_coords = Vec::new();
    for (axis, &(start, _)) in ranges.iter().enumerate().take(ndim) {
        let coord = var(ph::ridx(axis)) + const_int(start as i64);
        src_coords.push(coord);
    }

    // 入力オフセットを計算（入力のViewを考慮）
    let input_offset = build_offset_from_coords_with_view(&src_coords, &input.view);

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
///
/// 入力のView（転置等）を考慮してオフセットを計算します。
pub fn build_cast_function(
    node: &GraphNode,
    target_dtype: &GraphDType,
    name: &str,
) -> Option<AstNode> {
    use super::helpers::{build_contiguous_offset_with_shape, wrap_with_loops_with_shape};

    let input = node.src.first()?;
    let shape = node.view.shape();
    let ndim = shape.len();

    // 入力のViewに基づいてオフセットを計算
    let input_offset = build_strided_offset(&input.view, ndim);
    // 出力は連続メモリ配置（具体的なshapeを使用）
    let output_offset = build_contiguous_offset_with_shape(ndim, Some(shape));

    let load_dtype = graph_dtype_to_ast(&input.dtype);
    let target_ast_dtype = graph_dtype_to_ast(target_dtype);

    let load_expr = load(var(ph::input(0)), input_offset, load_dtype);
    let cast_expr = cast(load_expr, target_ast_dtype);
    let store_stmt = store(var(ph::OUTPUT), output_offset, cast_expr);

    // 具体的なshapeを使用してループを生成
    let body = wrap_with_loops_with_shape(ndim, vec![store_stmt], Some(shape));

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        body,
    ))
}
