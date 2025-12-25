//! Fold操作（col2im）のLowering
//!
//! Unfold後のテンソルを元の形状に戻す操作です。
//! 重複するパッチ位置は加算されます。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::GraphNode;
use crate::graph::View;
use crate::graph::ops::custom_placeholders as ph;

use super::helpers::{build_offset_from_coords_with_view, graph_dtype_to_ast, wrap_with_loops};

/// Fold操作の関数を生成
///
/// # Arguments
/// * `node` - Foldノード
/// * `output_size` - 出力サイズ（unfold前の形状）
/// * `kernel_size` - カーネルサイズ
/// * `stride` - ストライド
/// * `dilation` - 膨張率
/// * `groups` - グループ数
/// * `name` - 関数名
#[allow(clippy::too_many_arguments)]
pub fn build_fold_function(
    node: &GraphNode,
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let spatial_dims = kernel_size.len();
    let output_ndim = output_size.len();

    // 入力形状の検証
    // 入力: (C, k1, k2, ..., L1', L2', ...) または (groups, C/groups, k1, ..., L1', ...)
    let has_groups = groups > 1;

    let load_dtype = graph_dtype_to_ast(&input.dtype);

    // パッチ位置サイズ（L'）を計算
    let patch_sizes = compute_patch_sizes(output_size, kernel_size, stride, dilation, spatial_dims);

    // 本体を構築
    let body = build_fold_body(
        input_shape,
        &input.view,
        output_size,
        kernel_size,
        stride,
        dilation,
        &patch_sizes,
        spatial_dims,
        has_groups,
        groups,
        &load_dtype,
    );

    // 出力次元でループをラップ
    let wrapped_body = wrap_with_loops(output_ndim, vec![body]);

    Some(function(
        Some(name.to_string()),
        vec![],
        AstDType::Tuple(vec![]),
        wrapped_body,
    ))
}

/// パッチ位置サイズ（L'）を計算
/// L' = (output_size - effective_kernel) / stride + 1
/// effective_kernel = (kernel_size - 1) * dilation + 1
fn compute_patch_sizes(
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    dilation: &[usize],
    spatial_dims: usize,
) -> Vec<usize> {
    let output_spatial_start = output_size.len() - spatial_dims;

    (0..spatial_dims)
        .map(|i| {
            let out_dim = output_size[output_spatial_start + i];
            let eff_kernel = (kernel_size[i] - 1) * dilation[i] + 1;
            (out_dim - eff_kernel) / stride[i] + 1
        })
        .collect()
}

/// Fold本体を構築
#[allow(clippy::too_many_arguments)]
fn build_fold_body(
    input_shape: &[crate::graph::Expr],
    input_view: &View,
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    dilation: &[usize],
    patch_sizes: &[usize],
    spatial_dims: usize,
    has_groups: bool,
    groups: usize,
    load_dtype: &AstDType,
) -> AstNode {
    let output_ndim = output_size.len();

    // スコープ設定
    let mut scope = Scope::new();
    let _ = scope.declare("acc".to_string(), load_dtype.clone(), Mutability::Mutable);

    // アキュムレータ初期化
    let acc_init = assign("acc", const_f32(0.0));

    // 出力座標を取得
    let output_coords: Vec<AstNode> = (0..output_ndim).map(|i| var(ph::ridx(i))).collect();

    // カーネルループを構築
    let kernel_loops = build_fold_kernel_loops(
        input_shape,
        input_view,
        output_size,
        kernel_size,
        stride,
        dilation,
        patch_sizes,
        spatial_dims,
        has_groups,
        groups,
        load_dtype,
        &output_coords,
    );

    // 出力オフセット
    let output_offset = build_output_offset(output_size);

    // ストア
    let store_stmt = store(var(ph::OUTPUT), output_offset, var("acc"));

    block(vec![acc_init, kernel_loops, store_stmt], scope)
}

/// 出力オフセットを構築（row-major）
fn build_output_offset(output_size: &[usize]) -> AstNode {
    let ndim = output_size.len();
    if ndim == 0 {
        return const_int(0);
    }

    let mut offset = var(ph::ridx(ndim - 1));
    for axis in (0..ndim - 1).rev() {
        let stride: usize = output_size[(axis + 1)..].iter().product();
        offset = var(ph::ridx(axis)) * const_int(stride as i64) + offset;
    }
    offset
}

/// カーネル位置のネストループを構築
#[allow(clippy::too_many_arguments)]
fn build_fold_kernel_loops(
    input_shape: &[crate::graph::Expr],
    input_view: &View,
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    dilation: &[usize],
    patch_sizes: &[usize],
    spatial_dims: usize,
    has_groups: bool,
    groups: usize,
    load_dtype: &AstDType,
    output_coords: &[AstNode],
) -> AstNode {
    // 最内部: 入力からの読み込みと加算
    let inner_body = build_fold_inner_body(
        input_shape,
        input_view,
        output_size,
        kernel_size,
        stride,
        dilation,
        patch_sizes,
        spatial_dims,
        has_groups,
        groups,
        load_dtype,
        output_coords,
    );

    // カーネル次元でネストループ
    let mut body = inner_body;
    for k_axis in (0..spatial_dims).rev() {
        let k_var = format!("k{}", k_axis);
        body = range(
            k_var,
            const_int(0),
            const_int(1),
            const_int(kernel_size[k_axis] as i64),
            body,
        );
    }

    body
}

/// Foldの最内部ボディ: 境界チェックと入力アクセス
#[allow(clippy::too_many_arguments)]
fn build_fold_inner_body(
    _input_shape: &[crate::graph::Expr],
    input_view: &View,
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    dilation: &[usize],
    patch_sizes: &[usize],
    spatial_dims: usize,
    has_groups: bool,
    groups: usize,
    load_dtype: &AstDType,
    _output_coords: &[AstNode],
) -> AstNode {
    let output_ndim = output_size.len();
    let output_spatial_start = output_ndim - spatial_dims;

    // 各空間軸について、パッチ位置とストライド条件をチェック
    // h_offset = h - k * dilation
    // if h_offset >= 0 && h_offset % stride == 0:
    //   l = h_offset / stride
    //   if l < L':
    //     access input

    let mut conditions = Vec::new();
    let mut patch_indices = Vec::new();

    for s_axis in 0..spatial_dims {
        let k_var = var(format!("k{}", s_axis));
        let spatial_coord = var(ph::ridx(output_spatial_start + s_axis));

        // h_offset = h - k * dilation
        let offset_expr =
            spatial_coord.clone() - k_var.clone() * const_int(dilation[s_axis] as i64);

        // 条件1: h_offset >= 0
        conditions.push(ge(offset_expr.clone(), const_int(0)));

        // 条件2: h_offset % stride == 0
        if stride[s_axis] > 1 {
            conditions.push(eq(
                rem(offset_expr.clone(), const_int(stride[s_axis] as i64)),
                const_int(0),
            ));
        }

        // パッチ位置: l = h_offset / stride
        let patch_idx = if stride[s_axis] > 1 {
            idiv(offset_expr, const_int(stride[s_axis] as i64))
        } else {
            offset_expr
        };

        // 条件3: l < L'
        conditions.push(lt(patch_idx.clone(), const_int(patch_sizes[s_axis] as i64)));

        patch_indices.push(patch_idx);
    }

    // 入力座標を構築
    // 入力形状: (C, k1, k2, ..., L1', L2', ...) for groups=1
    // または (groups, C/groups, k1, ..., L1', ...) for groups>1
    let c_per_group = if has_groups {
        output_size[0] / groups
    } else {
        output_size[0]
    };
    let input_coords = build_input_coords(
        kernel_size,
        spatial_dims,
        has_groups,
        c_per_group,
        &patch_indices,
    );

    // 入力オフセットを計算（入力のViewを考慮）
    let input_offset = build_offset_from_coords_with_view(&input_coords, input_view);

    // 入力からロードしてアキュムレート
    let load_expr = load(var(ph::input(0)), input_offset, load_dtype.clone());
    let acc_update = assign("acc", var("acc") + load_expr);

    // すべての条件をANDで結合 (BitwiseAnd)
    let combined_condition = conditions
        .into_iter()
        .reduce(|a, b| a & b)
        .unwrap_or_else(|| AstNode::Const(true.into()));

    if_then(combined_condition, block(vec![acc_update], Scope::new()))
}

/// 入力座標を構築
///
/// 入力テンソルの各次元に対応する論理座標のリストを返します。
///
/// groups=1の場合:
///   入力形状: (C, k1, k2, ..., L1', L2', ...)
///   座標: [c, k0, k1, ..., l0, l1, ...]
///
/// groups>1の場合:
///   入力形状: (groups, C/groups, k1, k2, ..., L1', L2', ...)
///   座標: [g, c_local, k0, k1, ..., l0, l1, ...]
fn build_input_coords(
    _kernel_size: &[usize],
    spatial_dims: usize,
    has_groups: bool,
    c_per_group: usize,
    patch_indices: &[AstNode],
) -> Vec<AstNode> {
    let channel_coord = var(ph::ridx(0));
    let mut coords = Vec::new();

    if has_groups {
        // groups > 1: 入力形状は (groups, C/groups, k1, k2, ..., L1', L2', ...)
        // 出力チャネル c から g と c_local を計算
        let group_idx = idiv(channel_coord.clone(), const_int(c_per_group as i64));
        let c_local = rem(channel_coord, const_int(c_per_group as i64));

        coords.push(group_idx);
        coords.push(c_local);
    } else {
        // groups = 1: 入力形状は (C, k1, k2, ..., L1', L2', ...)
        coords.push(channel_coord);
    }

    // カーネル次元の座標
    for k_axis in 0..spatial_dims {
        coords.push(var(format!("k{}", k_axis)));
    }

    // パッチ位置の座標
    for patch_idx in patch_indices {
        coords.push(patch_idx.clone());
    }

    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_fold_lowering_1d_basic() {
        let mut graph = Graph::new();

        // 1D fold: (2, 3, 8) -> (2, 10)
        // kernel_size=3, stride=1, dilation=1
        let x = graph.input("x", DType::F32, vec![2, 3, 8]);
        let folded = x.fold(vec![2, 10], 3, 1, 1, 1);

        let ast = build_fold_function(&folded, &[2, 10], &[3], &[1], &[1], 1, "fold_1d");

        assert!(ast.is_some());
    }

    #[test]
    fn test_fold_lowering_2d_basic() {
        let mut graph = Graph::new();

        // 2D fold: (3, 3, 3, 30, 30) -> (3, 32, 32)
        // kernel_size=(3,3), stride=(1,1), dilation=(1,1)
        let x = graph.input("x", DType::F32, vec![3, 3, 3, 30, 30]);
        let folded = x.fold(vec![3, 32, 32], (3, 3), (1, 1), (1, 1), 1);

        let ast = build_fold_function(
            &folded,
            &[3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            1,
            "fold_2d",
        );

        assert!(ast.is_some());
    }

    #[test]
    fn test_compute_patch_sizes() {
        // output_size=(32,), kernel_size=(3,), stride=(1,), dilation=(1,)
        // L' = (32 - 3) / 1 + 1 = 30
        let patches = compute_patch_sizes(&[32], &[3], &[1], &[1], 1);
        assert_eq!(patches, vec![30]);

        // output_size=(32, 32), kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)
        // L' = (32 - 3) / 2 + 1 = 15
        let patches2 = compute_patch_sizes(&[32, 32], &[3, 3], &[2, 2], &[1, 1], 2);
        assert_eq!(patches2, vec![15, 15]);
    }

    #[test]
    fn test_fold_lowering_1d_groups() {
        let mut graph = Graph::new();

        // 1D fold with groups=2: (2, 2, 3, 8) -> (4, 10)
        // Input: (groups, C/groups, k, L') = (2, 2, 3, 8)
        // Output: (C, L) = (4, 10)
        // kernel_size=3, stride=1, dilation=1, groups=2
        let x = graph.input("x", DType::F32, vec![2, 2, 3, 8]);
        let folded = x.fold(vec![4, 10], 3, 1, 1, 2);

        let ast = build_fold_function(&folded, &[4, 10], &[3], &[1], &[1], 2, "fold_1d_groups");

        assert!(ast.is_some());
    }

    #[test]
    fn test_fold_lowering_2d_groups() {
        let mut graph = Graph::new();

        // 2D fold with groups=2: (2, 2, 3, 3, 6, 6) -> (4, 8, 8)
        // Input: (groups, C/groups, k1, k2, L1', L2') = (2, 2, 3, 3, 6, 6)
        // Output: (C, H, W) = (4, 8, 8)
        // kernel_size=(3,3), stride=(1,1), dilation=(1,1), groups=2
        let x = graph.input("x", DType::F32, vec![2, 2, 3, 3, 6, 6]);
        let folded = x.fold(vec![4, 8, 8], (3, 3), (1, 1), (1, 1), 2);

        let ast = build_fold_function(
            &folded,
            &[4, 8, 8],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            2,
            "fold_2d_groups",
        );

        assert!(ast.is_some());
    }
}
