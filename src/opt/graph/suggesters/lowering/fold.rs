//! Fold操作（col2im）のLowering
//!
//! Unfold後のテンソルを元の形状に戻す操作です。
//! 重複するパッチ位置は加算されます。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind, helper::*};
use crate::graph::GraphNode;
use crate::graph::ops::custom_placeholders as ph;

use super::helpers::{graph_dtype_to_ast, wrap_with_loops};
use super::parallel::ParallelizationStrategy;

/// Fold操作のカーネル/関数を生成
///
/// # Arguments
/// * `node` - Foldノード
/// * `output_size` - 出力サイズ（unfold前の形状）
/// * `kernel_size` - カーネルサイズ
/// * `stride` - ストライド
/// * `dilation` - 膨張率
/// * `groups` - グループ数
/// * `name` - カーネル名
/// * `strategy` - 並列化戦略
#[allow(clippy::too_many_arguments)]
pub fn build_fold_kernel(
    node: &GraphNode,
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    name: &str,
    strategy: &ParallelizationStrategy,
) -> Option<AstNode> {
    match strategy {
        ParallelizationStrategy::Sequential => build_fold_function(
            node,
            output_size,
            kernel_size,
            stride,
            dilation,
            groups,
            name,
        ),
        ParallelizationStrategy::FlatParallel {
            thread_group_size, ..
        } => build_fold_parallel_kernel(
            node,
            output_size,
            kernel_size,
            stride,
            dilation,
            groups,
            name,
            *thread_group_size,
        ),
    }
}

/// Sequential版: Fold関数を生成
fn build_fold_function(
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

/// Parallel版: Fold GPUカーネルを生成
#[allow(clippy::too_many_arguments)]
fn build_fold_parallel_kernel(
    node: &GraphNode,
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    name: &str,
    thread_group_size: usize,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let spatial_dims = kernel_size.len();

    let has_groups = groups > 1;
    let load_dtype = graph_dtype_to_ast(&input.dtype);

    // パッチ位置サイズを計算
    let patch_sizes = compute_patch_sizes(output_size, kernel_size, stride, dilation, spatial_dims);

    // 出力要素数
    let total_elements: usize = output_size.iter().product();

    // パラメータ生成
    let mut params = vec![VarDecl {
        name: "gidx".to_string(),
        dtype: AstDType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::GroupId(0),
    }];

    // 入力バッファ
    params.push(VarDecl {
        name: ph::input(0),
        dtype: load_dtype.clone().to_ptr(),
        mutability: Mutability::Immutable,
        kind: VarKind::Normal,
    });

    // 出力バッファ
    params.push(VarDecl {
        name: ph::OUTPUT.to_string(),
        dtype: load_dtype.clone().to_ptr(),
        mutability: Mutability::Mutable,
        kind: VarKind::Normal,
    });

    // gidxから出力座標への変換
    let coord_exprs = build_gidx_to_coords(output_size);

    // アキュムレータの初期化
    let mut scope = Scope::new();
    let _ = scope.declare("acc".to_string(), load_dtype.clone(), Mutability::Mutable);

    let acc_init = assign("acc", const_f32(0.0));

    // カーネルループと入力アクセス
    let kernel_loops = build_fold_kernel_loops(
        input_shape,
        output_size,
        kernel_size,
        stride,
        dilation,
        &patch_sizes,
        spatial_dims,
        has_groups,
        groups,
        &load_dtype,
        &coord_exprs,
    );

    // ストア
    let store_stmt = store(var(ph::OUTPUT), var("gidx"), var("acc"));

    // 境界チェック
    let body_stmts = vec![acc_init, kernel_loops, store_stmt];
    let body_block = block(body_stmts, scope);

    let checked_body = if_then(
        lt(var("gidx"), const_int(total_elements as isize)),
        body_block,
    );

    // グリッドサイズ計算
    let grid_size = total_elements.div_ceil(thread_group_size) * thread_group_size;

    Some(kernel(
        Some(name.to_string()),
        params,
        AstDType::Tuple(vec![]),
        block(vec![checked_body], Scope::new()),
        [const_int(grid_size as isize), const_int(1), const_int(1)],
        [
            const_int(thread_group_size as isize),
            const_int(1),
            const_int(1),
        ],
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

/// gidxから出力座標への変換式を構築
fn build_gidx_to_coords(output_size: &[usize]) -> Vec<AstNode> {
    let ndim = output_size.len();
    let mut coords = Vec::with_capacity(ndim);
    let mut remaining = var("gidx");

    for axis in 0..ndim {
        let stride: usize = output_size[(axis + 1)..].iter().product();
        if stride > 1 {
            coords.push(idiv(remaining.clone(), const_int(stride as isize)));
            remaining = rem(remaining, const_int(stride as isize));
        } else {
            coords.push(remaining.clone());
        }
    }

    coords
}

/// Sequential版のFold本体を構築
#[allow(clippy::too_many_arguments)]
fn build_fold_body(
    input_shape: &[crate::graph::Expr],
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
    let kernel_loops = build_fold_kernel_loops_sequential(
        input_shape,
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
        offset = var(ph::ridx(axis)) * const_int(stride as isize) + offset;
    }
    offset
}

/// Sequential版: カーネル位置のネストループを構築
#[allow(clippy::too_many_arguments)]
fn build_fold_kernel_loops_sequential(
    input_shape: &[crate::graph::Expr],
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
        true, // use ridx variables for kernel indices
    );

    // カーネル次元でネストループ
    let mut body = inner_body;
    for k_axis in (0..spatial_dims).rev() {
        let k_var = format!("k{}", k_axis);
        body = range(
            k_var,
            const_int(0),
            const_int(1),
            const_int(kernel_size[k_axis] as isize),
            body,
        );
    }

    body
}

/// Parallel版: カーネル位置のネストループを構築
#[allow(clippy::too_many_arguments)]
fn build_fold_kernel_loops(
    input_shape: &[crate::graph::Expr],
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
        false, // use coord expressions directly
    );

    // カーネル次元でネストループ
    let mut body = inner_body;
    for k_axis in (0..spatial_dims).rev() {
        let k_var = format!("k{}", k_axis);
        body = range(
            k_var,
            const_int(0),
            const_int(1),
            const_int(kernel_size[k_axis] as isize),
            body,
        );
    }

    body
}

/// Foldの最内部ボディ: 境界チェックと入力アクセス
#[allow(clippy::too_many_arguments)]
fn build_fold_inner_body(
    input_shape: &[crate::graph::Expr],
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
    use_ridx_for_spatial: bool,
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
        let spatial_coord = if use_ridx_for_spatial {
            var(ph::ridx(output_spatial_start + s_axis))
        } else {
            output_coords[output_spatial_start + s_axis].clone()
        };

        // h_offset = h - k * dilation
        let offset_expr =
            spatial_coord.clone() - k_var.clone() * const_int(dilation[s_axis] as isize);

        // 条件1: h_offset >= 0
        conditions.push(ge(offset_expr.clone(), const_int(0)));

        // 条件2: h_offset % stride == 0
        if stride[s_axis] > 1 {
            conditions.push(eq(
                rem(offset_expr.clone(), const_int(stride[s_axis] as isize)),
                const_int(0),
            ));
        }

        // パッチ位置: l = h_offset / stride
        let patch_idx = if stride[s_axis] > 1 {
            idiv(offset_expr, const_int(stride[s_axis] as isize))
        } else {
            offset_expr
        };

        // 条件3: l < L'
        conditions.push(lt(
            patch_idx.clone(),
            const_int(patch_sizes[s_axis] as isize),
        ));

        patch_indices.push(patch_idx);
    }

    // 入力インデックス計算
    // 入力形状: (C, k1, k2, ..., L1', L2', ...) for groups=1
    // または (groups, C/groups, k1, ..., L1', ...) for groups>1
    let c_per_group = if has_groups {
        output_size[0] / groups
    } else {
        output_size[0]
    };
    let input_idx = build_input_index(
        input_shape,
        kernel_size,
        patch_sizes,
        spatial_dims,
        has_groups,
        groups,
        c_per_group,
        output_coords,
        &patch_indices,
        use_ridx_for_spatial,
        output_spatial_start,
    );

    // 入力からロードしてアキュムレート
    let load_expr = load(var(ph::input(0)), input_idx, load_dtype.clone());
    let acc_update = assign("acc", var("acc") + load_expr);

    // すべての条件をANDで結合 (BitwiseAnd)
    let combined_condition = conditions
        .into_iter()
        .reduce(|a, b| a & b)
        .unwrap_or_else(|| AstNode::Const(true.into()));

    if_then(combined_condition, block(vec![acc_update], Scope::new()))
}

/// 入力インデックスを計算
///
/// groups=1の場合:
///   入力形状: (C, k1, k2, ..., L1', L2', ...)
///   インデックス: c * (k1*k2*...*L1'*L2'*...) + k1_idx * (k2*...*L1'*...) + ... + l1 * L2' + l2
///
/// groups>1の場合:
///   入力形状: (groups, C/groups, k1, k2, ..., L1', L2', ...)
///   出力チャネル c から: g = c / c_per_group, c_local = c % c_per_group
///   インデックス: g * (c_per_group*k1*k2*...*L1'*L2'*...) + c_local * (k1*k2*...*L1'*...) + ...
#[allow(clippy::too_many_arguments)]
fn build_input_index(
    _input_shape: &[crate::graph::Expr],
    kernel_size: &[usize],
    patch_sizes: &[usize],
    _spatial_dims: usize,
    has_groups: bool,
    _groups: usize,
    c_per_group: usize,
    output_coords: &[AstNode],
    patch_indices: &[AstNode],
    use_ridx_for_spatial: bool,
    _output_spatial_start: usize,
) -> AstNode {
    let channel_coord = if use_ridx_for_spatial {
        var(ph::ridx(0))
    } else {
        output_coords[0].clone()
    };

    // カーネル次元とパッチ次元のストライドを計算
    let kernel_total: usize = kernel_size.iter().product();
    let patch_total: usize = patch_sizes.iter().product();

    if has_groups {
        // groups > 1: 入力形状は (groups, C/groups, k1, k2, ..., L1', L2', ...)
        // 出力チャネル c から g と c_local を計算
        let group_idx = idiv(channel_coord.clone(), const_int(c_per_group as isize));
        let c_local = rem(channel_coord, const_int(c_per_group as isize));

        // グループ内の要素数
        let per_group_size = c_per_group * kernel_total * patch_total;

        // インデックス計算
        let mut idx = group_idx * const_int(per_group_size as isize)
            + c_local * const_int((kernel_total * patch_total) as isize);

        // カーネル次元部分
        for (k_axis, _k_size) in kernel_size.iter().enumerate() {
            let k_var = var(format!("k{}", k_axis));
            let k_stride: usize =
                kernel_size[(k_axis + 1)..].iter().product::<usize>() * patch_total;
            idx = idx + k_var * const_int(k_stride as isize);
        }

        // パッチ位置部分
        for (p_axis, patch_idx) in patch_indices.iter().enumerate() {
            let p_stride: usize = patch_sizes[(p_axis + 1)..].iter().product();
            idx = idx + patch_idx.clone() * const_int(p_stride as isize);
        }

        idx
    } else {
        // groups = 1: 入力形状は (C, k1, k2, ..., L1', L2', ...)
        // チャネル部分
        let mut idx = channel_coord * const_int((kernel_total * patch_total) as isize);

        // カーネル次元部分
        for (k_axis, _k_size) in kernel_size.iter().enumerate() {
            let k_var = var(format!("k{}", k_axis));
            let k_stride: usize =
                kernel_size[(k_axis + 1)..].iter().product::<usize>() * patch_total;
            idx = idx + k_var * const_int(k_stride as isize);
        }

        // パッチ位置部分
        for (p_axis, patch_idx) in patch_indices.iter().enumerate() {
            let p_stride: usize = patch_sizes[(p_axis + 1)..].iter().product();
            idx = idx + patch_idx.clone() * const_int(p_stride as isize);
        }

        idx
    }
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

        let ast = build_fold_kernel(
            &folded,
            &[2, 10],
            &[3],
            &[1],
            &[1],
            1,
            "fold_1d",
            &ParallelizationStrategy::Sequential,
        );

        assert!(ast.is_some());
    }

    #[test]
    fn test_fold_lowering_2d_basic() {
        let mut graph = Graph::new();

        // 2D fold: (3, 3, 3, 30, 30) -> (3, 32, 32)
        // kernel_size=(3,3), stride=(1,1), dilation=(1,1)
        let x = graph.input("x", DType::F32, vec![3, 3, 3, 30, 30]);
        let folded = x.fold(vec![3, 32, 32], (3, 3), (1, 1), (1, 1), 1);

        let ast = build_fold_kernel(
            &folded,
            &[3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            1,
            "fold_2d",
            &ParallelizationStrategy::Sequential,
        );

        assert!(ast.is_some());
    }

    #[test]
    fn test_fold_lowering_parallel() {
        let mut graph = Graph::new();

        let x = graph.input("x", DType::F32, vec![2, 3, 8]);
        let folded = x.fold(vec![2, 10], 3, 1, 1, 1);

        let ast = build_fold_kernel(
            &folded,
            &[2, 10],
            &[3],
            &[1],
            &[1],
            1,
            "fold_parallel",
            &ParallelizationStrategy::FlatParallel {
                thread_group_size: 256,
                vector_width: None,
            },
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

        let ast = build_fold_kernel(
            &folded,
            &[4, 10],
            &[3],
            &[1],
            &[1],
            2,
            "fold_1d_groups",
            &ParallelizationStrategy::Sequential,
        );

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

        let ast = build_fold_kernel(
            &folded,
            &[4, 8, 8],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            2,
            "fold_2d_groups",
            &ParallelizationStrategy::Sequential,
        );

        assert!(ast.is_some());
    }

    #[test]
    fn test_fold_lowering_parallel_groups() {
        let mut graph = Graph::new();

        // Parallel fold with groups=2
        let x = graph.input("x", DType::F32, vec![2, 2, 3, 8]);
        let folded = x.fold(vec![4, 10], 3, 1, 1, 2);

        let ast = build_fold_kernel(
            &folded,
            &[4, 10],
            &[3],
            &[1],
            &[1],
            2,
            "fold_parallel_groups",
            &ParallelizationStrategy::FlatParallel {
                thread_group_size: 256,
                vector_width: None,
            },
        );

        assert!(ast.is_some());
    }
}
