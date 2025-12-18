//! LoweringSuggester用のヘルパー関数
//!
//! オフセット計算、ループ生成、型変換など共通のユーティリティを提供します。

use crate::ast::{AstNode, DType as AstDType, Scope, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::shape::Expr;
use crate::graph::{
    CumulativeOp, DType as GraphDType, GraphNode, GraphNodeData, GraphOp, ReduceOp, View,
};
use std::collections::HashSet;

/// Shape式をAstNodeに変換する
///
/// 式を簡約して定数になる場合は定数ノードを返し、
/// そうでない場合はシンボリック変数として返す。
pub fn shape_expr_to_ast(expr: &Expr) -> AstNode {
    // From<Expr> for AstNodeが自動的にsimplifyしてConstに変換する
    expr.clone().into()
}

/// 指定軸のShape式をAstNodeに変換する
///
/// shape配列が利用可能な場合は具体値を使用し、
/// そうでない場合はプレースホルダー変数を使用する。
pub fn shape_dim_to_ast(shape: Option<&[Expr]>, axis: usize) -> AstNode {
    if let Some(s) = shape
        && axis < s.len()
    {
        return shape_expr_to_ast(&s[axis]);
    }
    // フォールバック: プレースホルダー変数を使用
    var(ph::shape(axis))
}

/// GraphのDTypeをAstのDTypeに変換
pub fn graph_dtype_to_ast(dtype: &GraphDType) -> AstDType {
    match dtype {
        GraphDType::Bool => AstDType::Bool,
        GraphDType::I32 => AstDType::Int,
        GraphDType::F32 => AstDType::F32,
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

/// アキュムレータの型エイリアス
pub type AccumulateFn = Box<dyn Fn(AstNode, AstNode) -> AstNode>;

/// Reduce演算のアキュムレータ（初期値と更新関数）を生成
///
/// # Arguments
/// * `op` - Reduce演算の種類
/// * `dtype` - 出力の型
///
/// # Returns
/// (初期値, 更新関数) のタプル
pub fn build_reduce_accumulator(op: &ReduceOp, dtype: &GraphDType) -> (AstNode, AccumulateFn) {
    match op {
        ReduceOp::Sum => (get_reduce_init(dtype, op), Box::new(|acc, val| acc + val)),
        ReduceOp::Prod => (get_reduce_init(dtype, op), Box::new(|acc, val| acc * val)),
        ReduceOp::Max => (get_reduce_init(dtype, op), Box::new(max)),
    }
}

/// Cumulative演算のアキュムレータ（初期値と更新関数）を生成
///
/// # Arguments
/// * `op` - Cumulative演算の種類
/// * `dtype` - 出力の型
///
/// # Returns
/// (初期値, 更新関数) のタプル
pub fn build_cumulative_accumulator(
    op: &CumulativeOp,
    dtype: &GraphDType,
) -> (AstNode, AccumulateFn) {
    match op {
        CumulativeOp::Sum => {
            let init = match dtype {
                GraphDType::I32 => const_int(0),
                _ => const_f32(0.0),
            };
            (init, Box::new(|acc, val| acc + val))
        }
        CumulativeOp::Prod => {
            let init = match dtype {
                GraphDType::I32 => const_int(1),
                _ => const_f32(1.0),
            };
            (init, Box::new(|acc, val| acc * val))
        }
    }
}

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

/// 特定軸を除いた連続メモリのオフセット計算式を構築（Reduce用）
#[allow(dead_code)]
pub fn build_contiguous_offset_excluding_axis(ndim: usize, exclude_axis: usize) -> AstNode {
    build_contiguous_offset_excluding_axes_with_shape(ndim, &[exclude_axis], None)
}

/// 特定軸を除いた連続メモリのオフセット計算式を構築（Reduce用、具体的なshapeを使用）
pub fn build_contiguous_offset_excluding_axis_with_shape(
    ndim: usize,
    exclude_axis: usize,
    shape: Option<&[Expr]>,
) -> AstNode {
    build_contiguous_offset_excluding_axes_with_shape(ndim, &[exclude_axis], shape)
}

/// 複数の軸を除いた連続メモリのオフセット計算式を構築（複数軸Reduce用）
#[allow(dead_code)]
pub fn build_contiguous_offset_excluding_axes(ndim: usize, exclude_axes: &[usize]) -> AstNode {
    build_contiguous_offset_excluding_axes_with_shape(ndim, exclude_axes, None)
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
        View::IndexExpr { index_expr, .. } => {
            // IndexExprはExpr::Idxを含む式で、From<Expr> for AstNodeが
            // Idx(i)をridx(i)変数に変換する
            index_expr.clone().into()
        }
    }
}

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

/// 特定軸を除いたネストされたループで本体をラップ（スコープ付き）
#[allow(dead_code)]
pub fn wrap_with_loops_excluding_axis_with_scope(
    ndim: usize,
    exclude_axis: usize,
    inner_body: Vec<AstNode>,
    scope: Scope,
) -> AstNode {
    wrap_with_loops_excluding_axes_with_scope_and_shape(
        ndim,
        &[exclude_axis],
        inner_body,
        scope,
        None,
    )
}

/// 特定軸を除いたネストされたループで本体をラップ（スコープ付き、具体的なshapeを使用）
pub fn wrap_with_loops_excluding_axis_with_scope_and_shape(
    ndim: usize,
    exclude_axis: usize,
    inner_body: Vec<AstNode>,
    scope: Scope,
    shape: Option<&[Expr]>,
) -> AstNode {
    wrap_with_loops_excluding_axes_with_scope_and_shape(
        ndim,
        &[exclude_axis],
        inner_body,
        scope,
        shape,
    )
}

/// 複数軸を除いたネストされたループで本体をラップ（スコープ付き、複数軸対応）
#[allow(dead_code)]
pub fn wrap_with_loops_excluding_axes_with_scope(
    ndim: usize,
    exclude_axes: &[usize],
    inner_body: Vec<AstNode>,
    scope: Scope,
) -> AstNode {
    wrap_with_loops_excluding_axes_with_scope_and_shape(ndim, exclude_axes, inner_body, scope, None)
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

/// 最内軸をSIMD化したネストされたループで本体をラップ
///
/// 最内軸（ndim-1）をsimd_widthずつ処理するSIMDループを生成し、
/// 残りの要素を処理するテールループも生成します。
///
/// # Arguments
/// * `ndim` - テンソルの次元数
/// * `simd_body` - SIMDループの本体
/// * `scalar_body` - テールループの本体（スカラー処理）
/// * `shape` - テンソルのshape配列
/// * `simd_width` - SIMD幅（例: 4, 8）
///
/// # Generated structure
/// ```text
/// for ridx_0 in 0..shape[0]:
///   for ridx_{n-2} in 0..shape[n-2]:
///     // SIMDループ
///     for ridx_{n-1} in 0..floor(shape[n-1]/simd_width)*simd_width step simd_width:
///       simd_body
///     // テールループ
///     for ridx_{n-1} in floor(shape[n-1]/simd_width)*simd_width..shape[n-1]:
///       scalar_body
/// ```
pub fn wrap_with_simd_innermost_loop(
    ndim: usize,
    simd_body: Vec<AstNode>,
    scalar_body: Vec<AstNode>,
    shape: Option<&[Expr]>,
    simd_width: usize,
) -> AstNode {
    if ndim == 0 {
        return block(simd_body, Scope::new());
    }

    let innermost_axis = ndim - 1;
    let innermost_size = shape_dim_to_ast(shape, innermost_axis);

    // SIMD境界: floor(size / simd_width) * simd_width
    let simd_width_ast = const_int(simd_width as isize);
    let simd_limit = (innermost_size.clone() / simd_width_ast.clone()) * simd_width_ast.clone();

    // SIMDループ: 0..simd_limit step simd_width
    let simd_loop = range(
        ph::ridx(innermost_axis),
        const_int(0),
        simd_width_ast,
        simd_limit.clone(),
        block(simd_body, Scope::new()),
    );

    // テールループ: simd_limit..innermost_size step 1
    let tail_loop = range(
        ph::ridx(innermost_axis),
        simd_limit,
        const_int(1),
        innermost_size,
        block(scalar_body, Scope::new()),
    );

    // 内側のボディを構築（SIMDループ + テールループ）
    let inner_body = vec![simd_loop, tail_loop];

    // 外側のループを構築（最内軸を除く）
    if ndim == 1 {
        return block(inner_body, Scope::new());
    }

    let mut body = block(inner_body, Scope::new());

    for axis in (0..ndim - 1).rev() {
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

/// 縮約軸のストライドを計算
///
/// 連続メモリレイアウトを前提として、指定軸のストライド（要素数）を計算します。
/// stride = shape[axis+1] * shape[axis+2] * ... * shape[ndim-1]
///
/// # Arguments
/// * `axis` - ストライドを計算する軸
/// * `shape` - テンソルのshape配列
///
/// # Returns
/// ストライドを表すASTノード
pub fn build_reduce_axis_stride(axis: usize, shape: Option<&[Expr]>) -> AstNode {
    let ndim = shape.map(|s| s.len()).unwrap_or(0);
    if axis >= ndim.saturating_sub(1) {
        // 最内軸またはそれより外側の場合、stride = 1
        return const_int(1);
    }

    let mut stride = shape_dim_to_ast(shape, axis + 1);
    for inner_axis in (axis + 2)..ndim {
        stride = stride * shape_dim_to_ast(shape, inner_axis);
    }
    stride
}

/// アンロールされた縮約ループを生成
///
/// 指定されたアンロールファクターで縮約ループを展開し、
/// 端数を処理するテールループも生成します。
///
/// # Arguments
/// * `reduce_axis` - 縮約軸
/// * `unroll_factor` - アンロールファクター（例: 4, 8）
/// * `input_shape` - 入力テンソルのshape
/// * `base_offset` - ベースオフセット計算式
/// * `accumulate_fn` - アキュムレータ更新関数
/// * `value_loader` - 値をロードする関数（offset差分を受け取る）
///
/// # Generated structure
/// ```text
/// // アンロールループ（0..unroll_limit, step=factor）
/// for ridx_axis in 0..floor(size/factor)*factor step factor:
///   acc = acc + load(base_offset)
///   acc = acc + load(base_offset + stride)
///   acc = acc + load(base_offset + 2*stride)
///   acc = acc + load(base_offset + 3*stride)
/// // テールループ（unroll_limit..size, step=1）
/// for ridx_axis in floor(size/factor)*factor..size:
///   acc = acc + load(offset)
/// ```
pub fn build_unrolled_reduce_loop<F>(
    reduce_axis: usize,
    unroll_factor: usize,
    input_shape: &[Expr],
    accumulate_fn: &AccumulateFn,
    value_loader: F,
) -> AstNode
where
    F: Fn(AstNode) -> AstNode,
{
    let acc_var = "acc";
    let axis_size = shape_dim_to_ast(Some(input_shape), reduce_axis);
    let factor_ast = const_int(unroll_factor as isize);
    let stride = build_reduce_axis_stride(reduce_axis, Some(input_shape));

    // アンロール境界: floor(size / factor) * factor
    let unroll_limit = (axis_size.clone() / factor_ast.clone()) * factor_ast.clone();

    // アンロールループ本体を構築
    let mut unroll_body_stmts = Vec::new();
    for k in 0..unroll_factor {
        let offset_delta = if k == 0 {
            const_int(0)
        } else {
            stride.clone() * const_int(k as isize)
        };
        let value = value_loader(offset_delta);
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value));
        unroll_body_stmts.push(acc_update);
    }

    // アンロールループ
    let unroll_loop = range(
        ph::ridx(reduce_axis),
        const_int(0),
        factor_ast,
        unroll_limit.clone(),
        block(unroll_body_stmts, Scope::new()),
    );

    // テールループ本体
    let tail_value = value_loader(const_int(0));
    let tail_acc_update = assign(acc_var, accumulate_fn(var(acc_var), tail_value));

    // テールループ
    let tail_loop = range(
        ph::ridx(reduce_axis),
        unroll_limit,
        const_int(1),
        axis_size,
        block(vec![tail_acc_update], Scope::new()),
    );

    block(vec![unroll_loop, tail_loop], Scope::new())
}

/// srcノードからView経由でInputまで辿り、対応するBufferノードとKernel依存を収集する
///
/// View操作を経由してInputノードまで辿り、各Inputに対応するBufferノードを作成します。
/// また、依存するKernelノードも収集して実行順序の依存関係を保持します。
/// 重複するInputは1つのBufferにまとめられます。
///
/// # 重要
/// 各srcノードに対して、そのsrcのview（変形後の形状）を保持してBufferノードを作成します。
/// これにより、Kernelの入力形状が正しく追跡されます。
///
/// # Arguments
/// * `src_nodes` - 入力ソースノードのスライス
///
/// # Returns
/// 収集されたBufferノードとKernelノードのベクタ（重複排除済み）
/// 順序: [入力Bufferノード..., 依存Kernelノード...]
pub fn collect_input_buffers(src_nodes: &[GraphNode]) -> Vec<GraphNode> {
    let mut input_names = Vec::new();
    let mut kernel_deps: Vec<GraphNode> = Vec::new();
    let mut seen_buffers = HashSet::new();
    let mut seen_kernels: HashSet<*const GraphNodeData> = HashSet::new();

    for src in src_nodes {
        // 各srcノードのviewを"entry view"として保持
        // これがKernelが期待する入力形状
        let entry_view = src.view.clone();
        collect_inputs_recursive(
            src,
            &mut input_names,
            &mut kernel_deps,
            &mut seen_buffers,
            &mut seen_kernels,
            &entry_view,
        );
    }

    // 収集したInputノード情報からBufferノードを作成
    let mut result: Vec<GraphNode> = input_names
        .into_iter()
        .map(|(name, dtype, view)| GraphNode::new(dtype, GraphOp::Buffer { name }, vec![], view))
        .collect();

    // 依存Kernelを追加（実行順序の依存関係を保持）
    result.extend(kernel_deps);

    result
}

/// 再帰的に入力Bufferノードと依存Kernelを収集する
///
/// # Arguments
/// * `node` - 現在探索中のノード
/// * `inputs` - 収集された入力Buffer情報
/// * `kernel_deps` - 収集された依存Kernelノード
/// * `seen_buffers` - Buffer重複排除用のセット
/// * `seen_kernels` - Kernel重複排除用のセット
/// * `entry_view` - トレース開始点のview（Kernelが期待する入力形状）
fn collect_inputs_recursive(
    node: &GraphNode,
    inputs: &mut Vec<(String, crate::graph::DType, View)>,
    kernel_deps: &mut Vec<GraphNode>,
    seen_buffers: &mut HashSet<String>,
    seen_kernels: &mut HashSet<*const GraphNodeData>,
    entry_view: &View,
) {
    match &node.op {
        GraphOp::Buffer { name } => {
            // 出力バッファー（output_で始まる）は除外
            if !name.starts_with("output_") && !seen_buffers.contains(name) {
                seen_buffers.insert(name.clone());
                // entry_viewを使用（元のBufferのviewではなく、Kernelが期待する形状）
                inputs.push((name.clone(), node.dtype.clone(), entry_view.clone()));
            }
        }
        GraphOp::View(_) => {
            // Viewノードの場合、srcを辿る（entry_viewは変更しない）
            for src in &node.src {
                collect_inputs_recursive(
                    src,
                    inputs,
                    kernel_deps,
                    seen_buffers,
                    seen_kernels,
                    entry_view,
                );
            }
        }
        GraphOp::Const(_) => {
            // 定数は無視
        }
        GraphOp::Kernel { .. } => {
            let ptr = node.as_ptr();

            // 依存Kernelを記録（実行順序の依存関係を保持するため）
            if !seen_kernels.contains(&ptr) {
                seen_kernels.insert(ptr);
                kernel_deps.push(node.clone());
            }

            // Kernelノードの場合、その出力バッファを入力として使用する
            // Kernelのsrcは [..., 出力バッファ] の形式
            // 他のノードがKernelに依存する場合、Kernelの出力を読み取る必要がある
            if let Some(output_buffer) = node.src.last()
                && let GraphOp::Buffer { name } = &output_buffer.op
                && !seen_buffers.contains(name)
            {
                seen_buffers.insert(name.clone());
                // entry_viewを使用（Kernelの出力形状を入力として期待）
                inputs.push((
                    name.clone(),
                    output_buffer.dtype.clone(),
                    entry_view.clone(),
                ));
            }
        }
        _ => {
            // まだlowerされていない計算ノード（FusedElementwiseReduce等）は
            // ここに到達すべきではない - can_lower()で弾かれるべき
            log::warn!(
                "collect_input_buffers: unexpected node type {:?}, this node should have been lowered first",
                std::mem::discriminant(&node.op)
            );
        }
    }
}
