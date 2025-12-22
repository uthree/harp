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

/// ExprをAstNodeに変換する（LoadIndex対応版）
///
/// LoadIndexを含むExprを変換できます。LoadIndexは指定されたバッファから
/// 値を読み込むLoad ASTノードに変換されます。
///
/// # Arguments
/// * `expr` - 変換するExpr
/// * `src_vars` - ソースバッファの変数名リスト（例: ["input0", "input1", ...]）
/// * `dtype` - Load時のデータ型
///
/// # Panics
/// src_indexがsrc_varsの範囲外の場合にpanicします。
pub fn expr_to_ast_with_sources(expr: &Expr, src_vars: &[String], dtype: AstDType) -> AstNode {
    use crate::ast::Literal;

    let expr = expr.clone().simplify();
    match expr {
        Expr::Const(c) => AstNode::Const(Literal::I64(c)),
        Expr::Var(s) => AstNode::Var(s),
        Expr::Idx(i) => AstNode::Var(ph::ridx(i)),
        Expr::Add(l, r) => {
            let left = expr_to_ast_with_sources(&l, src_vars, dtype.clone());
            let right = expr_to_ast_with_sources(&r, src_vars, dtype.clone());
            left + right
        }
        Expr::Sub(l, r) => {
            let left = expr_to_ast_with_sources(&l, src_vars, dtype.clone());
            let right = expr_to_ast_with_sources(&r, src_vars, dtype.clone());
            left + (-right)
        }
        Expr::Mul(l, r) => {
            let left = expr_to_ast_with_sources(&l, src_vars, dtype.clone());
            let right = expr_to_ast_with_sources(&r, src_vars, dtype.clone());
            AstNode::Mul(Box::new(left), Box::new(right))
        }
        Expr::Div(l, r) => {
            let left = expr_to_ast_with_sources(&l, src_vars, dtype.clone());
            let right = expr_to_ast_with_sources(&r, src_vars, dtype.clone());
            left * recip(right)
        }
        Expr::Rem(l, r) => {
            let left = expr_to_ast_with_sources(&l, src_vars, dtype.clone());
            let right = expr_to_ast_with_sources(&r, src_vars, dtype.clone());
            AstNode::Rem(Box::new(left), Box::new(right))
        }
        Expr::LoadIndex {
            src_index,
            offset_expr,
        } => {
            // offset_exprを再帰的に変換
            // LoadIndexの読み込み型はI32（インデックス値）
            let offset_ast = expr_to_ast_with_sources(&offset_expr, src_vars, AstDType::I64);
            // src_varsからバッファ名を取得
            let buffer_name = &src_vars[src_index];
            // Load ASTノードを生成（インデックスはI64として読み込み）
            load(var(buffer_name.clone()), offset_ast, AstDType::I64)
        }
    }
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
        GraphDType::I64 => AstDType::I64, // 64-bit signed integer (for indexing)
        GraphDType::I32 => AstDType::I32, // 32-bit signed integer
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
            GraphDType::I32 => const_int(i32::MIN as i64),
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

/// 特定軸を除いた連続メモリのオフセット計算式を構築（Reduce用、具体的なshapeを使用）
pub fn build_contiguous_offset_excluding_axis_with_shape(
    ndim: usize,
    exclude_axis: usize,
    shape: Option<&[Expr]>,
) -> AstNode {
    build_contiguous_offset_excluding_axes_with_shape(ndim, &[exclude_axis], shape)
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

/// Viewを考慮したストライドベースのオフセット計算式を構築（LoadIndex対応版）
///
/// LoadIndexを含むIndexExpr Viewに対応します。LoadIndexは指定されたバッファから
/// インデックス値を読み込むLoad ASTノードに変換されます。
///
/// # Arguments
/// * `view` - 対象のView
/// * `ndim` - 次元数
/// * `src_vars` - ソースバッファの変数名リスト
/// * `dtype` - データ型（LoadIndex以外の値の型）
pub fn build_strided_offset_with_sources(
    view: &View,
    ndim: usize,
    src_vars: &[String],
    dtype: AstDType,
) -> AstNode {
    if ndim == 0 {
        return const_int(0);
    }

    match view {
        View::Linear {
            strides, offset, ..
        } => {
            // Linear Viewは既存の処理と同じ
            let mut result: AstNode = offset.clone().into();

            for (axis, stride_expr) in strides.iter().enumerate().take(ndim) {
                let stride: AstNode = stride_expr.clone().into();
                result = result + var(ph::ridx(axis)) * stride;
            }

            result
        }
        View::IndexExpr { index_expr, .. } => {
            // IndexExprはLoadIndexを含む可能性があるので専用関数を使用
            expr_to_ast_with_sources(index_expr, src_vars, dtype)
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

/// 座標リストからViewを考慮したオフセットを計算
///
/// Pad/Slice演算のように、論理座標から物理オフセットを計算する必要がある場合に使用します。
/// View::Linearの場合はストライドとオフセットを使用して正しく計算されます。
///
/// # Arguments
/// * `coords` - 論理座標のリスト（AstNode）
/// * `view` - 入力テンソルのView
///
/// # Returns
/// 物理オフセットを表すAstNode
///
/// # Panics
/// View::IndexExprは座標ベースのオフセット計算に対応していないためpanicします。
pub fn build_offset_from_coords_with_view(coords: &[AstNode], view: &View) -> AstNode {
    match view {
        View::Linear {
            strides, offset, ..
        } => {
            let mut result: AstNode = offset.clone().into();
            for (coord, stride) in coords.iter().zip(strides.iter()) {
                let s: AstNode = stride.clone().into();
                result = result + coord.clone() * s;
            }
            result
        }
        View::IndexExpr { .. } => {
            // IndexExprは任意の座標から物理オフセットへの単純なマッピングを持たない
            // このケースはPad/Sliceでは通常発生しない（IndexExprはGather等の特殊操作で使用）
            panic!(
                "IndexExpr view is not supported for coordinate-based offset calculation \
                 (Pad/Slice). Use Contiguous to materialize the tensor first."
            );
        }
    }
}
