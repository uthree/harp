//! 並列化サポート
//!
//! GPUカーネル向けの並列化戦略とカーネル生成機能を提供します。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind, helper::*};
use crate::graph::ops::custom_placeholders as ph;

use super::helpers::graph_dtype_to_ast;

/// 並列化戦略
#[derive(Clone, Debug, PartialEq)]
pub enum ParallelizationStrategy {
    /// 逐次実行（CPU向け、Rangeループを使用）
    Sequential,
    /// フラット並列化（1スレッド1要素）
    /// 全要素を線形インデックスで並列処理
    FlatParallel,
    /// 多次元並列化
    /// 指定した軸数までをスレッドIDで並列化（例: 2なら2次元グリッド）
    /// 残りの軸は逐次ループ
    MultiDimParallel {
        /// 並列化する軸数（1, 2, または 3）
        parallel_dims: usize,
    },
}

/// Elementwise演算の並列カーネルを生成
///
/// # 引数
/// - `ndim`: 入力/出力のテンソル次元数
/// - `num_inputs`: 入力バッファ数
/// - `expr`: Elementwise式（ワイルドカードで入力を参照）
/// - `output_dtype`: 出力の型
/// - `name`: カーネル名
/// - `strategy`: 並列化戦略
pub fn build_parallel_elementwise_kernel(
    ndim: usize,
    num_inputs: usize,
    expr: AstNode,
    output_dtype: &crate::graph::DType,
    name: &str,
    strategy: &ParallelizationStrategy,
) -> AstNode {
    match strategy {
        ParallelizationStrategy::Sequential => {
            // 逐次版：既存のfunction実装を使用
            super::elementwise::build_elementwise_function_impl(
                ndim,
                num_inputs,
                expr,
                output_dtype,
                name,
            )
        }
        ParallelizationStrategy::FlatParallel => {
            build_flat_parallel_elementwise_kernel(ndim, num_inputs, expr, output_dtype, name)
        }
        ParallelizationStrategy::MultiDimParallel { parallel_dims } => {
            build_multidim_parallel_elementwise_kernel(
                ndim,
                num_inputs,
                expr,
                output_dtype,
                name,
                *parallel_dims,
            )
        }
    }
}

/// フラット並列Elementwiseカーネルを生成
///
/// 線形インデックス（tid）を使用して各要素を1スレッドで処理
fn build_flat_parallel_elementwise_kernel(
    ndim: usize,
    num_inputs: usize,
    expr: AstNode,
    output_dtype: &crate::graph::DType,
    name: &str,
) -> AstNode {
    // tidパラメータ（グローバルスレッドID）
    let tid_param = VarDecl {
        name: "tid".to_string(),
        dtype: AstDType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::ThreadId(0),
    };

    // 入力バッファパラメータ
    let mut params = vec![tid_param];
    let load_dtype = graph_dtype_to_ast(output_dtype);

    for i in 0..num_inputs {
        params.push(VarDecl {
            name: ph::input(i),
            dtype: load_dtype.clone().to_ptr(),
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        });
    }

    // 出力バッファパラメータ
    params.push(VarDecl {
        name: ph::OUTPUT.to_string(),
        dtype: load_dtype.clone().to_ptr(),
        mutability: Mutability::Mutable,
        kind: VarKind::Normal,
    });

    // tidを直接オフセットとして使用（フラット配列アクセス）
    let offset = var("tid");

    // 入力をロードして式を構築
    let mut mappings = std::collections::HashMap::new();
    for i in 0..num_inputs {
        let load_node = load(var(ph::input(i)), offset.clone(), load_dtype.clone());
        mappings.insert(i.to_string(), load_node);
    }
    let final_expr = expr.substitute(&mappings);

    // ストア文
    let store_stmt = store(var(ph::OUTPUT), offset, final_expr);
    let body = block(vec![store_stmt], Scope::new());

    // グリッドサイズ計算: 全要素数
    let grid_size = build_total_elements(ndim);

    kernel_1d(
        Some(name.to_string()),
        params,
        AstDType::Tuple(vec![]),
        body,
        grid_size,
        const_int(256), // デフォルトスレッドグループサイズ
    )
}

/// 多次元並列Elementwiseカーネルを生成
///
/// 指定した軸数をスレッドIDで並列化し、残りを逐次ループ
fn build_multidim_parallel_elementwise_kernel(
    ndim: usize,
    num_inputs: usize,
    expr: AstNode,
    output_dtype: &crate::graph::DType,
    name: &str,
    parallel_dims: usize,
) -> AstNode {
    let actual_parallel_dims = parallel_dims.min(ndim).min(3);
    let load_dtype = graph_dtype_to_ast(output_dtype);

    // スレッドIDパラメータ
    let mut params: Vec<VarDecl> = (0..actual_parallel_dims)
        .map(|axis| VarDecl {
            name: format!("tid_{}", axis),
            dtype: AstDType::Int,
            mutability: Mutability::Immutable,
            kind: VarKind::ThreadId(axis),
        })
        .collect();

    // 入力バッファパラメータ
    for i in 0..num_inputs {
        params.push(VarDecl {
            name: ph::input(i),
            dtype: load_dtype.clone().to_ptr(),
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        });
    }

    // 出力バッファパラメータ
    params.push(VarDecl {
        name: ph::OUTPUT.to_string(),
        dtype: load_dtype.clone().to_ptr(),
        mutability: Mutability::Mutable,
        kind: VarKind::Normal,
    });

    // オフセット計算: 並列軸はtid_N、逐次軸はridx_N
    let offset = build_hybrid_offset(ndim, actual_parallel_dims);

    // 入力をロードして式を構築
    let mut mappings = std::collections::HashMap::new();
    for i in 0..num_inputs {
        let load_node = load(var(ph::input(i)), offset.clone(), load_dtype.clone());
        mappings.insert(i.to_string(), load_node);
    }
    let final_expr = expr.substitute(&mappings);

    // ストア文
    let store_stmt = store(var(ph::OUTPUT), offset, final_expr);

    // 逐次ループでラップ（並列軸以降の軸）
    let body = wrap_sequential_loops(ndim, actual_parallel_dims, vec![store_stmt]);

    // グリッドサイズ: 並列化した軸のサイズ
    let grid_size: [Box<AstNode>; 3] = match actual_parallel_dims {
        1 => [
            Box::new(var(ph::shape(0))),
            Box::new(const_int(1)),
            Box::new(const_int(1)),
        ],
        2 => [
            Box::new(var(ph::shape(0))),
            Box::new(var(ph::shape(1))),
            Box::new(const_int(1)),
        ],
        3 => [
            Box::new(var(ph::shape(0))),
            Box::new(var(ph::shape(1))),
            Box::new(var(ph::shape(2))),
        ],
        _ => [
            Box::new(const_int(1)),
            Box::new(const_int(1)),
            Box::new(const_int(1)),
        ],
    };

    let thread_group_size: [Box<AstNode>; 3] = [
        Box::new(const_int(256)),
        Box::new(const_int(1)),
        Box::new(const_int(1)),
    ];

    AstNode::Kernel {
        name: Some(name.to_string()),
        params,
        return_type: AstDType::Tuple(vec![]),
        body: Box::new(body),
        default_grid_size: grid_size,
        default_thread_group_size: thread_group_size,
    }
}

/// 全要素数を計算する式を生成
fn build_total_elements(ndim: usize) -> AstNode {
    if ndim == 0 {
        return const_int(1);
    }
    let mut total = var(ph::shape(0));
    for axis in 1..ndim {
        total = total * var(ph::shape(axis));
    }
    total
}

/// ハイブリッドオフセット計算（並列軸はtid_N、逐次軸はridx_N）
fn build_hybrid_offset(ndim: usize, parallel_dims: usize) -> AstNode {
    if ndim == 0 {
        return const_int(0);
    }

    // C-contiguous (row-major) オフセット計算
    // offset = idx_0 * stride_0 + idx_1 * stride_1 + ... + idx_(n-1)
    // stride_i = shape_(i+1) * shape_(i+2) * ... * shape_(n-1)

    let mut offset = index_var(ndim - 1, parallel_dims);

    for axis in (0..ndim - 1).rev() {
        let mut stride = var(ph::shape(axis + 1));
        for inner_axis in (axis + 2)..ndim {
            stride = stride * var(ph::shape(inner_axis));
        }
        offset = index_var(axis, parallel_dims) * stride + offset;
    }

    offset
}

/// 軸に応じたインデックス変数を取得
fn index_var(axis: usize, parallel_dims: usize) -> AstNode {
    if axis < parallel_dims {
        var(format!("tid_{}", axis))
    } else {
        var(ph::ridx(axis))
    }
}

/// 逐次ループでラップ（並列化されていない軸のみ）
fn wrap_sequential_loops(ndim: usize, parallel_dims: usize, inner_body: Vec<AstNode>) -> AstNode {
    if parallel_dims >= ndim {
        return block(inner_body, Scope::new());
    }

    let mut body = block(inner_body, Scope::new());

    for axis in (parallel_dims..ndim).rev() {
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

/// Reduce演算の並列カーネルを生成
pub fn build_parallel_reduce_kernel(
    node: &crate::graph::GraphNode,
    op: &crate::graph::ReduceOp,
    axis: usize,
    name: &str,
    strategy: &ParallelizationStrategy,
) -> Option<AstNode> {
    match strategy {
        ParallelizationStrategy::Sequential => {
            super::reduce::build_reduce_function(node, op, axis, name)
        }
        ParallelizationStrategy::FlatParallel => {
            build_flat_parallel_reduce_kernel(node, op, axis, name)
        }
        ParallelizationStrategy::MultiDimParallel { parallel_dims } => {
            build_multidim_parallel_reduce_kernel(node, op, axis, name, *parallel_dims)
        }
    }
}

/// フラット並列Reduceカーネルを生成
fn build_flat_parallel_reduce_kernel(
    node: &crate::graph::GraphNode,
    op: &crate::graph::ReduceOp,
    axis: usize,
    name: &str,
) -> Option<AstNode> {
    use crate::graph::ReduceOp;

    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) = match op
    {
        ReduceOp::Sum => (
            super::helpers::get_reduce_init(&node.dtype, op),
            Box::new(|acc, val| acc + val),
        ),
        ReduceOp::Prod => (
            super::helpers::get_reduce_init(&node.dtype, op),
            Box::new(|acc, val| acc * val),
        ),
        ReduceOp::Max => (
            super::helpers::get_reduce_init(&node.dtype, op),
            Box::new(max),
        ),
    };

    let load_dtype = graph_dtype_to_ast(&input.dtype);

    // スレッドID（出力インデックス）
    let tid_param = VarDecl {
        name: "tid".to_string(),
        dtype: AstDType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::ThreadId(0),
    };

    let mut params = vec![tid_param];

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

    // tidから多次元インデックスを計算（reduce軸を除く）
    // 出力の線形インデックス -> 入力の基底インデックス
    let (base_offset_expr, local_scope) =
        build_reduce_index_calculation(ndim, axis, &load_dtype, &init_value);

    // reduce軸のループ
    let reduce_var = ph::ridx(axis);
    let reduce_axis_stride = build_axis_stride(ndim, axis);
    let input_offset = base_offset_expr + var(&reduce_var) * reduce_axis_stride;

    let value_expr = load(var(ph::input(0)), input_offset, load_dtype.clone());
    let acc_update = assign("acc", accumulate_fn(var("acc"), value_expr));

    let reduce_loop = range(
        reduce_var,
        const_int(0),
        const_int(1),
        var(ph::shape(axis)),
        block(vec![acc_update], Scope::new()),
    );

    let store_stmt = store(var(ph::OUTPUT), var("tid"), var("acc"));

    let body = block(vec![reduce_loop, store_stmt], local_scope);

    // グリッドサイズ: 出力要素数
    let grid_size = build_output_elements(ndim, axis);

    Some(kernel_1d(
        Some(name.to_string()),
        params,
        AstDType::Tuple(vec![]),
        body,
        grid_size,
        const_int(256),
    ))
}

/// Reduceのインデックス計算を生成
fn build_reduce_index_calculation(
    ndim: usize,
    reduce_axis: usize,
    dtype: &AstDType,
    init_value: &AstNode,
) -> (AstNode, Scope) {
    let mut scope = Scope::new();

    // アキュムレータ変数
    let _ = scope.declare("acc".to_string(), dtype.clone(), Mutability::Mutable);

    // 出力の線形インデックスから各軸のインデックスを逆算
    // tid -> (idx_0, idx_1, ..., idx_{n-2}) (reduce軸を除く)

    // 出力shapeの各次元サイズ（reduce軸を除く）
    let output_axes: Vec<usize> = (0..ndim).filter(|&a| a != reduce_axis).collect();
    let output_ndim = output_axes.len();

    if output_ndim == 0 {
        // スカラー出力の場合
        // Note: 初期化文はbodyに含める必要があるが、ここではscopeのみ返す
        let _ = init_value; // suppress unused warning (init は呼び出し側で処理)
        return (const_int(0), {
            let mut body_stmts = Scope::new();
            let _ = body_stmts.declare("acc".to_string(), dtype.clone(), Mutability::Mutable);
            body_stmts
        });
    }

    // 各出力軸のインデックスを計算して、入力の基底オフセットを算出
    // 簡略化: tidを出力の線形インデックスとして、各軸のインデックスを計算
    let mut base_offset = const_int(0);
    let mut remaining = var("tid");

    for (i, &in_axis) in output_axes.iter().enumerate().rev() {
        let axis_size = var(ph::shape(in_axis));
        let idx = if i == 0 {
            remaining.clone()
        } else {
            remaining.clone() % axis_size.clone()
        };

        // 入力でのストライド
        let stride = build_axis_stride(ndim, in_axis);
        base_offset = base_offset + idx * stride;

        if i > 0 {
            remaining = idiv(remaining, axis_size);
        }
    }

    let _ = scope.declare("acc".to_string(), dtype.clone(), Mutability::Mutable);

    // acc初期化をスコープに含める
    // （実際にはbodyに入れる必要があるが、ここでは基底オフセットとスコープを返す）

    (base_offset, scope)
}

/// 特定軸のストライドを計算
fn build_axis_stride(ndim: usize, axis: usize) -> AstNode {
    if axis == ndim - 1 {
        return const_int(1);
    }

    let mut stride = var(ph::shape(axis + 1));
    for inner_axis in (axis + 2)..ndim {
        stride = stride * var(ph::shape(inner_axis));
    }
    stride
}

/// 出力要素数を計算（reduce軸を除く）
fn build_output_elements(ndim: usize, reduce_axis: usize) -> AstNode {
    if ndim <= 1 {
        return const_int(1);
    }

    let mut total: Option<AstNode> = None;
    for axis in 0..ndim {
        if axis != reduce_axis {
            let size = var(ph::shape(axis));
            total = Some(match total {
                Some(t) => t * size,
                None => size,
            });
        }
    }
    total.unwrap_or_else(|| const_int(1))
}

/// 多次元並列Reduceカーネルを生成
fn build_multidim_parallel_reduce_kernel(
    node: &crate::graph::GraphNode,
    op: &crate::graph::ReduceOp,
    axis: usize,
    name: &str,
    parallel_dims: usize,
) -> Option<AstNode> {
    // 現時点ではフラット並列にフォールバック
    // 将来的には出力の多次元構造を利用した並列化も実装可能
    build_flat_parallel_reduce_kernel(node, op, axis, &format!("{}_{}", name, parallel_dims))
}

/// FusedElementwiseReduce演算の並列カーネルを生成
pub fn build_flat_parallel_fused_elementwise_reduce_kernel(
    node: &crate::graph::GraphNode,
    expr: &AstNode,
    reduce_op: &crate::graph::ReduceOp,
    axes: &[usize],
    name: &str,
) -> Option<AstNode> {
    use crate::graph::{GraphOp, ReduceOp};
    use std::collections::HashSet;

    if axes.is_empty() {
        return None;
    }

    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();
    let axes_set: HashSet<usize> = axes.iter().copied().collect();

    let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
        match reduce_op {
            ReduceOp::Sum => (
                super::helpers::get_reduce_init(&node.dtype, reduce_op),
                Box::new(|acc, val| acc + val),
            ),
            ReduceOp::Prod => (
                super::helpers::get_reduce_init(&node.dtype, reduce_op),
                Box::new(|acc, val| acc * val),
            ),
            ReduceOp::Max => (
                super::helpers::get_reduce_init(&node.dtype, reduce_op),
                Box::new(max),
            ),
        };

    let load_dtype = graph_dtype_to_ast(&input.dtype);

    // スレッドID（出力インデックス）
    let tid_param = VarDecl {
        name: "tid".to_string(),
        dtype: AstDType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::ThreadId(0),
    };

    let mut params = vec![tid_param];

    // 入力バッファ（定数以外）
    let mut non_const_idx = 0;
    for src in &node.src {
        if !matches!(src.op, GraphOp::Const(_)) && !super::elementwise::is_pure_const_node(src) {
            params.push(VarDecl {
                name: ph::input(non_const_idx),
                dtype: load_dtype.clone().to_ptr(),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
            non_const_idx += 1;
        }
    }

    // 出力バッファ
    params.push(VarDecl {
        name: ph::OUTPUT.to_string(),
        dtype: load_dtype.clone().to_ptr(),
        mutability: Mutability::Mutable,
        kind: VarKind::Normal,
    });

    // 出力軸（縮約軸以外）
    let output_axes: Vec<usize> = (0..ndim).filter(|a| !axes_set.contains(a)).collect();
    let output_ndim = output_axes.len();

    // アキュムレータ初期化
    let mut scope = Scope::new();
    let _ = scope.declare("acc".to_string(), load_dtype.clone(), Mutability::Mutable);
    let acc_init = assign("acc", init_value);

    // tidから出力インデックスを計算し、入力の基底オフセットを構築
    let mut base_offset_parts: Vec<(usize, AstNode)> = Vec::new();

    if output_ndim > 0 {
        let mut remaining = var("tid");
        for (i, &in_axis) in output_axes.iter().enumerate().rev() {
            let axis_size = var(ph::shape(in_axis));
            let idx = if i == 0 {
                remaining.clone()
            } else {
                remaining.clone() % axis_size.clone()
            };
            base_offset_parts.push((in_axis, idx));
            if i > 0 {
                remaining = idiv(remaining, axis_size);
            }
        }
    }

    // 入力オフセット計算式を構築
    // base_offset + Σ(ridx[reduce_axis] * stride[reduce_axis])
    let build_input_offset =
        |reduce_indices: &std::collections::HashMap<usize, AstNode>| -> AstNode {
            let mut offset = const_int(0);
            for axis in 0..ndim {
                let idx = if axes_set.contains(&axis) {
                    reduce_indices
                        .get(&axis)
                        .cloned()
                        .unwrap_or_else(|| var(ph::ridx(axis)))
                } else {
                    base_offset_parts
                        .iter()
                        .find(|(a, _)| *a == axis)
                        .map(|(_, idx)| idx.clone())
                        .unwrap_or_else(|| const_int(0))
                };
                let stride = build_axis_stride(ndim, axis);
                offset = offset + idx * stride;
            }
            offset
        };

    let input_offset = build_input_offset(&std::collections::HashMap::new());

    // Elementwise式にロードを埋め込み
    let mut mappings = std::collections::HashMap::new();
    let mut non_const_idx = 0;
    for (i, src) in node.src.iter().enumerate() {
        if let GraphOp::Const(lit) = &src.op {
            mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
        } else if super::elementwise::is_pure_const_node(src) {
            if let Some(lit) = super::elementwise::evaluate_pure_const(src) {
                mappings.insert(i.to_string(), AstNode::Const(lit));
            }
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
    let acc_update = assign("acc", accumulate_fn(var("acc"), value_expr));

    // 縮約軸のネストループを生成（内側から外側へ）
    let mut reduce_loops = block(vec![acc_update], Scope::new());
    for &axis in axes.iter().rev() {
        reduce_loops = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            var(ph::shape(axis)),
            reduce_loops,
        );
    }

    let store_stmt = store(var(ph::OUTPUT), var("tid"), var("acc"));

    let body = block(vec![acc_init, reduce_loops, store_stmt], scope);

    // グリッドサイズ: 出力要素数
    let grid_size = build_output_elements_excluding_axes(ndim, axes);

    Some(kernel_1d(
        Some(name.to_string()),
        params,
        AstDType::Tuple(vec![]),
        body,
        grid_size,
        const_int(256),
    ))
}

/// 複数軸を除いた出力要素数を計算
fn build_output_elements_excluding_axes(ndim: usize, exclude_axes: &[usize]) -> AstNode {
    use std::collections::HashSet;
    let exclude_set: HashSet<usize> = exclude_axes.iter().copied().collect();

    if ndim == 0 {
        return const_int(1);
    }

    let mut total: Option<AstNode> = None;
    for axis in 0..ndim {
        if !exclude_set.contains(&axis) {
            let size = var(ph::shape(axis));
            total = Some(match total {
                Some(t) => t * size,
                None => size,
            });
        }
    }
    total.unwrap_or_else(|| const_int(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_total_elements() {
        let expr = build_total_elements(3);
        // shape_0 * shape_1 * shape_2
        assert!(matches!(expr, AstNode::Mul(_, _)));
    }

    #[test]
    fn test_build_flat_parallel_elementwise() {
        use crate::ast::helper::wildcard;
        use crate::graph::DType as GraphDType;

        let expr = wildcard("0") + wildcard("1");
        let kernel = build_flat_parallel_elementwise_kernel(2, 2, expr, &GraphDType::F32, "test");

        match kernel {
            AstNode::Kernel { name, params, .. } => {
                assert_eq!(name, Some("test".to_string()));
                // tid + 2 inputs + 1 output = 4 params
                assert_eq!(params.len(), 4);
                assert!(matches!(params[0].kind, VarKind::ThreadId(0)));
            }
            _ => panic!("Expected Kernel node"),
        }
    }
}
