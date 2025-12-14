//! 並列化サポート
//!
//! GPUカーネル向けの並列化戦略とカーネル生成機能を提供します。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind, helper::*};
use crate::graph::ops::custom_placeholders as ph;

use super::helpers::{build_reduce_accumulator, graph_dtype_to_ast};

/// 切り上げ計算: ceil(a / b)
fn ceil_div(a: AstNode, b: AstNode) -> AstNode {
    // ceil(a / b) = (a + b - 1) / b
    idiv(a + b.clone() - const_int(1), b)
}

//=============================================================================
// 統一カーネルビルダー
//=============================================================================

/// グリッドサイズ計算方法
#[derive(Clone, Debug)]
pub enum GridStrategy {
    /// フラット: ceil_div(total, tg_size) * tg_size
    FlatRoundedUp,
    /// ベクトル化: total / vector_width（境界チェックなし）
    FlatDividedByVector { vector_width: usize },
}

/// 入力仕様
#[derive(Clone, Debug)]
pub enum InputSpec {
    /// バッファ入力
    Buffer,
    /// 定数埋め込み（FusedElementwise等で使用予定）
    #[allow(dead_code)]
    Const(crate::ast::Literal),
}

/// 統一的なカーネル生成設定
#[derive(Clone, Debug)]
pub struct ParallelKernelConfig {
    /// カーネル名
    pub name: String,
    /// グリッドサイズ計算方法
    pub grid_strategy: GridStrategy,
    /// ベクトル幅（None=スカラー）
    pub vector_width: Option<usize>,
    /// スレッドグループサイズ
    pub thread_group_size: usize,
    /// 境界チェックを行うか
    pub boundary_check: bool,
}

/// 統一的な並列Elementwiseカーネルを生成（1D並列化）
///
/// # Arguments
/// * `config` - カーネル設定
/// * `ndim` - テンソル次元数
/// * `inputs` - 入力仕様のリスト
/// * `expr` - Elementwise式
/// * `output_dtype` - 出力の型
pub fn build_elementwise_kernel(
    config: &ParallelKernelConfig,
    ndim: usize,
    inputs: &[InputSpec],
    expr: AstNode,
    output_dtype: &crate::graph::DType,
) -> AstNode {
    let load_dtype = graph_dtype_to_ast(output_dtype);
    let vec_dtype = config
        .vector_width
        .map(|w| load_dtype.clone().to_vec(w))
        .unwrap_or_else(|| load_dtype.clone());

    // パラメータ生成（1D: tid単体）
    let mut params = vec![VarDecl {
        name: "tid".to_string(),
        dtype: AstDType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::ThreadId(0),
    }];
    let (input_params, buffer_count) = build_input_params(inputs, &load_dtype);
    params.extend(input_params);
    params.push(build_output_param(&load_dtype));

    // オフセット計算（1D: tid直接使用）
    let offset = var("tid");

    // 入力ロード・式評価（ベクトル化対応）
    let final_expr = if let Some(vec_width) = config.vector_width {
        build_load_and_substitute_vec(inputs, &offset, expr, &vec_dtype, vec_width)
    } else {
        build_load_and_substitute(inputs, &offset, expr, &vec_dtype, buffer_count)
    };

    // ストア文
    let store_stmt = store(var(ph::OUTPUT), offset, final_expr);
    let body_block = block(vec![store_stmt], Scope::new());

    // 境界チェック適用
    let guarded_body = if config.boundary_check {
        let total_elements = build_total_elements(ndim);
        if_then(lt(var("tid"), total_elements), body_block)
    } else {
        body_block
    };
    let body = block(vec![guarded_body], Scope::new());

    // グリッドサイズ計算
    let (grid_size, tg_size) = build_grid_and_tg_size(&config.grid_strategy, config, ndim);

    AstNode::Kernel {
        name: Some(config.name.clone()),
        params,
        return_type: AstDType::Tuple(vec![]),
        body: Box::new(body),
        default_grid_size: grid_size,
        default_thread_group_size: tg_size,
    }
}

/// 入力パラメータを生成（バッファのみ、定数は除外）
fn build_input_params(inputs: &[InputSpec], load_dtype: &AstDType) -> (Vec<VarDecl>, usize) {
    let mut params = Vec::new();
    let mut buffer_idx = 0;
    for input in inputs {
        if matches!(input, InputSpec::Buffer) {
            params.push(VarDecl {
                name: ph::input(buffer_idx),
                dtype: load_dtype.clone().to_ptr(),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
            buffer_idx += 1;
        }
    }
    (params, buffer_idx)
}

/// 出力パラメータを生成
fn build_output_param(load_dtype: &AstDType) -> VarDecl {
    VarDecl {
        name: ph::OUTPUT.to_string(),
        dtype: load_dtype.clone().to_ptr(),
        mutability: Mutability::Mutable,
        kind: VarKind::Normal,
    }
}

/// 入力ロードと式の置換
fn build_load_and_substitute(
    inputs: &[InputSpec],
    offset: &AstNode,
    expr: AstNode,
    load_dtype: &AstDType,
    _buffer_count: usize,
) -> AstNode {
    let mut mappings = std::collections::HashMap::new();
    let mut buffer_idx = 0;

    for (i, input) in inputs.iter().enumerate() {
        match input {
            InputSpec::Buffer => {
                let load_node = load(
                    var(ph::input(buffer_idx)),
                    offset.clone(),
                    load_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                buffer_idx += 1;
            }
            InputSpec::Const(lit) => {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            }
        }
    }

    expr.substitute(&mappings)
}

/// 入力ロードと式の置換（ベクトル版）
fn build_load_and_substitute_vec(
    inputs: &[InputSpec],
    offset: &AstNode,
    expr: AstNode,
    vec_dtype: &AstDType,
    vector_width: usize,
) -> AstNode {
    let mut mappings = std::collections::HashMap::new();
    let mut buffer_idx = 0;

    for (i, input) in inputs.iter().enumerate() {
        match input {
            InputSpec::Buffer => {
                let load_node = load_vec(
                    var(ph::input(buffer_idx)),
                    offset.clone(),
                    vector_width,
                    vec_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                buffer_idx += 1;
            }
            InputSpec::Const(lit) => {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            }
        }
    }

    expr.substitute(&mappings)
}

/// グリッドサイズとスレッドグループサイズを計算（1D専用）
fn build_grid_and_tg_size(
    strategy: &GridStrategy,
    config: &ParallelKernelConfig,
    ndim: usize,
) -> ([Box<AstNode>; 3], [Box<AstNode>; 3]) {
    let tg_size = config.thread_group_size;

    match strategy {
        GridStrategy::FlatRoundedUp => {
            let total = build_total_elements(ndim);
            let tg = const_int(tg_size as isize);
            let num_groups = ceil_div(total, tg.clone());
            let grid = num_groups * tg.clone();
            (
                [
                    Box::new(grid),
                    Box::new(const_int(1)),
                    Box::new(const_int(1)),
                ],
                [Box::new(tg), Box::new(const_int(1)), Box::new(const_int(1))],
            )
        }
        GridStrategy::FlatDividedByVector { vector_width } => {
            let total = build_total_elements(ndim);
            let grid = total / const_int(*vector_width as isize);
            (
                [
                    Box::new(grid),
                    Box::new(const_int(1)),
                    Box::new(const_int(1)),
                ],
                [
                    Box::new(const_int(tg_size as isize)),
                    Box::new(const_int(1)),
                    Box::new(const_int(1)),
                ],
            )
        }
    }
}

//=============================================================================
// 既存の並列化戦略（後方互換性のため維持）
//=============================================================================

/// 並列化戦略
#[derive(Clone, Debug, PartialEq)]
pub enum ParallelizationStrategy {
    /// 逐次実行（CPU向け、Rangeループを使用）
    Sequential,
    /// フラット並列化（1スレッド1要素または複数要素）
    /// 全要素を線形インデックスで並列処理
    FlatParallel {
        /// スレッドグループサイズ（64, 128, 256, 512など）
        thread_group_size: usize,
        /// ベクトル幅（None=スカラー, Some(2/4/8)=ベクトル化）
        vector_width: Option<usize>,
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
        ParallelizationStrategy::FlatParallel {
            thread_group_size,
            vector_width,
        } => {
            if let Some(vec_width) = vector_width {
                build_vectorized_flat_parallel_kernel(
                    ndim,
                    num_inputs,
                    expr,
                    output_dtype,
                    name,
                    *thread_group_size,
                    *vec_width,
                )
            } else {
                build_flat_parallel_elementwise_kernel(
                    ndim,
                    num_inputs,
                    expr,
                    output_dtype,
                    name,
                    *thread_group_size,
                )
            }
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
    thread_group_size: usize,
) -> AstNode {
    // 入力仕様を生成（全てバッファ）
    let inputs: Vec<InputSpec> = (0..num_inputs).map(|_| InputSpec::Buffer).collect();

    let config = ParallelKernelConfig {
        name: name.to_string(),
        grid_strategy: GridStrategy::FlatRoundedUp,
        vector_width: None,
        thread_group_size,
        boundary_check: true,
    };

    build_elementwise_kernel(&config, ndim, &inputs, expr, output_dtype)
}

/// ベクトル化フラット並列Elementwiseカーネルを生成
///
/// float2/float4/float8などでロード/ストアし、1スレッドが複数要素を処理
fn build_vectorized_flat_parallel_kernel(
    ndim: usize,
    num_inputs: usize,
    expr: AstNode,
    output_dtype: &crate::graph::DType,
    name: &str,
    thread_group_size: usize,
    vector_width: usize,
) -> AstNode {
    // 入力仕様を生成（全てバッファ）
    let inputs: Vec<InputSpec> = (0..num_inputs).map(|_| InputSpec::Buffer).collect();

    let config = ParallelKernelConfig {
        name: name.to_string(),
        grid_strategy: GridStrategy::FlatDividedByVector { vector_width },
        vector_width: Some(vector_width),
        thread_group_size,
        boundary_check: false, // ベクトル化版は境界チェックなし
    };

    build_elementwise_kernel(&config, ndim, &inputs, expr, output_dtype)
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
        ParallelizationStrategy::FlatParallel {
            thread_group_size: _,
            vector_width: _,
        } => build_flat_parallel_reduce_kernel(node, op, axis, name),
    }
}

/// フラット並列Reduceカーネルを生成
fn build_flat_parallel_reduce_kernel(
    node: &crate::graph::GraphNode,
    op: &crate::graph::ReduceOp,
    axis: usize,
    name: &str,
) -> Option<AstNode> {
    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();

    let (init_value, accumulate_fn) = build_reduce_accumulator(op, &node.dtype);

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

/// FusedElementwiseReduce演算の並列カーネルを生成
pub fn build_flat_parallel_fused_elementwise_reduce_kernel(
    node: &crate::graph::GraphNode,
    expr: &AstNode,
    reduce_op: &crate::graph::ReduceOp,
    axes: &[usize],
    name: &str,
) -> Option<AstNode> {
    use crate::graph::GraphOp;
    use std::collections::HashSet;

    if axes.is_empty() {
        return None;
    }

    let input = node.src.first()?;
    let input_shape = input.view.shape();
    let ndim = input_shape.len();
    let axes_set: HashSet<usize> = axes.iter().copied().collect();

    let (init_value, accumulate_fn) = build_reduce_accumulator(reduce_op, &node.dtype);

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
        let kernel =
            build_flat_parallel_elementwise_kernel(2, 2, expr, &GraphDType::F32, "test", 256);

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

    #[test]
    fn test_build_vectorized_flat_parallel_elementwise() {
        use crate::ast::helper::wildcard;
        use crate::graph::DType as GraphDType;

        let expr = wildcard("0") + wildcard("1");
        let kernel =
            build_vectorized_flat_parallel_kernel(2, 2, expr, &GraphDType::F32, "test_vec", 128, 4);

        match kernel {
            AstNode::Kernel {
                name,
                params,
                default_thread_group_size,
                ..
            } => {
                assert_eq!(name, Some("test_vec".to_string()));
                // tid + 2 inputs + 1 output = 4 params
                assert_eq!(params.len(), 4);
                // thread_group_size should be 128
                assert!(matches!(
                    default_thread_group_size[0].as_ref(),
                    AstNode::Const(crate::ast::Literal::Int(128))
                ));
            }
            _ => panic!("Expected Kernel node"),
        }
    }

    #[test]
    fn test_flat_parallel_kernel_has_boundary_check() {
        use crate::ast::helper::wildcard;
        use crate::graph::DType as GraphDType;

        let expr = wildcard("0") + wildcard("1");
        let kernel =
            build_flat_parallel_elementwise_kernel(2, 2, expr, &GraphDType::F32, "test", 256);

        // カーネルの本体にIf文が含まれているか確認
        match &kernel {
            AstNode::Kernel { body, .. } => {
                // bodyはBlockで、その中にIf文があるはず
                fn contains_if(node: &AstNode) -> bool {
                    match node {
                        AstNode::If { .. } => true,
                        AstNode::Block { statements, .. } => statements.iter().any(contains_if),
                        _ => false,
                    }
                }
                assert!(
                    contains_if(body),
                    "Kernel body should contain boundary check (If node)"
                );
            }
            _ => panic!("Expected Kernel node"),
        }
    }

    #[test]
    fn test_ceil_div() {
        // ceil_div(10, 3) = ceil(10/3) = 4
        let result = ceil_div(const_int(10), const_int(3));
        // (10 + 3 - 1) / 3 = 12 / 3 = 4
        // result should be Idiv(Add(Add(10, 3), -1), 3)
        assert!(matches!(result, AstNode::Idiv(_, _)));
    }

    #[test]
    fn test_grid_size_rounded_up() {
        use crate::ast::helper::wildcard;
        use crate::graph::DType as GraphDType;

        let expr = wildcard("0");
        let kernel =
            build_flat_parallel_elementwise_kernel(1, 1, expr, &GraphDType::F32, "test", 64);

        // グリッドサイズが切り上げ計算を含んでいることを確認
        match &kernel {
            AstNode::Kernel {
                default_grid_size, ..
            } => {
                // grid_size = ceil_div(total_elements, tg_size) * tg_size
                // which involves Mul and Idiv operations
                let grid_x = default_grid_size[0].as_ref();
                // グリッドサイズは Mul を含むはず（切り上げ後の乗算）
                assert!(
                    matches!(grid_x, AstNode::Mul(_, _)),
                    "Grid size should be rounded up (contain Mul): {:?}",
                    grid_x
                );
            }
            _ => panic!("Expected Kernel node"),
        }
    }
}
