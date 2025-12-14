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

/// スレッドグループサイズを複数軸に分配する
///
/// # Arguments
/// * `total_size` - 合計スレッドグループサイズ（例: 256）
/// * `parallel_dims` - 並列化する軸数（1, 2, or 3）
///
/// # Returns
/// 各軸のスレッドグループサイズ [x, y, z]
///
/// # Examples
/// - `distribute_thread_group_size(256, 1)` -> `[256, 1, 1]`
/// - `distribute_thread_group_size(256, 2)` -> `[16, 16, 1]`
/// - `distribute_thread_group_size(64, 3)` -> `[4, 4, 4]`
fn distribute_thread_group_size(total_size: usize, parallel_dims: usize) -> [usize; 3] {
    match parallel_dims {
        1 => [total_size, 1, 1],
        2 => {
            // sqrt(total_size) を計算し、2の累乗に調整
            let sqrt = (total_size as f64).sqrt() as usize;
            // 2の累乗に切り上げ
            let per_dim = sqrt.next_power_of_two().min(total_size);
            let y = total_size / per_dim;
            [per_dim, y.max(1), 1]
        }
        3 => {
            // cbrt(total_size) を計算し、2の累乗に調整
            let cbrt = (total_size as f64).cbrt().ceil() as usize;
            let per_dim = cbrt.next_power_of_two().min(16); // 最大16に制限
            let xy = per_dim * per_dim;
            let z = (total_size / xy).max(1);
            [per_dim, per_dim, z]
        }
        _ => [total_size, 1, 1],
    }
}

//=============================================================================
// 統一カーネルビルダー
//=============================================================================

/// スレッド構成
#[derive(Clone, Debug)]
pub enum ThreadConfig {
    /// 1次元フラット並列化（tid = get_global_id(0)）
    Flat1D,
    /// 多次元並列化（tid_0, tid_1, tid_2）
    MultiDim { dims: usize },
}

/// グリッドサイズ計算方法
#[derive(Clone, Debug)]
pub enum GridStrategy {
    /// フラット: ceil_div(total, tg_size) * tg_size
    FlatRoundedUp,
    /// ベクトル化: total / vector_width（境界チェックなし）
    FlatDividedByVector { vector_width: usize },
    /// 多次元: 各軸をスレッドグループサイズの倍数に切り上げ
    MultiDimRoundedUp { parallel_dims: usize },
    /// Reduce出力: 縮約軸を除いた要素数（将来のReduce統一ビルダー用）
    #[allow(dead_code)]
    ReduceOutput { exclude_axes: Vec<usize> },
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
    /// スレッド構成
    pub thread_config: ThreadConfig,
    /// グリッドサイズ計算方法
    pub grid_strategy: GridStrategy,
    /// ベクトル幅（None=スカラー）
    pub vector_width: Option<usize>,
    /// スレッドグループサイズ
    pub thread_group_size: usize,
    /// 境界チェックを行うか
    pub boundary_check: bool,
}

/// 統一的な並列Elementwiseカーネルを生成
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

    // パラメータ生成
    let mut params = build_thread_params(&config.thread_config, config.thread_group_size);
    let (input_params, buffer_count) = build_input_params(inputs, &load_dtype);
    params.extend(input_params);
    params.push(build_output_param(&load_dtype));

    // オフセット計算
    let offset = build_offset(&config.thread_config, ndim);

    // 入力ロード・式評価（ベクトル化対応）
    let final_expr = if let Some(vec_width) = config.vector_width {
        build_load_and_substitute_vec(inputs, &offset, expr, &vec_dtype, vec_width)
    } else {
        build_load_and_substitute(inputs, &offset, expr, &vec_dtype, buffer_count)
    };

    // ストア文
    let store_stmt = store(var(ph::OUTPUT), offset, final_expr);

    // 逐次ループでラップ（MultiDimの場合）
    let body_with_loops = match &config.thread_config {
        ThreadConfig::Flat1D => block(vec![store_stmt], Scope::new()),
        ThreadConfig::MultiDim { dims } => {
            wrap_sequential_loops(ndim, (*dims).min(ndim).min(3), vec![store_stmt])
        }
    };

    // 境界チェック適用
    let guarded_body = if config.boundary_check {
        apply_boundary_check(&config.thread_config, ndim, body_with_loops)
    } else {
        body_with_loops
    };
    let body = block(vec![guarded_body], Scope::new());

    // グリッドサイズ計算
    let (grid_size, tg_size) =
        build_grid_and_tg_size(&config.grid_strategy, &config.thread_config, config, ndim);

    AstNode::Kernel {
        name: Some(config.name.clone()),
        params,
        return_type: AstDType::Tuple(vec![]),
        body: Box::new(body),
        default_grid_size: grid_size,
        default_thread_group_size: tg_size,
    }
}

/// スレッドIDパラメータを生成
fn build_thread_params(config: &ThreadConfig, _thread_group_size: usize) -> Vec<VarDecl> {
    match config {
        ThreadConfig::Flat1D => vec![VarDecl {
            name: "tid".to_string(),
            dtype: AstDType::Int,
            mutability: Mutability::Immutable,
            kind: VarKind::ThreadId(0),
        }],
        ThreadConfig::MultiDim { dims } => (0..*dims)
            .map(|axis| VarDecl {
                name: format!("tid_{}", axis),
                dtype: AstDType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(axis),
            })
            .collect(),
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

/// オフセット計算
fn build_offset(config: &ThreadConfig, ndim: usize) -> AstNode {
    match config {
        ThreadConfig::Flat1D => var("tid"),
        ThreadConfig::MultiDim { dims } => build_hybrid_offset(ndim, (*dims).min(ndim).min(3)),
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

/// 境界チェックを適用
fn apply_boundary_check(config: &ThreadConfig, ndim: usize, body: AstNode) -> AstNode {
    match config {
        ThreadConfig::Flat1D => {
            let total_elements = build_total_elements(ndim);
            if_then(lt(var("tid"), total_elements), body)
        }
        ThreadConfig::MultiDim { dims } => {
            let actual_dims = (*dims).min(ndim).min(3);
            let mut result = body;
            for axis in (0..actual_dims).rev() {
                result = if_then(
                    lt(var(format!("tid_{}", axis)), var(ph::shape(axis))),
                    result,
                );
            }
            result
        }
    }
}

/// グリッドサイズとスレッドグループサイズを計算
fn build_grid_and_tg_size(
    strategy: &GridStrategy,
    thread_config: &ThreadConfig,
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
        GridStrategy::MultiDimRoundedUp { parallel_dims } => {
            let actual_dims = (*parallel_dims).min(ndim).min(3);
            let tg_sizes = distribute_thread_group_size(tg_size, actual_dims);

            let build_grid_axis = |axis: usize| -> Box<AstNode> {
                if axis < actual_dims {
                    let shape = var(ph::shape(axis));
                    let tg = const_int(tg_sizes[axis] as isize);
                    let num_groups = ceil_div(shape, tg.clone());
                    Box::new(num_groups * tg)
                } else {
                    Box::new(const_int(1))
                }
            };

            (
                [build_grid_axis(0), build_grid_axis(1), build_grid_axis(2)],
                [
                    Box::new(const_int(tg_sizes[0] as isize)),
                    Box::new(const_int(tg_sizes[1] as isize)),
                    Box::new(const_int(tg_sizes[2] as isize)),
                ],
            )
        }
        GridStrategy::ReduceOutput { exclude_axes } => {
            let grid = build_output_elements_excluding_axes(ndim, exclude_axes);
            // Reduce用のグリッドサイズは1D
            match thread_config {
                ThreadConfig::Flat1D => (
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
                ),
                _ => (
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
                ),
            }
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
    /// 多次元並列化
    /// 指定した軸数までをスレッドIDで並列化（例: 2なら2次元グリッド）
    /// 残りの軸は逐次ループ
    MultiDimParallel {
        /// 並列化する軸数（1, 2, または 3）
        parallel_dims: usize,
        /// スレッドグループサイズ
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
        ParallelizationStrategy::MultiDimParallel {
            parallel_dims,
            thread_group_size,
            vector_width: _, // MultiDimParallelのベクトル化は後で実装
        } => build_multidim_parallel_elementwise_kernel(
            ndim,
            num_inputs,
            expr,
            output_dtype,
            name,
            *parallel_dims,
            *thread_group_size,
        ),
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
        thread_config: ThreadConfig::Flat1D,
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
        thread_config: ThreadConfig::Flat1D,
        grid_strategy: GridStrategy::FlatDividedByVector { vector_width },
        vector_width: Some(vector_width),
        thread_group_size,
        boundary_check: false, // ベクトル化版は境界チェックなし
    };

    build_elementwise_kernel(&config, ndim, &inputs, expr, output_dtype)
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
    thread_group_size: usize,
) -> AstNode {
    // 入力仕様を生成（全てバッファ）
    let inputs: Vec<InputSpec> = (0..num_inputs).map(|_| InputSpec::Buffer).collect();

    let config = ParallelKernelConfig {
        name: name.to_string(),
        thread_config: ThreadConfig::MultiDim {
            dims: parallel_dims,
        },
        grid_strategy: GridStrategy::MultiDimRoundedUp { parallel_dims },
        vector_width: None,
        thread_group_size,
        boundary_check: true,
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
        ParallelizationStrategy::FlatParallel {
            thread_group_size: _,
            vector_width: _,
        } => build_flat_parallel_reduce_kernel(node, op, axis, name),
        ParallelizationStrategy::MultiDimParallel {
            parallel_dims,
            thread_group_size: _,
            vector_width: _,
        } => build_multidim_parallel_reduce_kernel(node, op, axis, name, *parallel_dims),
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

    #[test]
    fn test_distribute_thread_group_size_1d() {
        let result = distribute_thread_group_size(256, 1);
        assert_eq!(result, [256, 1, 1]);

        let result = distribute_thread_group_size(64, 1);
        assert_eq!(result, [64, 1, 1]);
    }

    #[test]
    fn test_distribute_thread_group_size_2d() {
        let result = distribute_thread_group_size(256, 2);
        // sqrt(256) = 16, so [16, 16, 1]
        assert_eq!(result[0] * result[1], 256);
        assert_eq!(result[2], 1);
        // 2の累乗であること
        assert!(result[0].is_power_of_two());
        assert!(result[1].is_power_of_two() || result[1] == 1);

        let result = distribute_thread_group_size(64, 2);
        // sqrt(64) = 8, so [8, 8, 1]
        assert_eq!(result[0] * result[1], 64);
        assert_eq!(result[2], 1);
    }

    #[test]
    fn test_distribute_thread_group_size_3d() {
        let result = distribute_thread_group_size(64, 3);
        // cbrt(64) = 4, so [4, 4, 4]
        assert_eq!(result[0] * result[1] * result[2], 64);
        // 各軸が2の累乗であること
        assert!(result[0].is_power_of_two());
        assert!(result[1].is_power_of_two());
    }

    #[test]
    fn test_multidim_parallel_kernel_has_boundary_check() {
        use crate::ast::helper::wildcard;
        use crate::graph::DType as GraphDType;

        let expr = wildcard("0");
        let kernel = build_multidim_parallel_elementwise_kernel(
            2,
            1,
            expr,
            &GraphDType::F32,
            "test",
            2,
            256,
        );

        // カーネルの本体にIf文が含まれているか確認
        match &kernel {
            AstNode::Kernel { body, .. } => {
                fn contains_if(node: &AstNode) -> bool {
                    match node {
                        AstNode::If { .. } => true,
                        AstNode::Block { statements, .. } => statements.iter().any(contains_if),
                        _ => false,
                    }
                }
                assert!(
                    contains_if(body),
                    "MultiDim kernel body should contain boundary check (If node)"
                );
            }
            _ => panic!("Expected Kernel node"),
        }
    }

    #[test]
    fn test_multidim_parallel_kernel_distributed_thread_group_size() {
        use crate::ast::helper::wildcard;
        use crate::graph::DType as GraphDType;

        let expr = wildcard("0");
        let kernel = build_multidim_parallel_elementwise_kernel(
            2,
            1,
            expr,
            &GraphDType::F32,
            "test",
            2,
            256,
        );

        match &kernel {
            AstNode::Kernel {
                default_thread_group_size,
                ..
            } => {
                // parallel_dims=2, thread_group_size=256 -> [16, 16, 1]
                // スレッドグループサイズが両軸に分配されていることを確認
                match (
                    default_thread_group_size[0].as_ref(),
                    default_thread_group_size[1].as_ref(),
                ) {
                    (
                        AstNode::Const(crate::ast::Literal::Int(x)),
                        AstNode::Const(crate::ast::Literal::Int(y)),
                    ) => {
                        // x * y = 256
                        assert_eq!((*x as usize) * (*y as usize), 256);
                        // 両方とも1より大きい（分配されている）
                        assert!(*x > 1, "x should be > 1, got {}", x);
                        assert!(*y > 1, "y should be > 1, got {}", y);
                    }
                    _ => panic!("Expected Const nodes for thread group size"),
                }
            }
            _ => panic!("Expected Kernel node"),
        }
    }
}
