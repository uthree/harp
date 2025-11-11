//! 高レベル演算のヘルパー関数
//!
//! このモジュールは既存の基本的なグラフ演算を組み合わせて、
//! より高レベルな数学的演算や便利な演算を提供します。

use crate::graph::GraphNode;
use crate::graph::ops::{ElementwiseOp, GraphOp, max, reduce_sum};

/// 二乗: x^2
///
/// # 例
/// ```no_run
/// use harp::prelude::*;
/// use harp::graph::hlops;
///
/// let mut graph = Graph::new();
/// let x = graph.input("x")
///     .with_dtype(DType::F32)
///     .with_shape(vec![10, 20])
///     .build();
///
/// let x_squared = hlops::square(x);
/// ```
pub fn square(x: GraphNode) -> GraphNode {
    x.clone() * x
}

/// 立方: x^3
pub fn cube(x: GraphNode) -> GraphNode {
    x.clone() * x.clone() * x
}

/// 累乗: x^n (正の整数のみ)
///
/// # パニック
/// n が 0 の場合はパニックします
pub fn powi(x: GraphNode, n: u32) -> GraphNode {
    assert!(n > 0, "powi: n must be positive");

    if n == 1 {
        return x;
    }

    let mut result = x.clone();
    for _ in 1..n {
        result = result * x.clone();
    }
    result
}

/// 絶対値の二乗: x^2 (常に非負)
///
/// Note: 本物の絶対値を実装するには max(x, -x) が必要ですが、
/// 現在の実装では異なるテンソル間の要素ごとの最大値しかサポートしていないため、
/// 代わりに二乗を使って非負の値を得ることができます。
pub fn abs_square(x: GraphNode) -> GraphNode {
    square(x)
}

/// 2つのテンソルの要素ごとの最小値: min(a, b) = -max(-a, -b)
pub fn min(a: GraphNode, b: GraphNode) -> GraphNode {
    -max(-a, -b)
}

/// クランプ: min_val <= x <= max_val に制限
///
/// # 例
/// ```no_run
/// use harp::prelude::*;
/// use harp::graph::hlops;
///
/// let mut graph = Graph::new();
/// let x = graph.input("x")
///     .with_dtype(DType::F32)
///     .with_shape(vec![10])
///     .build();
/// let min_val = graph.input("min")
///     .with_dtype(DType::F32)
///     .with_shape(vec![10])
///     .build();
/// let max_val = graph.input("max")
///     .with_dtype(DType::F32)
///     .with_shape(vec![10])
///     .build();
///
/// let clamped = hlops::clamp(x, min_val, max_val);
/// ```
pub fn clamp(x: GraphNode, min_val: GraphNode, max_val: GraphNode) -> GraphNode {
    min(max(x, min_val), max_val)
}

/// 平均を計算: mean(x, axis)
///
/// 指定された軸に沿った平均を計算します。
/// sum(x, axis) / size(axis) として実装されます。
///
/// Note: 現在の実装では、軸のサイズが定数（Expr::Const）の場合のみサポートされます。
/// 変数サイズの軸に対してはパニックします。
pub fn mean(x: GraphNode, axis: usize) -> GraphNode {
    use crate::graph::shape::Expr;

    let shape = x.view.shape();
    if axis >= shape.len() {
        panic!("mean: axis {} is out of bounds for shape {:?}", axis, shape);
    }

    // 軸のサイズを取得
    let axis_size = &shape[axis];

    // 軸サイズが定数の場合のみ処理
    let size_value = match axis_size {
        Expr::Const(n) => *n as f32,
        _ => panic!(
            "mean: axis size must be constant, got symbolic expression: {:?}",
            axis_size
        ),
    };

    // 合計を計算
    let sum = reduce_sum(x.clone(), axis);

    // 定数ノード（1/size）を作成
    let inv_size = GraphNode::constant(1.0f32 / size_value);

    // sumの各要素に1/sizeを掛ける
    // 定数（スカラー）をsumのshapeにexpand
    // まずスカラーを必要な次元数まで unsqueeze してから expand
    let sum_ndim = sum.view.ndim();
    let mut view = inv_size.view.clone();
    for _ in 0..sum_ndim {
        view = view.unsqueeze(0);
    }
    view = view.expand(sum.view.shape().to_vec());
    let expanded_inv_size = inv_size.view(view);

    sum * expanded_inv_size
}

/// 分散を計算: var(x, axis)
///
/// 不偏分散を計算します: E[(x - mean(x))^2]
///
/// Note: この実装は効率的ではありません（meanを2回計算）。
/// より効率的な実装には E[x^2] - E[x]^2 を使用できますが、
/// 数値安定性の問題があります。
pub fn variance(x: GraphNode, axis: usize) -> GraphNode {
    // 平均を計算
    let x_mean = mean(x.clone(), axis);

    // meanの次元を復元してbroadcast可能にする
    let x_mean_expanded = x_mean.view(
        x_mean
            .view
            .clone()
            .unsqueeze(axis)
            .expand(x.view.shape().to_vec()),
    );

    // (x - mean)^2 の平均を計算
    mean(square(x - x_mean_expanded), axis)
}

/// 行列積 (2次元テンソル限定): C[i,j] = sum_k(A[i,k] * B[k,j])
///
/// # パニック
/// - a が 2次元テンソルでない場合
/// - b が 2次元テンソルでない場合
/// - a の列数と b の行数が一致しない場合
///
/// # 実装の詳細
/// 1. a を [i, k] -> [i, k, 1] に reshape (unsqueeze)
/// 2. b を [k, j] -> [1, k, j] に reshape (unsqueeze)
/// 3. a と b を broadcast して [i, k, j] にする
/// 4. 要素ごとの積を取る: [i, k, j]
/// 5. k 軸で reduce_sum: [i, j]
///
/// Note: 現在の実装ではbroadcastとunsqueezeが必要なため、
/// 将来的な実装として残します。
pub fn matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    let a_shape = a.view.shape();
    let b_shape = b.view.shape();

    // 2次元チェック
    assert_eq!(a_shape.len(), 2, "matmul: first argument must be 2D");
    assert_eq!(b_shape.len(), 2, "matmul: second argument must be 2D");

    // 次元の互換性チェック
    assert_eq!(
        a_shape[1], b_shape[0],
        "matmul: incompatible dimensions: {:?} and {:?}",
        a_shape, b_shape
    );

    // TODO: unsqueeze, expand, broadcastの実装が必要
    // 現在は実装を保留
    panic!("matmul: not yet implemented - requires unsqueeze and broadcast operations");
}

/// バッチ行列積 (3次元テンソル): C[b,i,j] = sum_k(A[b,i,k] * B[b,k,j])
///
/// バッチ次元を持つ行列積を計算します。
///
/// # パニック
/// - a が 3次元テンソルでない場合
/// - b が 3次元テンソルでない場合
/// - バッチサイズが一致しない場合
/// - 行列の次元が互換性がない場合
pub fn batch_matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    let a_shape = a.view.shape();
    let b_shape = b.view.shape();

    // 3次元チェック
    assert_eq!(a_shape.len(), 3, "batch_matmul: first argument must be 3D");
    assert_eq!(b_shape.len(), 3, "batch_matmul: second argument must be 3D");

    // バッチサイズチェック
    assert_eq!(
        a_shape[0], b_shape[0],
        "batch_matmul: batch sizes must match: {:?} vs {:?}",
        a_shape[0], b_shape[0]
    );

    // 行列次元の互換性チェック
    assert_eq!(
        a_shape[2], b_shape[1],
        "batch_matmul: incompatible matrix dimensions: {:?} and {:?}",
        a_shape, b_shape
    );

    // TODO: unsqueeze, expand, broadcastの実装が必要
    panic!("batch_matmul: not yet implemented - requires unsqueeze and broadcast operations");
}

// ============================================================================
// 基本的な数学関数
// ============================================================================

/// 底が2の対数: log2(x)
pub fn log2(x: GraphNode) -> GraphNode {
    let dtype = x.dtype.clone();
    let view = x.view.clone();
    GraphNode::new(
        dtype,
        GraphOp::Elementwise {
            op: ElementwiseOp::Log2,
            elementwise_strategies: None,
        },
        vec![x],
        view,
    )
}

/// 2の累乗: 2^x
pub fn exp2(x: GraphNode) -> GraphNode {
    let dtype = x.dtype.clone();
    let view = x.view.clone();
    GraphNode::new(
        dtype,
        GraphOp::Elementwise {
            op: ElementwiseOp::Exp2,
            elementwise_strategies: None,
        },
        vec![x],
        view,
    )
}

/// 自然対数: ln(x) = log(x)
///
/// log2を使って実装: log(x) = log2(x) / log2(e)
pub fn log(x: GraphNode) -> GraphNode {
    // log(x) = log2(x) * (1 / log2(e))
    // 1 / log2(e) ≈ 0.6931471805599453
    const INV_LOG2_E: f32 = 1.0 / std::f32::consts::LOG2_E;

    let log2_x = log2(x);
    let inv_log2_e = GraphNode::constant(INV_LOG2_E);

    // 定数（スカラー）をlog2_xのshapeにexpand
    let ndim = log2_x.view.ndim();
    let mut view = inv_log2_e.view.clone();
    for _ in 0..ndim {
        view = view.unsqueeze(0);
    }
    view = view.expand(log2_x.view.shape().to_vec());
    let expanded_const = inv_log2_e.view(view);

    log2_x * expanded_const
}

/// 指数関数: e^x = exp(x)
///
/// exp2を使って実装: exp(x) = 2^(x * log2(e))
pub fn exp(x: GraphNode) -> GraphNode {
    // exp(x) = 2^(x * log2(e))
    const LOG2_E: f32 = std::f32::consts::LOG2_E;

    let log2_e = GraphNode::constant(LOG2_E);

    // 定数（スカラー）をxのshapeにexpand
    let ndim = x.view.ndim();
    let mut view = log2_e.view.clone();
    for _ in 0..ndim {
        view = view.unsqueeze(0);
    }
    view = view.expand(x.view.shape().to_vec());
    let expanded_const = log2_e.view(view);

    exp2(x * expanded_const)
}

// ============================================================================
// 以下は基本的な数学関数がElementwiseOpに追加された後に実装可能な関数
// ============================================================================

// /// 正規化線形関数: ReLU(x) = max(0, x)
// pub fn relu(x: GraphNode) -> GraphNode {
//     // TODO: 0定数ノードの実装が必要
//     max(x, zero_like(x))
// }

// /// Leaky ReLU: LeakyReLU(x, alpha) = max(alpha * x, x)
// pub fn leaky_relu(x: GraphNode, alpha: GraphNode) -> GraphNode {
//     max(alpha * x.clone(), x)
// }

// /// シグモイド関数: sigmoid(x) = 1 / (1 + exp(-x))
// ///
// /// Note: exp関数がElementwiseOpに追加される必要があります
// pub fn sigmoid(x: GraphNode) -> GraphNode {
//     // TODO: exp と const(1) の実装が必要
//     // recip(const(1) + exp(-x))
//     unimplemented!("sigmoid: requires exp operation")
// }

// /// 双曲線正接: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// ///
// /// あるいは: tanh(x) = 2*sigmoid(2*x) - 1
// ///
// /// Note: exp関数がElementwiseOpに追加される必要があります
// pub fn tanh(x: GraphNode) -> GraphNode {
//     // TODO: exp の実装が必要
//     unimplemented!("tanh: requires exp operation")
// }

// /// 正接: tan(x) = sin(x) / cos(x)
// ///
// /// Note: sin, cos関数がElementwiseOpに追加される必要があります
// pub fn tan(x: GraphNode) -> GraphNode {
//     // TODO: sin, cos の実装が必要
//     // sin(x) / cos(x)
//     unimplemented!("tan: requires sin and cos operations")
// }

// /// ソフトプラス: softplus(x) = log(1 + exp(x))
// ///
// /// Note: log, exp関数がElementwiseOpに追加される必要があります
// pub fn softplus(x: GraphNode) -> GraphNode {
//     // TODO: log, exp の実装が必要
//     unimplemented!("softplus: requires log and exp operations")
// }

// /// ソフトマックス: softmax(x, axis) = exp(x - max(x)) / sum(exp(x - max(x)), axis)
// ///
// /// 数値安定性のため、max(x)を引いてから計算します。
// ///
// /// Note: exp関数がElementwiseOpに追加される必要があります
// pub fn softmax(x: GraphNode, axis: usize) -> GraphNode {
//     // TODO: exp, max(reduce), expand の実装が必要
//     unimplemented!("softmax: requires exp operation and broadcasting")
// }

// /// ログソフトマックス: log_softmax(x, axis) = x - log(sum(exp(x), axis))
// ///
// /// ソフトマックスの対数を数値安定的に計算します。
// ///
// /// Note: log, exp関数がElementwiseOpに追加される必要があります
// pub fn log_softmax(x: GraphNode, axis: usize) -> GraphNode {
//     // TODO: log, exp, expand の実装が必要
//     unimplemented!("log_softmax: requires log and exp operations")
// }

// /// L2ノルム: ||x||_2 = sqrt(sum(x^2, axis))
// ///
// /// Note: sqrt関数がElementwiseOpに追加される必要があります
// pub fn l2_norm(x: GraphNode, axis: usize) -> GraphNode {
//     // TODO: sqrt の実装が必要
//     // sqrt(reduce_sum(square(x), axis))
//     unimplemented!("l2_norm: requires sqrt operation")
// }

// /// L2正規化: x / ||x||_2
// ///
// /// Note: sqrt関数とbroadcastの実装が必要です
// pub fn l2_normalize(x: GraphNode, axis: usize) -> GraphNode {
//     // TODO: sqrt, broadcast の実装が必要
//     // let norm = l2_norm(x.clone(), axis);
//     // x / norm.expand_to(x.shape())
//     unimplemented!("l2_normalize: requires sqrt and broadcast operations")
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::shape::Expr;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_square() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let x_squared = square(x);

        // 型とshapeが保持されていることを確認
        match x_squared.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(x_squared.view.ndim(), 2);
        assert_eq!(x_squared.view.shape()[0], Expr::from(10));
        assert_eq!(x_squared.view.shape()[1], Expr::from(20));
    }

    #[test]
    fn test_cube() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5])
            .build();

        let x_cubed = cube(x);

        match x_cubed.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(x_cubed.view.ndim(), 1);
    }

    #[test]
    fn test_powi() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5])
            .build();

        let x_pow4 = powi(x, 4);

        match x_pow4.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(x_pow4.view.ndim(), 1);
    }

    #[test]
    #[should_panic(expected = "powi: n must be positive")]
    fn test_powi_zero() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5])
            .build();

        let _ = powi(x, 0);
    }

    #[test]
    fn test_abs_square() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let abs_sq = abs_square(x);

        match abs_sq.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(abs_sq.view.ndim(), 1);
    }

    #[test]
    fn test_min() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let min_ab = min(a, b);

        match min_ab.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(min_ab.view.ndim(), 1);
    }

    #[test]
    fn test_clamp() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let min_val = graph
            .input("min")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let max_val = graph
            .input("max")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let clamped = clamp(x, min_val, max_val);

        match clamped.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(clamped.view.ndim(), 1);
    }

    #[test]
    fn test_mean() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20, 30])
            .build();

        // Note: 現在の実装では実際にはsumのみを計算
        let mean_x = mean(x, 1);

        match mean_x.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        // 軸1が縮約されている
        assert_eq!(mean_x.view.ndim(), 2);
    }

    #[test]
    #[should_panic(expected = "mean: axis")]
    fn test_mean_out_of_bounds() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let _ = mean(x, 3);
    }

    #[test]
    #[should_panic(expected = "matmul: not yet implemented")]
    fn test_matmul_not_implemented() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![20, 30])
            .build();

        let _ = matmul(a, b);
    }

    #[test]
    #[should_panic(expected = "matmul: first argument must be 2D")]
    fn test_matmul_wrong_dim_a() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let _ = matmul(a, b);
    }

    #[test]
    #[should_panic(expected = "batch_matmul: not yet implemented")]
    fn test_batch_matmul_not_implemented() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 10, 20])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 20, 30])
            .build();

        let _ = batch_matmul(a, b);
    }

    #[test]
    fn test_log2() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = log2(x);

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);

        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Log2,
                ..
            } => {}
            _ => panic!("Expected Log2 operation"),
        }
    }

    #[test]
    fn test_exp2() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 5])
            .build();

        let result = exp2(x);

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 2);

        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Exp2,
                ..
            } => {}
            _ => panic!("Expected Exp2 operation"),
        }
    }

    #[test]
    fn test_log() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = log(x);

        // log(x) = log2(x) * const なので、最終的にMul演算になる
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
    }

    #[test]
    fn test_exp() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = exp(x);

        // exp(x) = exp2(x * const) なので、exp2演算が含まれる
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
    }
}
