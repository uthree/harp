//! 線形代数の高級演算（Matmul フォールバック）
//!
//! primops の組み合わせで実装される線形代数演算。
//! primops::Matmul は効率的な特殊化実装（BLAS など）用で、
//! こちらは汎用的なフォールバック実装を提供する。
//!
//! ## Matmul フォールバック設計
//!
//! 行列積 C = A @ B（A: [M, K], B: [K, N] → C: [M, N]）は
//! 以下の primops の組み合わせで表現できる：
//!
//! ```text
//! 1. Unsqueeze: A [M, K] → [M, K, 1]
//! 2. Unsqueeze: B [K, N] → [1, K, N]
//! 3. Expand:    A [M, K, 1] → [M, K, N]
//! 4. Expand:    B [1, K, N] → [M, K, N]
//! 5. Mul:       A * B → [M, K, N]  (要素ごとの積)
//! 6. Sum:       axis=1 で縮約 → [M, 1, N]
//! 7. Squeeze:   [M, 1, N] → [M, N]
//! ```

use std::ops;

use crate::primops::{Expand, Shape, Squeeze, Sum, Unsqueeze};

// ============================================================================
// matmul_fallback 関数（primops の組み合わせで実装）
// ============================================================================

/// 行列積のフォールバック実装
///
/// Unsqueeze, Expand, Mul, Sum, Squeeze の組み合わせで matmul を実現する。
/// primops::Matmul を実装していない型に対して使用可能。
///
/// # 型パラメータ
///
/// - `T2`: 2次元テンソル型（入力・出力）
/// - `T3`: 3次元テンソル型（中間計算用）
///
/// # 制約
///
/// T2 と T3 の関係:
/// - `T2::Unsqueeze::Output = T3`（2D → 3D）
/// - `T3::Squeeze::Output = T2`（3D → 2D）
pub fn matmul_fallback<T2, T3>(a: &T2, b: &T2) -> T2
where
    T2: Shape + Unsqueeze<Output = T3>,
    T3: Clone
        + Shape
        + Expand<Output = T3>
        + Sum<Output = T3>
        + Squeeze<Output = T2>
        + ops::Mul<T3, Output = T3>,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(a_shape.len(), 2, "matmul requires 2D tensors");
    assert_eq!(b_shape.len(), 2, "matmul requires 2D tensors");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    assert_eq!(
        k, b_shape[0],
        "matmul dimension mismatch: [{}, {}] @ [{}, {}]",
        m, k, b_shape[0], n
    );

    // 1. Unsqueeze: A [M, K] → [M, K, 1]
    let a_3d: T3 = a.unsqueeze(2);

    // 2. Unsqueeze: B [K, N] → [1, K, N]
    let b_3d: T3 = b.unsqueeze(0);

    // 3. Expand: A [M, K, 1] → [M, K, N]
    let a_expanded: T3 = a_3d.expand(2, n);

    // 4. Expand: B [1, K, N] → [M, K, N]
    let b_expanded: T3 = b_3d.expand(0, m);

    // 5. Mul: element-wise → [M, K, N]
    let product: T3 = a_expanded * b_expanded;

    // 6. Sum: axis=1 → [M, 1, N]
    let summed: T3 = product.sum(1);

    // 7. Squeeze: [M, 1, N] → [M, N]
    summed.squeeze(1)
}
