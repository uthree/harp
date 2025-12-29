//! 損失関数
//!
//! 学習時に使用する損失関数を提供します。

use harp::tensor::{DimDyn, FloatDType, Tensor};

/// 平均二乗誤差（Mean Squared Error）
///
/// `MSE = mean((pred - target)^2)`
///
/// # Type Parameters
///
/// * `T` - テンソルのデータ型
///
/// # Example
///
/// ```ignore
/// let loss = mse_loss(&predictions, &targets);
/// loss.backward();
/// ```
pub fn mse_loss<T: FloatDType>(
    pred: &Tensor<T, DimDyn>,
    target: &Tensor<T, DimDyn>,
) -> Tensor<T, DimDyn> {
    let diff = pred - target;
    let sq = &diff * &diff;

    // 全要素の平均を計算
    let n_elements = sq.shape().iter().product::<usize>();

    // 全軸でsum
    let mut result = sq;
    while result.ndim() > 0 {
        result = result.sum(0);
    }

    // 要素数で割る
    let n_tensor = Tensor::<T, DimDyn>::full_dyn(&[], T::from_usize(n_elements));
    &result / &n_tensor
}

// TODO: L1損失はabs演算を実装後に追加

#[cfg(test)]
mod tests {
    // テストは統合テストで行う
}
