//! 損失関数
//!
//! 学習時に使用する損失関数を提供します。

use harp::tensor::{DimDyn, Tensor};

/// 平均二乗誤差（Mean Squared Error）
///
/// `MSE = mean((pred - target)^2)`
///
/// # Example
///
/// ```ignore
/// let loss = mse_loss(&predictions, &targets);
/// loss.backward();
/// ```
pub fn mse_loss(pred: &Tensor<f32, DimDyn>, target: &Tensor<f32, DimDyn>) -> Tensor<f32, DimDyn> {
    let diff = pred - target;
    let sq = &diff * &diff;

    // 全要素の平均を計算
    let n_elements = sq.shape().iter().product::<usize>() as f32;

    // 全軸でsum
    let mut result = sq;
    while result.ndim() > 0 {
        result = result.sum(0);
    }

    // 要素数で割る（勾配追跡のためテンソル演算を使用）
    let inv_n = Tensor::<f32, DimDyn>::full_dyn(&[], 1.0 / n_elements);
    &result * &inv_n
}

// TODO: L1損失はabs演算を実装後に追加

#[cfg(test)]
mod tests {
    // テストは統合テストで行う
}
