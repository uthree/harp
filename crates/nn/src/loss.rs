//! 損失関数
//!
//! 学習時に使用する損失関数を提供します。

use harp::tensor::{Dim0, Dimension, FloatDType, Tensor};

/// 平均二乗誤差（Mean Squared Error）
///
/// `MSE = mean((pred - target)^2)`
///
/// 入力は任意の次元 `D` を受け取り、出力はスカラー `Dim0` を返します。
///
/// # Type Parameters
///
/// * `T` - テンソルのデータ型
/// * `D` - テンソルの次元
///
/// # Example
///
/// ```ignore
/// use harp::tensor::{Tensor, Dim2, Dim0};
///
/// let pred: Tensor<f32, Dim2> = Tensor::ones([3, 4]);
/// let target: Tensor<f32, Dim2> = Tensor::zeros([3, 4]);
/// let loss: Tensor<f32, Dim0> = mse_loss(&pred, &target);
/// ```
pub fn mse_loss<T: FloatDType, D: Dimension>(
    pred: &Tensor<T, D>,
    target: &Tensor<T, D>,
) -> Tensor<T, Dim0> {
    let diff = pred - target;
    let sq = &diff * &diff;
    sq.mean()
}

// TODO: L1損失はabs演算を実装後に追加

#[cfg(test)]
mod tests {
    use super::*;
    use harp::tensor::{Dim1, Dim2, Dim3};

    #[test]
    fn test_mse_loss_dim1() {
        let pred = Tensor::<f32, Dim1>::ones([4]);
        let target = Tensor::<f32, Dim1>::zeros([4]);
        let loss: Tensor<f32, Dim0> = mse_loss(&pred, &target);
        assert_eq!(loss.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_mse_loss_dim2() {
        let pred = Tensor::<f32, Dim2>::ones([3, 4]);
        let target = Tensor::<f32, Dim2>::zeros([3, 4]);
        let loss: Tensor<f32, Dim0> = mse_loss(&pred, &target);
        assert_eq!(loss.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_mse_loss_dim3() {
        let pred = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        let target = Tensor::<f32, Dim3>::zeros([2, 3, 4]);
        let loss: Tensor<f32, Dim0> = mse_loss(&pred, &target);
        assert_eq!(loss.shape(), &[] as &[usize]);
    }
}
