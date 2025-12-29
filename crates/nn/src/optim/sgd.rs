//! 確率的勾配降下法（SGD）

use std::ops::{Mul, Sub};

use harp::tensor::{DimDyn, FloatDType, Tensor};

use super::Optimizer;
use crate::Module;

/// 確率的勾配降下法（SGD）
///
/// `param = param - lr * grad`
///
/// # Type Parameters
///
/// * `T` - パラメータのデータ型（デフォルト: f32）
///
/// # Example
///
/// ```ignore
/// let mut model = Linear::<f32>::new(784, 128);
/// let mut sgd = SGD::<f32>::new(0.01);
///
/// // 学習ループ
/// model.zero_grad();  // Module側のメソッド
/// let output = model.forward(&input);
/// let loss = mse_loss(&output, &target);
/// loss.backward();
/// sgd.step(&mut model);
/// ```
pub struct SGD<T: FloatDType = f32> {
    /// 学習率
    lr: T,
}

impl<T: FloatDType> SGD<T> {
    /// 新しいSGDオプティマイザを作成
    pub fn new(lr: T) -> Self {
        Self { lr }
    }

    /// 学習率を取得
    pub fn learning_rate(&self) -> T {
        self.lr.clone()
    }

    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: T) {
        self.lr = lr;
    }
}

// FloatDType でジェネリックな最適化実装
impl<T> Optimizer<T> for SGD<T>
where
    T: FloatDType + Copy + Sub<T, Output = T> + Mul<T, Output = T>,
{
    fn step<M: Module<T>>(&mut self, module: &mut M) {
        for (_name, param) in module.parameters() {
            if let Some(grad) = param.grad_generic() {
                // 勾配を実体化
                grad.realize().expect("Failed to realize gradient");
                let grad_data = grad.data().expect("Failed to get gradient data");

                // パラメータを実体化
                param.realize().expect("Failed to realize parameter");
                let param_data = param.data().expect("Failed to get parameter data");

                // SGD更新: param = param - lr * grad
                let lr = self.lr;
                let new_data: Vec<T> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&p, &g)| p - lr * g)
                    .collect();

                // 新しいテンソルを作成してパラメータを更新
                let shape = param.shape().to_vec();
                let new_tensor = Tensor::<T, DimDyn>::from_data(new_data, shape);
                param.set(new_tensor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;

    #[test]
    fn test_sgd_creation() {
        let sgd = SGD::<f32>::new(0.01);
        assert!((sgd.learning_rate() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_set_lr() {
        let mut sgd = SGD::<f32>::new(0.01);
        sgd.set_learning_rate(0.001);
        assert!((sgd.learning_rate() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_optimizer_trait() {
        let mut linear = Linear::<f32>::new(10, 5);
        let mut sgd = SGD::<f32>::new(0.01);

        // トレイトメソッドが呼べることを確認
        linear.zero_grad(); // Module側
        sgd.step(&mut linear); // Optimizer側
    }

    #[test]
    fn test_sgd_f64() {
        let sgd = SGD::<f64>::new(0.01);
        assert!((sgd.learning_rate() - 0.01).abs() < 1e-10);
    }
}
