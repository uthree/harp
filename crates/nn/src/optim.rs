//! オプティマイザ
//!
//! パラメータ更新のための最適化アルゴリズムを提供します。

use harp::tensor::{DimDyn, Tensor};

use crate::Module;

/// オプティマイザの基底トレイト
pub trait Optimizer {
    /// パラメータを更新（勾配降下ステップを実行）
    fn step<M: Module>(&mut self, module: &mut M);
}

/// 確率的勾配降下法（SGD）
///
/// `param = param - lr * grad`
///
/// # Example
///
/// ```ignore
/// let mut model = Linear::new(784, 128);
/// let mut sgd = SGD::new(0.01);
///
/// // 学習ループ
/// model.zero_grad();  // Module側のメソッド
/// let output = model.forward(&input);
/// let loss = mse_loss(&output, &target);
/// loss.backward();
/// sgd.step(&mut model);
/// ```
pub struct SGD {
    /// 学習率
    lr: f32,
}

impl SGD {
    /// 新しいSGDオプティマイザを作成
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// 学習率を取得
    pub fn learning_rate(&self) -> f32 {
        self.lr
    }

    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for SGD {
    fn step<M: Module>(&mut self, module: &mut M) {
        for (_name, param) in module.parameters() {
            if let Some(grad) = param.grad() {
                // 勾配を実体化
                grad.realize().expect("Failed to realize gradient");
                let grad_data = grad.data().expect("Failed to get gradient data");

                // パラメータを実体化
                param.realize().expect("Failed to realize parameter");
                let param_data = param.data().expect("Failed to get parameter data");

                // SGD更新: param = param - lr * grad
                let new_data: Vec<f32> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(p, g)| p - self.lr * g)
                    .collect();

                // 新しいテンソルを作成してパラメータを更新
                let shape = param.shape().to_vec();
                let new_tensor = Tensor::<f32, DimDyn>::from_data(new_data, shape);
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
        let sgd = SGD::new(0.01);
        assert!((sgd.learning_rate() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_set_lr() {
        let mut sgd = SGD::new(0.01);
        sgd.set_learning_rate(0.001);
        assert!((sgd.learning_rate() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_optimizer_trait() {
        let mut linear = Linear::new(10, 5);
        let mut sgd = SGD::new(0.01);

        // トレイトメソッドが呼べることを確認
        linear.zero_grad(); // Module側
        sgd.step(&mut linear); // Optimizer側
    }
}
