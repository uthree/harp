//! RMSProp オプティマイザ
//!
//! Hinton, G. (2012) - Lecture 6a: Overview of mini-batch gradient descent

use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

use harp::tensor::{DimDyn, FloatDType, Tensor};

use super::Optimizer;
use crate::Module;

/// RMSProp オプティマイザ
///
/// 勾配の二乗の移動平均を使って学習率を適応的に調整します。
///
/// ```text
/// v_t = rho * v_{t-1} + (1 - rho) * g_t^2
/// θ_t = θ_{t-1} - lr * g_t / (sqrt(v_t) + ε)
/// ```
///
/// # Type Parameters
///
/// * `T` - パラメータのデータ型（デフォルト: f32）
///
/// # Example
///
/// ```ignore
/// let mut model = Linear::<f32>::new(784, 128);
/// let mut optimizer = RMSProp::<f32>::new(0.001);
///
/// // 学習ループ
/// model.zero_grad();
/// let output = model.forward(&input);
/// let loss = mse_loss(&output, &target);
/// loss.backward();
/// optimizer.step(&mut model);
/// ```
pub struct RMSProp<T: FloatDType = f32> {
    /// 学習率
    lr: T,
    /// 減衰率（通常 0.99）
    rho: T,
    /// 数値安定性のための小さな値
    epsilon: T,
    /// 各パラメータの勾配二乗の移動平均
    square_avg: HashMap<String, Vec<T>>,
}

impl<T: FloatDType + Div<T, Output = T>> RMSProp<T> {
    /// 新しい RMSProp オプティマイザを作成
    ///
    /// デフォルト: rho=0.99, epsilon=1e-8
    pub fn new(lr: T) -> Self {
        Self {
            lr,
            rho: T::from_usize(99) / T::from_usize(100), // 0.99
            epsilon: T::EPSILON,
            square_avg: HashMap::new(),
        }
    }

    /// rho を設定（ビルダーパターン）
    pub fn with_rho(mut self, rho: T) -> Self {
        self.rho = rho;
        self
    }

    /// epsilon を設定（ビルダーパターン）
    pub fn with_eps(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// 学習率を取得
    pub fn learning_rate(&self) -> T {
        self.lr.clone()
    }

    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: T) {
        self.lr = lr;
    }

    /// 減衰率を取得
    pub fn rho(&self) -> T {
        self.rho.clone()
    }

    /// 減衰率を設定
    pub fn set_rho(&mut self, rho: T) {
        self.rho = rho;
    }
}

impl<T> Optimizer<T> for RMSProp<T>
where
    T: FloatDType
        + Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>,
{
    fn step<M: Module<T>>(&mut self, module: &mut M) {
        let one = <T as FloatDType>::ONE;

        for (name, param) in module.parameters() {
            if let Some(grad) = param.grad_generic() {
                // 勾配を実体化
                grad.realize().expect("Failed to realize gradient");
                let grad_data = grad.data().expect("Failed to get gradient data");

                // パラメータを実体化
                param.realize().expect("Failed to realize parameter");
                let param_data = param.data().expect("Failed to get parameter data");

                let lr = self.lr;
                let rho = self.rho;
                let epsilon = self.epsilon;

                // 勾配二乗の移動平均を取得または初期化
                let sq_avg = self
                    .square_avg
                    .entry(name.clone())
                    .or_insert_with(|| vec![<T as FloatDType>::ZERO; param_data.len()]);

                // RMSProp更新:
                // v = rho * v + (1 - rho) * g^2
                // param = param - lr * g / (sqrt(v) + eps)
                let new_data: Vec<T> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .zip(sq_avg.iter_mut())
                    .map(|((&p, &g), v)| {
                        *v = rho * *v + (one - rho) * g * g;
                        p - lr * g / ((*v).sqrt() + epsilon)
                    })
                    .collect();

                // 新しいテンソルを作成してパラメータを更新
                let shape = param.shape().to_vec();
                let new_tensor = Tensor::<T, DimDyn>::from_data(new_data, shape);
                param.set_dyn(new_tensor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;

    #[test]
    fn test_rmsprop_creation() {
        let optimizer = RMSProp::<f32>::new(0.001);
        assert!((optimizer.learning_rate() - 0.001).abs() < 1e-6);
        assert!((optimizer.rho() - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_builder() {
        let optimizer = RMSProp::<f32>::new(0.01).with_rho(0.9).with_eps(1e-7);
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.rho() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_set_params() {
        let mut optimizer = RMSProp::<f32>::new(0.001);
        optimizer.set_learning_rate(0.01);
        optimizer.set_rho(0.95);
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.rho() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_optimizer_trait() {
        let mut linear = Linear::<f32>::new(10, 5);
        let mut optimizer = RMSProp::<f32>::new(0.001);

        // トレイトメソッドが呼べることを確認
        linear.zero_grad();
        optimizer.step(&mut linear);
    }

    #[test]
    fn test_rmsprop_f64() {
        let optimizer = RMSProp::<f64>::new(0.001);
        assert!((optimizer.learning_rate() - 0.001).abs() < 1e-10);
        assert!((optimizer.rho() - 0.99).abs() < 1e-10);
    }
}
