//! モメンタム付き確率的勾配降下法（Momentum SGD）

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use harp::tensor::{DimDyn, FloatDType, NumericDType, Tensor};
use typed_builder::TypedBuilder;

use super::Optimizer;
use crate::Module;

/// Momentumオプティマイザの設定
///
/// # Example
///
/// ```ignore
/// let mut optimizer = Momentum::<f32>::builder()
///     .lr(0.01)
///     .momentum(0.95)
///     .build();
/// ```
#[derive(TypedBuilder)]
#[builder(build_method(into = Momentum<T>))]
pub struct MomentumConfig<T: FloatDType = f32>
where
    T: Div<T, Output = T>,
{
    /// 学習率
    lr: T,
    /// モメンタム係数（デフォルト: 0.9）
    #[builder(default_code = "T::from_usize(9) / T::from_usize(10)")]
    momentum: T,
    #[builder(default, setter(skip))]
    _marker: PhantomData<T>,
}

impl<T: FloatDType + Div<T, Output = T>> From<MomentumConfig<T>> for Momentum<T> {
    fn from(config: MomentumConfig<T>) -> Self {
        Momentum {
            lr: config.lr,
            momentum: config.momentum,
            velocities: HashMap::new(),
        }
    }
}

/// モメンタム付き確率的勾配降下法（Momentum SGD）
///
/// 速度項を導入して収束を加速します。
///
/// ```text
/// v_t = momentum * v_{t-1} + grad
/// param = param - lr * v_t
/// ```
///
/// # Type Parameters
///
/// * `T` - パラメータのデータ型（デフォルト: f32）
///
/// # Example
///
/// ```ignore
/// let mut model = Linear::<f32>::new(784, 128).build();
/// let mut optimizer = Momentum::<f32>::builder()
///     .lr(0.01)
///     .momentum(0.9)
///     .build();
///
/// // 学習ループ
/// model.zero_grad();
/// let output = model.forward(&input);
/// let loss = mse_loss(&output, &target);
/// loss.backward();
/// optimizer.step(&mut model);
/// ```
pub struct Momentum<T: FloatDType = f32> {
    /// 学習率
    lr: T,
    /// モメンタム係数（通常 0.9）
    momentum: T,
    /// 各パラメータの速度
    velocities: HashMap<String, Vec<T>>,
}

impl<T: FloatDType + Div<T, Output = T>> Momentum<T> {
    /// ビルダーを作成
    pub fn builder() -> MomentumConfigBuilder<T, ((), ())> {
        MomentumConfig::builder()
    }

    /// 学習率を取得
    pub fn learning_rate(&self) -> T {
        self.lr.clone()
    }

    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: T) {
        self.lr = lr;
    }

    /// モメンタム係数を取得
    pub fn momentum(&self) -> T {
        self.momentum.clone()
    }

    /// モメンタム係数を設定
    pub fn set_momentum(&mut self, momentum: T) {
        self.momentum = momentum;
    }
}

// FloatDType でジェネリックな最適化実装
impl<T> Optimizer<T> for Momentum<T>
where
    T: FloatDType + Copy + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T>,
{
    fn step<M: Module<T>>(&mut self, module: &mut M) {
        for (name, param) in module.parameters() {
            if let Some(grad) = param.grad_generic() {
                // 勾配を実体化
                grad.realize().expect("Failed to realize gradient");
                let grad_data = grad.data().expect("Failed to get gradient data");

                // パラメータを実体化
                param.realize().expect("Failed to realize parameter");
                let param_data = param.data().expect("Failed to get parameter data");

                let lr = self.lr;
                let momentum = self.momentum;

                // 速度を取得または初期化
                let velocity = self
                    .velocities
                    .entry(name.clone())
                    .or_insert_with(|| vec![<T as NumericDType>::ZERO; param_data.len()]);

                // Momentum更新: v = momentum * v + grad
                //              param = param - lr * v
                let new_data: Vec<T> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .zip(velocity.iter_mut())
                    .map(|((&p, &g), v)| {
                        *v = momentum * *v + g;
                        p - lr * *v
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
    fn test_momentum_creation() {
        let optimizer = Momentum::<f32>::builder().lr(0.01).build();
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.momentum() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_builder() {
        let optimizer = Momentum::<f32>::builder().lr(0.01).momentum(0.95).build();
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.momentum() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_set_params() {
        let mut optimizer = Momentum::<f32>::builder().lr(0.01).build();
        optimizer.set_learning_rate(0.001);
        optimizer.set_momentum(0.95);
        assert!((optimizer.learning_rate() - 0.001).abs() < 1e-6);
        assert!((optimizer.momentum() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_optimizer_trait() {
        let mut linear = Linear::<f32>::new(10, 5).build();
        let mut optimizer = Momentum::<f32>::builder().lr(0.01).build();

        // トレイトメソッドが呼べることを確認
        linear.zero_grad();
        optimizer.step(&mut linear);
    }

    #[test]
    fn test_momentum_f64() {
        let optimizer = Momentum::<f64>::builder().lr(0.01).build();
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-10);
        assert!((optimizer.momentum() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_momentum_velocity_initialization() {
        let mut linear = Linear::<f32>::new(2, 1).build();
        let mut optimizer = Momentum::<f32>::builder().lr(0.1).build();

        // 勾配がない状態でstepを呼んでも速度は保存されない
        linear.zero_grad();
        optimizer.step(&mut linear);
        assert!(optimizer.velocities.is_empty());
    }
}
