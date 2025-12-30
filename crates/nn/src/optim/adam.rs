//! Adam オプティマイザ
//!
//! Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
//! arXiv:1412.6980

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use harp::tensor::{DimDyn, FloatDType, Tensor};
use typed_builder::TypedBuilder;

use super::Optimizer;
use crate::Module;

/// Adam オプティマイザの設定
///
/// # Example
///
/// ```ignore
/// let mut optimizer = Adam::<f32>::builder()
///     .lr(0.001)
///     .beta1(0.9)
///     .beta2(0.999)
///     .build();
/// ```
#[derive(TypedBuilder)]
#[builder(build_method(into = Adam<T>))]
pub struct AdamConfig<T: FloatDType = f32>
where
    T: Div<T, Output = T>,
{
    /// 学習率
    lr: T,
    /// 一次モーメントの減衰率（デフォルト: 0.9）
    #[builder(default_code = "T::from_usize(9) / T::from_usize(10)")]
    beta1: T,
    /// 二次モーメントの減衰率（デフォルト: 0.999）
    #[builder(default_code = "T::from_usize(999) / T::from_usize(1000)")]
    beta2: T,
    /// 数値安定性のための小さな値（デフォルト: T::EPSILON）
    #[builder(default_code = "T::EPSILON")]
    epsilon: T,
    #[builder(default, setter(skip))]
    _marker: PhantomData<T>,
}

impl<T: FloatDType + Div<T, Output = T>> From<AdamConfig<T>> for Adam<T> {
    fn from(config: AdamConfig<T>) -> Self {
        Adam {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

/// Adam オプティマイザ
///
/// 勾配の一次モーメント（平均）と二次モーメント（分散）の移動平均を使用して
/// パラメータごとに学習率を適応的に調整します。
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
/// m̂_t = m_t / (1 - β1^t)
/// v̂_t = v_t / (1 - β2^t)
/// θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
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
/// let mut optimizer = Adam::<f32>::builder().lr(0.001).build();
///
/// // 学習ループ
/// model.zero_grad();
/// let output = model.forward(&input);
/// let loss = mse_loss(&output, &target);
/// loss.backward();
/// optimizer.step(&mut model);
/// ```
pub struct Adam<T: FloatDType = f32> {
    /// 学習率
    lr: T,
    /// 一次モーメントの減衰率（通常 0.9）
    beta1: T,
    /// 二次モーメントの減衰率（通常 0.999）
    beta2: T,
    /// 数値安定性のための小さな値
    epsilon: T,
    /// 各パラメータの一次モーメント（勾配の移動平均）
    m: HashMap<String, Vec<T>>,
    /// 各パラメータの二次モーメント（勾配二乗の移動平均）
    v: HashMap<String, Vec<T>>,
    /// 現在のステップ数（bias correction用）
    t: usize,
}

impl<T: FloatDType + Div<T, Output = T>> Adam<T> {
    /// ビルダーを作成
    pub fn builder() -> AdamConfigBuilder<T, ((), (), (), ())> {
        AdamConfig::builder()
    }

    /// 学習率を取得
    pub fn learning_rate(&self) -> T {
        self.lr.clone()
    }

    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: T) {
        self.lr = lr;
    }

    /// β1を取得
    pub fn beta1(&self) -> T {
        self.beta1.clone()
    }

    /// β1を設定
    pub fn set_beta1(&mut self, beta1: T) {
        self.beta1 = beta1;
    }

    /// β2を取得
    pub fn beta2(&self) -> T {
        self.beta2.clone()
    }

    /// β2を設定
    pub fn set_beta2(&mut self, beta2: T) {
        self.beta2 = beta2;
    }

    /// 現在のステップ数を取得
    pub fn step_count(&self) -> usize {
        self.t
    }
}

impl<T> Optimizer<T> for Adam<T>
where
    T: FloatDType
        + Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>,
{
    fn step<M: Module<T>>(&mut self, module: &mut M) {
        self.t += 1;
        let one = <T as FloatDType>::ONE;

        // β1^t と β2^t を計算（bias correction用）
        let mut beta1_t = <T as FloatDType>::ONE;
        let mut beta2_t = <T as FloatDType>::ONE;
        for _ in 0..self.t {
            beta1_t = beta1_t * self.beta1;
            beta2_t = beta2_t * self.beta2;
        }

        for (name, param) in module.parameters() {
            if let Some(grad) = param.grad_generic() {
                // 勾配を実体化
                grad.realize().expect("Failed to realize gradient");
                let grad_data = grad.data().expect("Failed to get gradient data");

                // パラメータを実体化
                param.realize().expect("Failed to realize parameter");
                let param_data = param.data().expect("Failed to get parameter data");

                let lr = self.lr;
                let beta1 = self.beta1;
                let beta2 = self.beta2;
                let epsilon = self.epsilon;

                // モーメントを取得または初期化
                let m = self
                    .m
                    .entry(name.clone())
                    .or_insert_with(|| vec![<T as FloatDType>::ZERO; param_data.len()]);
                let v = self
                    .v
                    .entry(name.clone())
                    .or_insert_with(|| vec![<T as FloatDType>::ZERO; param_data.len()]);

                // Adam更新:
                // m = β1 * m + (1 - β1) * g
                // v = β2 * v + (1 - β2) * g^2
                // m_hat = m / (1 - β1^t)
                // v_hat = v / (1 - β2^t)
                // param = param - lr * m_hat / (sqrt(v_hat) + eps)
                let new_data: Vec<T> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .zip(m.iter_mut().zip(v.iter_mut()))
                    .map(|((&p, &g), (m_i, v_i))| {
                        // 一次モーメント（平均）の更新
                        *m_i = beta1 * *m_i + (one - beta1) * g;
                        // 二次モーメント（分散）の更新
                        *v_i = beta2 * *v_i + (one - beta2) * g * g;

                        // Bias correction
                        let m_hat = *m_i / (one - beta1_t);
                        let v_hat = *v_i / (one - beta2_t);

                        // パラメータ更新
                        p - lr * m_hat / (v_hat.sqrt() + epsilon)
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
    fn test_adam_creation() {
        let optimizer = Adam::<f32>::builder().lr(0.001).build();
        assert!((optimizer.learning_rate() - 0.001).abs() < 1e-6);
        assert!((optimizer.beta1() - 0.9).abs() < 1e-6);
        assert!((optimizer.beta2() - 0.999).abs() < 1e-6);
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_adam_builder() {
        let optimizer = Adam::<f32>::builder()
            .lr(0.01)
            .beta1(0.85)
            .beta2(0.99)
            .epsilon(1e-7)
            .build();
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.beta1() - 0.85).abs() < 1e-6);
        assert!((optimizer.beta2() - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_adam_set_params() {
        let mut optimizer = Adam::<f32>::builder().lr(0.001).build();
        optimizer.set_learning_rate(0.01);
        optimizer.set_beta1(0.85);
        optimizer.set_beta2(0.99);
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.beta1() - 0.85).abs() < 1e-6);
        assert!((optimizer.beta2() - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_adam_optimizer_trait() {
        let mut linear = Linear::<f32>::new(10, 5).build();
        let mut optimizer = Adam::<f32>::builder().lr(0.001).build();

        // トレイトメソッドが呼べることを確認
        linear.zero_grad();
        optimizer.step(&mut linear);

        // ステップカウントが更新されることを確認
        assert_eq!(optimizer.step_count(), 1);
    }

    #[test]
    fn test_adam_f64() {
        let optimizer = Adam::<f64>::builder().lr(0.001).build();
        assert!((optimizer.learning_rate() - 0.001).abs() < 1e-10);
        assert!((optimizer.beta1() - 0.9).abs() < 1e-10);
        assert!((optimizer.beta2() - 0.999).abs() < 1e-10);
    }

    #[test]
    fn test_adam_step_count_increments() {
        let mut linear = Linear::<f32>::new(2, 1).build();
        let mut optimizer = Adam::<f32>::builder().lr(0.001).build();

        for i in 1..=5 {
            linear.zero_grad();
            optimizer.step(&mut linear);
            assert_eq!(optimizer.step_count(), i);
        }
    }
}
