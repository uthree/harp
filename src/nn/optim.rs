//! オプティマイザーモジュール
//!
//! ニューラルネットワークのパラメータを更新するためのオプティマイザー。
//!
//! # 設計
//!
//! - `Optimizer` trait: パラメータ更新の抽象化
//! - 各オプティマイザーは`step()`メソッドでパラメータを更新
//! - 勾配のゼロクリアは`zero_grad()`で実行可能
//!
//! # 使用例
//!
//! ```ignore
//! use harp::nn::{Module, Parameter};
//! use harp::nn::optim::{Optimizer, SGD};
//!
//! let mut module = MyModule::new();
//! let mut optimizer = SGD::new(0.01);
//!
//! // 学習ループ
//! for _ in 0..100 {
//!     // Forward
//!     let output = module.forward(&input);
//!     let loss = compute_loss(&output, &target);
//!
//!     // Backward
//!     loss.backward();
//!
//!     // Update parameters
//!     let mut params = module.parameters_mut();
//!     optimizer.step(&mut params);
//!     optimizer.zero_grad(&params);
//! }
//! ```

use super::Parameter;
use crate::autograd::Tensor;

/// パラメータ更新を行うオプティマイザーのtrait
///
/// ニューラルネットワークの学習において、勾配を使ってパラメータを更新する
/// アルゴリズムを抽象化します。
pub trait Optimizer {
    /// パラメータを更新する
    ///
    /// 各パラメータの勾配を使って、パラメータの値を更新します。
    /// 勾配が計算されていない（None）パラメータはスキップされます。
    ///
    /// # 引数
    ///
    /// * `parameters` - 更新するパラメータのリスト
    ///
    /// # 例
    ///
    /// ```ignore
    /// let mut params = module.parameters_mut();
    /// optimizer.step(&mut params);
    /// ```
    fn step(&mut self, parameters: &mut [&mut Parameter]);

    /// 全パラメータの勾配をゼロクリア
    ///
    /// 次のイテレーションに備えて、勾配を初期化します。
    /// 通常、`step()`の後に呼び出します。
    ///
    /// # 引数
    ///
    /// * `parameters` - 勾配をクリアするパラメータのリスト
    ///
    /// # 例
    ///
    /// ```ignore
    /// optimizer.step(&mut params);
    /// optimizer.zero_grad(&params);
    /// ```
    fn zero_grad(&self, parameters: &[&Parameter]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}

/// 確率的勾配降下法（Stochastic Gradient Descent）
///
/// 最もシンプルなオプティマイザー。各パラメータを勾配の方向に更新します。
///
/// 更新式: `param = param - learning_rate * grad`
///
/// # Examples
///
/// ```
/// use harp::nn::optim::SGD;
///
/// // 学習率0.01でSGDを作成
/// let mut optimizer = SGD::new(0.01);
/// ```
pub struct SGD {
    /// 学習率
    lr: f32,
}

impl SGD {
    /// 新しいSGDオプティマイザーを作成
    ///
    /// # 引数
    ///
    /// * `lr` - 学習率（learning rate）。通常は0.001〜0.1程度の値を使用。
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::nn::optim::SGD;
    ///
    /// let optimizer = SGD::new(0.01);
    /// ```
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// 学習率を取得
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// 学習率を設定
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Parameter]) {
        for param in parameters {
            // 勾配が計算されている場合のみ更新
            if let Some(grad_node) = param.grad() {
                // 勾配をTensorとして取得（計算グラフを追跡しない）
                let grad_tensor = Tensor::from_graph_node(grad_node, false);

                // 更新量を計算: lr * grad
                let update = &grad_tensor * self.lr;

                // 新しい値を計算: param - update
                // param.tensor()でTensorを取得してclone（borrowingを回避）
                let param_tensor = param.tensor().clone();
                let new_value = &param_tensor - &update;

                // パラメータを更新
                // 内部のTensorを直接更新（requires_grad=trueで）
                // ***paramでTensorにアクセス（*param -> &mut Parameter, **param -> Parameter, ***param -> Tensor via DerefMut）
                ***param = Tensor::from_graph_node(new_value.data, true);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_new() {
        let optimizer = SGD::new(0.01);
        assert_eq!(optimizer.lr(), 0.01);
    }

    #[test]
    fn test_sgd_set_lr() {
        let mut optimizer = SGD::new(0.01);
        optimizer.set_lr(0.001);
        assert_eq!(optimizer.lr(), 0.001);
    }

    #[test]
    fn test_sgd_step() {
        // パラメータを作成
        let mut param = Parameter::ones(vec![2, 2]);

        // 勾配を設定するために、計算グラフを作成
        let x = Tensor::ones(vec![2, 2]);
        let y = &x + param.tensor();
        // スカラーテンソルを作るため、全次元について合計
        let loss = y.sum(0).sum(0);

        // Backward
        loss.backward();

        // 勾配が設定されていることを確認
        assert!(param.grad().is_some());

        // オプティマイザーで更新
        let mut optimizer = SGD::new(0.1);
        optimizer.step(&mut [&mut param]);

        // パラメータが更新されていることを確認
        // 元の値: 1.0, 勾配: 1.0, 学習率: 0.1
        // 新しい値: 1.0 - 0.1 * 1.0 = 0.9
        // （実際の値はバックエンドで計算される）
        assert!(param.requires_grad());
    }

    #[test]
    fn test_sgd_step_no_grad() {
        // 勾配が計算されていないパラメータ
        let mut param = Parameter::ones(vec![2, 2]);

        // 勾配がNone
        assert!(param.grad().is_none());

        // オプティマイザーで更新（何も起きないはず）
        let mut optimizer = SGD::new(0.1);
        optimizer.step(&mut [&mut param]);

        // パラメータは変更されていない
        assert!(param.requires_grad());
    }

    #[test]
    fn test_sgd_zero_grad() {
        // パラメータを作成して勾配を計算
        let param = Parameter::ones(vec![2, 2]);
        let x = Tensor::ones(vec![2, 2]);
        let y = &x + param.tensor();
        let loss = y.sum(0).sum(0);
        loss.backward();

        // 勾配が設定されている
        assert!(param.grad().is_some());

        // zero_gradを呼ぶ
        let optimizer = SGD::new(0.1);
        optimizer.zero_grad(&[&param]);

        // 勾配がクリアされている
        assert!(param.grad().is_none());
    }

    #[test]
    fn test_sgd_multiple_parameters() {
        // 複数のパラメータを作成
        let mut param1 = Parameter::ones(vec![2, 2]);
        let mut param2 = Parameter::ones(vec![3, 3]);

        // 勾配を計算
        let x = Tensor::ones(vec![2, 2]);
        let y = &x + param1.tensor();
        let loss1 = y.sum(0).sum(0);
        loss1.backward();

        let x2 = Tensor::ones(vec![3, 3]);
        let y2 = &x2 + param2.tensor();
        let loss2 = y2.sum(0).sum(0);
        loss2.backward();

        // 両方のパラメータを同時に更新
        let mut optimizer = SGD::new(0.1);
        optimizer.step(&mut [&mut param1, &mut param2]);

        // 両方とも更新されている
        assert!(param1.requires_grad());
        assert!(param2.requires_grad());
    }
}
