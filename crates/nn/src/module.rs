//! Module trait と Parameter 型
//!
//! ニューラルネットワーク層の基底トレイトと学習可能パラメータを提供します。

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use harp::tensor::{DimDyn, Tensor};

/// 学習可能なパラメータを表すラッパー
///
/// Tensorをラップし、自動的に `requires_grad = true` を設定します。
///
/// # Note
///
/// 現在はf32のみをサポート。将来的にはジェネリック化予定。
#[derive(Clone)]
pub struct Parameter(pub Tensor<f32, DimDyn>);

impl Parameter {
    /// 新しいParameterを作成
    ///
    /// テンソルは自動的に勾配追跡が有効になります。
    pub fn new(tensor: Tensor<f32, DimDyn>) -> Self {
        Self(tensor.set_requires_grad(true))
    }

    /// 内部テンソルへの参照を取得
    pub fn tensor(&self) -> &Tensor<f32, DimDyn> {
        &self.0
    }

    /// 内部テンソルへの可変参照を取得
    pub fn tensor_mut(&mut self) -> &mut Tensor<f32, DimDyn> {
        &mut self.0
    }

    /// テンソルを置き換え
    ///
    /// 新しいテンソルは自動的に勾配追跡が有効になります。
    pub fn set(&mut self, tensor: Tensor<f32, DimDyn>) {
        self.0 = tensor.set_requires_grad(true);
    }

    /// 勾配をゼロに初期化
    pub fn zero_grad(&self) {
        self.0.zero_grad();
    }
}

impl Deref for Parameter {
    type Target = Tensor<f32, DimDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// ニューラルネットワークモジュールの基底トレイト
///
/// 学習可能なパラメータを持つ計算ユニットを表現します。
/// `forward()` メソッドは各実装で独自に定義してください。
///
/// # Note
///
/// 現在はf32のみをサポート。将来的にはジェネリック化予定。
///
/// # Example
///
/// ```ignore
/// impl Module for Linear {
///     fn parameters(&mut self) -> HashMap<String, &mut Parameter> {
///         let mut params = HashMap::new();
///         params.insert("weight".to_string(), &mut self.weight);
///         params.insert("bias".to_string(), &mut self.bias);
///         params
///     }
///
///     fn load_parameters(&mut self, params: HashMap<String, Parameter>) {
///         if let Some(w) = params.get("weight") {
///             self.weight = w.clone();
///         }
///         if let Some(b) = params.get("bias") {
///             self.bias = b.clone();
///         }
///     }
/// }
/// ```
pub trait Module {
    /// 名前付きパラメータへの可変参照を取得
    ///
    /// ネストしたモジュールの場合、ドット区切りで名前空間化します。
    /// 例: `layer1.weight`, `layer1.bias`
    fn parameters(&mut self) -> HashMap<String, &mut Parameter>;

    /// 名前付きパラメータをロード（チェックポイント復元用）
    fn load_parameters(&mut self, params: HashMap<String, Parameter>);

    /// パラメータ総数を取得
    fn num_parameters(&mut self) -> usize {
        self.parameters()
            .values()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    /// 全パラメータの勾配をゼロに初期化
    fn zero_grad(&mut self) {
        for (_name, param) in self.parameters() {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_new() {
        let t = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4]);
        let p = Parameter::new(t);
        assert_eq!(p.shape(), &[3, 4]);
    }

    #[test]
    fn test_parameter_deref() {
        let t = Tensor::<f32, DimDyn>::zeros_dyn(&[2, 2]);
        let p = Parameter::new(t);
        // Deref経由でTensorメソッドにアクセス
        assert_eq!(p.ndim(), 2);
    }

    #[test]
    fn test_parameter_set() {
        let t1 = Tensor::<f32, DimDyn>::zeros_dyn(&[2, 2]);
        let mut p = Parameter::new(t1);

        let t2 = Tensor::<f32, DimDyn>::ones_dyn(&[3, 3]);
        p.set(t2);

        assert_eq!(p.shape(), &[3, 3]);
    }
}
