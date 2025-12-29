//! ニューラルネットワーク層
//!
//! 基本的な層の実装を提供します。

use std::collections::HashMap;
use std::marker::PhantomData;

use harp::tensor::{Dim2, DimDyn, FloatDType, Tensor};

use crate::{Module, Parameter};

/// 全結合層（Linear Layer）
///
/// `y = x @ W + b`
///
/// # Type Parameters
///
/// * `T` - テンソルのデータ型（デフォルト: f32）
///
/// # Example
///
/// ```ignore
/// let linear = Linear::<f32>::new(784, 128);
/// let output = linear.forward(&input);
/// ```
pub struct Linear<T: FloatDType = f32> {
    /// 重み行列 [in_features, out_features]
    weight: Parameter<T>,
    /// バイアス [out_features]
    bias: Parameter<T>,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Linear<T> {
    /// 新しいLinear層を作成
    ///
    /// 重みはランダム初期化、バイアスはゼロ初期化されます。
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Tensor::<T, Dim2>::rand([in_features, out_features]);
        let bias = Tensor::<T, DimDyn>::zeros_dyn(&[out_features]);

        Self {
            weight: Parameter::new(weight.into_dyn()),
            bias: Parameter::new(bias),
            _dtype: PhantomData,
        }
    }

    /// 入力特徴数
    pub fn in_features(&self) -> usize {
        self.weight.shape()[0]
    }

    /// 出力特徴数
    pub fn out_features(&self) -> usize {
        self.weight.shape()[1]
    }

    /// 順伝播
    ///
    /// `y = x @ W + b`
    pub fn forward(&self, input: &Tensor<T, Dim2>) -> Tensor<T, DimDyn> {
        // weightを[in, out]として保持しているので、x @ W で計算
        // weightをDim2として扱うためにreshape
        let weight_shape = self.weight.shape();
        let weight_dim2 = self
            .weight
            .clone()
            .reshape_dyn(&[weight_shape[0], weight_shape[1]])
            .into_dim2();

        let out = input.matmul2(&weight_dim2).into_dyn();

        // + b (broadcast)
        let bias_expanded = self.bias.clone().unsqueeze(0).expand(out.shape());

        &out + &bias_expanded
    }
}

impl<T: FloatDType> Module<T> for Linear<T> {
    fn parameters(&mut self) -> HashMap<String, &mut Parameter<T>> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        params.insert("bias".to_string(), &mut self.bias);
        params
    }

    fn load_parameters(&mut self, params: HashMap<String, Parameter<T>>) {
        if let Some(w) = params.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = params.get("bias") {
            self.bias = b.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let linear = Linear::<f32>::new(10, 5);
        assert_eq!(linear.in_features(), 10);
        assert_eq!(linear.out_features(), 5);
    }

    #[test]
    fn test_linear_named_parameters() {
        let mut linear = Linear::<f32>::new(10, 5);
        let params = linear.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_linear_num_parameters() {
        let mut linear = Linear::<f32>::new(10, 5);
        // weight: 10 * 5 = 50, bias: 5
        assert_eq!(linear.num_parameters(), 55);
    }

    #[test]
    fn test_linear_f64() {
        // f64でも動作することを確認
        let linear = Linear::<f64>::new(8, 4);
        assert_eq!(linear.in_features(), 8);
        assert_eq!(linear.out_features(), 4);
    }
}
