//! ニューラルネットワーク層
//!
//! 基本的な層の実装を提供します。

use std::collections::HashMap;

use harp::tensor::{Dim2, DimDyn, Tensor};

use crate::{Module, Parameter};

/// 全結合層（Linear Layer）
///
/// `y = x @ W + b`
///
/// # Example
///
/// ```ignore
/// let linear = Linear::new(784, 128);
/// let output = linear.forward(&input);
/// ```
pub struct Linear {
    /// 重み行列 [in_features, out_features]
    weight: Parameter,
    /// バイアス [out_features]
    bias: Parameter,
}

impl Linear {
    /// 新しいLinear層を作成
    ///
    /// 重みはXavier初期化、バイアスはゼロ初期化されます。
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier-like initialization: scale by sqrt(2 / fan_in)
        let scale = (2.0 / in_features as f32).sqrt();
        let weight =
            (Tensor::<f32, Dim2>::rand([in_features, out_features]) * 2.0 - 1.0) * scale * 0.5;
        let bias = Tensor::<f32, DimDyn>::zeros_dyn(&[out_features]);

        Self {
            weight: Parameter::new(weight.into_dyn()),
            bias: Parameter::new(bias),
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
    pub fn forward(&self, input: &Tensor<f32, Dim2>) -> Tensor<f32, DimDyn> {
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

impl Module for Linear {
    fn parameters(&mut self) -> HashMap<String, &mut Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        params.insert("bias".to_string(), &mut self.bias);
        params
    }

    fn load_parameters(&mut self, params: HashMap<String, Parameter>) {
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
        let linear = Linear::new(10, 5);
        assert_eq!(linear.in_features(), 10);
        assert_eq!(linear.out_features(), 5);
    }

    #[test]
    fn test_linear_named_parameters() {
        let mut linear = Linear::new(10, 5);
        let params = linear.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_linear_num_parameters() {
        let mut linear = Linear::new(10, 5);
        // weight: 10 * 5 = 50, bias: 5
        assert_eq!(linear.num_parameters(), 55);
    }
}
