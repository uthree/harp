//! 全結合層（Linear Layer）

use std::collections::HashMap;
use std::marker::PhantomData;

use harp::tensor::{Dim1, Dim2, DimDyn, FloatDType, Tensor};
use typed_builder::TypedBuilder;

use crate::{Module, Parameter, ParameterMut};

/// Linear層の設定
#[derive(TypedBuilder)]
#[builder(build_method(into = Linear<T>))]
pub struct LinearConfig<T: FloatDType = f32> {
    /// 入力特徴数
    in_features: usize,
    /// 出力特徴数
    out_features: usize,
    /// バイアスの有無（デフォルト: true）
    #[builder(default = true)]
    bias: bool,
    #[builder(default, setter(skip))]
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> From<LinearConfig<T>> for Linear<T> {
    fn from(config: LinearConfig<T>) -> Self {
        let weight = Tensor::<T, Dim2>::rand([config.in_features, config.out_features]);

        let bias = if config.bias {
            Some(Parameter::new(Tensor::<T, Dim1>::zeros([
                config.out_features
            ])))
        } else {
            None
        };

        Linear {
            weight: Parameter::new(weight),
            bias,
            _dtype: PhantomData,
        }
    }
}

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
/// let linear = Linear::<f32>::new(784, 128).build();
/// let output = linear.forward(&input);
///
/// // biasなし
/// let linear_no_bias = Linear::<f32>::new(784, 128).bias(false).build();
/// ```
pub struct Linear<T: FloatDType = f32> {
    /// 重み行列 [in_features, out_features]
    weight: Parameter<T, Dim2>,
    /// バイアス [out_features]（オプション）
    bias: Option<Parameter<T, Dim1>>,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Linear<T> {
    /// 新しいLinear層のビルダーを作成
    ///
    /// デフォルトではbiasが有効です。
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_features: usize,
        out_features: usize,
    ) -> LinearConfigBuilder<T, ((usize,), (usize,), ())> {
        LinearConfig::builder()
            .in_features(in_features)
            .out_features(out_features)
    }

    /// 入力特徴数
    pub fn in_features(&self) -> usize {
        self.weight.tensor().shape()[0]
    }

    /// 出力特徴数
    pub fn out_features(&self) -> usize {
        self.weight.tensor().shape()[1]
    }

    /// 順伝播
    ///
    /// `y = x @ W + b`（biasがある場合）
    /// `y = x @ W`（biasがない場合）
    pub fn forward(&self, input: &Tensor<T, Dim2>) -> Tensor<T, DimDyn> {
        // 静的次元でmatmulを実行
        let out = input.matmul2(self.weight.tensor()).into_dyn();

        // + b (broadcast) if bias exists
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .tensor()
                .clone()
                .into_dyn()
                .unsqueeze(0)
                .expand(out.shape());
            &out + &bias_expanded
        } else {
            out
        }
    }
}

impl<T: FloatDType> Module<T> for Linear<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        let mut params: HashMap<String, &mut dyn ParameterMut<T>> = HashMap::new();
        params.insert(
            "weight".to_string(),
            &mut self.weight as &mut dyn ParameterMut<T>,
        );
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias as &mut dyn ParameterMut<T>);
        }
        params
    }

    fn load_parameters(&mut self, params: HashMap<String, Tensor<T, DimDyn>>) {
        if let Some(w) = params.get("weight") {
            ParameterMut::set_dyn(&mut self.weight, w.clone());
        }
        if let Some(b) = params.get("bias")
            && let Some(ref mut bias) = self.bias {
                ParameterMut::set_dyn(bias, b.clone());
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Module;

    #[test]
    fn test_linear_creation() {
        let linear = Linear::<f32>::new(10, 5).build();
        assert_eq!(linear.in_features(), 10);
        assert_eq!(linear.out_features(), 5);
    }

    #[test]
    fn test_linear_named_parameters() {
        let mut linear = Linear::<f32>::new(10, 5).build();
        let params = linear.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_linear_num_parameters() {
        let mut linear = Linear::<f32>::new(10, 5).build();
        // weight: 10 * 5 = 50, bias: 5
        assert_eq!(linear.num_parameters(), 55);
    }

    #[test]
    fn test_linear_no_bias() {
        let mut linear = Linear::<f32>::new(10, 5).bias(false).build();
        let params = linear.parameters();
        assert!(params.contains_key("weight"));
        assert!(!params.contains_key("bias"));
        assert_eq!(params.len(), 1);
        // weight: 10 * 5 = 50, no bias
        assert_eq!(linear.num_parameters(), 50);
    }

    #[test]
    fn test_linear_f64() {
        // f64でも動作することを確認
        let linear = Linear::<f64>::new(8, 4).build();
        assert_eq!(linear.in_features(), 8);
        assert_eq!(linear.out_features(), 4);
    }
}
