//! Module trait と Parameter 型
//!
//! ニューラルネットワーク層の基底トレイトと学習可能パラメータを提供します。

use std::collections::HashMap;
use std::marker::PhantomData;

use harp::tensor::{DimDyn, Dimension, FloatDType, Tensor};

// ============================================================================
// ParameterMut trait (型消去用)
// ============================================================================

/// パラメータへの型消去されたアクセスを提供するトレイト
///
/// 異なる次元を持つ `Parameter<T, D>` を同一のコレクションで扱うために使用
pub trait ParameterMut<T: FloatDType>: Send + Sync {
    /// 動的次元テンソルへの参照を取得
    fn tensor_dyn(&self) -> &Tensor<T, DimDyn>;

    /// 動的次元テンソルへの可変参照を取得
    fn tensor_dyn_mut(&mut self) -> &mut Tensor<T, DimDyn>;

    /// 勾配を取得
    fn grad_generic(&self) -> Option<Tensor<T, DimDyn>>;

    /// テンソルを実体化
    fn realize(&self) -> Result<(), String>;

    /// テンソルデータを取得
    fn data(&self) -> Option<Vec<T>>;

    /// テンソルを置き換え
    fn set_dyn(&mut self, tensor: Tensor<T, DimDyn>);

    /// 勾配をゼロに初期化
    fn zero_grad(&self);

    /// テンソルの形状を取得
    fn shape(&self) -> &[usize];
}

// ============================================================================
// Parameter<T, D>
// ============================================================================

/// 学習可能なパラメータを表すラッパー
///
/// 静的次元 `D` でテンソルを保持し、必要に応じて動的次元でアクセス可能
///
/// # Type Parameters
///
/// * `T` - テンソルのデータ型（デフォルト: f32）
/// * `D` - テンソルの次元型（デフォルト: DimDyn）
#[derive(Clone)]
pub struct Parameter<T: FloatDType = f32, D: Dimension = DimDyn> {
    /// 静的次元テンソル
    tensor: Tensor<T, D>,
    /// 動的次元テンソル（同じArcを共有）
    tensor_dyn: Tensor<T, DimDyn>,
}

impl<T: FloatDType, D: Dimension> Parameter<T, D> {
    /// 新しいParameterを作成
    ///
    /// テンソルは自動的に勾配追跡が有効になります。
    pub fn new(tensor: Tensor<T, D>) -> Self {
        let tensor = tensor.set_requires_grad(true);
        let tensor_dyn = tensor.clone().into_dyn();
        Self { tensor, tensor_dyn }
    }

    /// 静的次元テンソルへの参照を取得
    pub fn tensor(&self) -> &Tensor<T, D> {
        &self.tensor
    }

    /// 静的次元テンソルへの可変参照を取得
    pub fn tensor_mut(&mut self) -> &mut Tensor<T, D> {
        &mut self.tensor
    }

    /// 動的次元テンソルへの参照を取得
    pub fn as_dyn(&self) -> &Tensor<T, DimDyn> {
        &self.tensor_dyn
    }

    /// 動的次元テンソルへの可変参照を取得
    pub fn as_dyn_mut(&mut self) -> &mut Tensor<T, DimDyn> {
        &mut self.tensor_dyn
    }
}

impl<T: FloatDType, D: Dimension> ParameterMut<T> for Parameter<T, D> {
    fn tensor_dyn(&self) -> &Tensor<T, DimDyn> {
        &self.tensor_dyn
    }

    fn tensor_dyn_mut(&mut self) -> &mut Tensor<T, DimDyn> {
        &mut self.tensor_dyn
    }

    fn grad_generic(&self) -> Option<Tensor<T, DimDyn>> {
        self.tensor_dyn.grad_generic()
    }

    fn realize(&self) -> Result<(), String> {
        self.tensor_dyn
            .realize()
            .map(|_| ())
            .map_err(|e| format!("Failed to realize: {:?}", e))
    }

    fn data(&self) -> Option<Vec<T>> {
        self.tensor_dyn.data()
    }

    fn set_dyn(&mut self, tensor: Tensor<T, DimDyn>) {
        let tensor = tensor.set_requires_grad(true);
        // 動的次元テンソルを更新し、静的次元も同期
        self.tensor_dyn = tensor.clone();
        // Note: 静的次元への変換は次元数が一致している前提
        self.tensor = tensor.into_dimensioned();
    }

    fn zero_grad(&self) {
        self.tensor_dyn.zero_grad();
    }

    fn shape(&self) -> &[usize] {
        self.tensor_dyn.shape()
    }
}

// ============================================================================
// Module trait
// ============================================================================

/// ニューラルネットワークモジュールの基底トレイト
///
/// 学習可能なパラメータを持つ計算ユニットを表現します。
/// `forward()` メソッドは各実装で独自に定義してください。
///
/// # Type Parameters
///
/// * `T` - パラメータのデータ型（デフォルト: f32）
pub trait Module<T: FloatDType = f32> {
    /// 名前付きパラメータへの可変参照を取得
    ///
    /// ネストしたモジュールの場合、ドット区切りで名前空間化します。
    /// 例: `layer1.weight`, `layer1.bias`
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>>;

    /// 名前付きパラメータをロード（チェックポイント復元用）
    fn load_parameters(&mut self, params: HashMap<String, Tensor<T, DimDyn>>);

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

/// Module実装のためのヘルパーマーカー
///
/// ジェネリックなModule実装で型パラメータを明示するために使用
pub struct ModuleMarker<T: FloatDType>(PhantomData<T>);

impl<T: FloatDType> Default for ModuleMarker<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harp::tensor::{Dim1, Dim2};

    #[test]
    fn test_parameter_new() {
        let t = Tensor::<f32, Dim2>::zeros([3, 4]);
        let p = Parameter::new(t);
        assert_eq!(p.shape(), &[3, 4]);
    }

    #[test]
    fn test_parameter_static_access() {
        let t = Tensor::<f32, Dim2>::zeros([2, 3]);
        let p = Parameter::new(t);
        // 静的次元でアクセス
        assert_eq!(p.tensor().shape(), &[2, 3]);
        // 動的次元でアクセス
        assert_eq!(p.as_dyn().shape(), &[2, 3]);
    }

    #[test]
    fn test_parameter_mut_trait() {
        let t = Tensor::<f32, Dim2>::zeros([2, 2]);
        let mut p = Parameter::new(t);

        // ParameterMut トレイト経由でアクセス
        let param_ref: &mut dyn ParameterMut<f32> = &mut p;
        assert_eq!(param_ref.shape(), &[2, 2]);
        assert_eq!(param_ref.tensor_dyn().ndim(), 2);
    }

    #[test]
    fn test_parameter_set_dyn() {
        let t1 = Tensor::<f32, Dim2>::zeros([2, 2]);
        let mut p = Parameter::new(t1);

        let t2 = Tensor::<f32, Dim2>::ones([2, 2]).into_dyn();
        p.set_dyn(t2);

        assert_eq!(p.shape(), &[2, 2]);
    }

    #[test]
    fn test_parameter_f64() {
        let t = Tensor::<f64, Dim1>::zeros([5]);
        let p = Parameter::<f64, Dim1>::new(t);
        assert_eq!(p.shape(), &[5]);
    }
}
