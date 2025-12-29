//! オプティマイザ
//!
//! パラメータ更新のための最適化アルゴリズムを提供します。
//!
//! # 実装済みオプティマイザ
//!
//! - [`SGD`] - 確率的勾配降下法
//! - [`Momentum`] - モメンタム付き SGD
//! - [`RMSProp`] - 勾配二乗の移動平均による適応的学習率
//! - [`Adam`] - 一次・二次モーメントによる適応的学習率

mod adam;
mod momentum;
mod rmsprop;
mod sgd;

pub use adam::Adam;
pub use momentum::Momentum;
pub use rmsprop::RMSProp;
pub use sgd::SGD;

use harp::tensor::FloatDType;

use crate::Module;

/// オプティマイザの基底トレイト
///
/// # Type Parameters
///
/// * `T` - パラメータのデータ型
pub trait Optimizer<T: FloatDType> {
    /// パラメータを更新（勾配降下ステップを実行）
    fn step<M: Module<T>>(&mut self, module: &mut M);
}
