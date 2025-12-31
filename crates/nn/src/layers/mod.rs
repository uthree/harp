//! ニューラルネットワーク層
//!
//! 基本的な層の実装を提供します。
//!
//! # 層の種類
//!
//! ## パラメータを持つ層
//! - [`Linear`] - 全結合層
//! - [`Conv1d`] - 1D畳み込み層
//! - [`Conv2d`] - 2D畳み込み層
//! - [`Conv3d`] - 3D畳み込み層
//! - [`ConvTranspose1d`] - 1D転置畳み込み層
//! - [`ConvTranspose2d`] - 2D転置畳み込み層
//! - [`ConvTranspose3d`] - 3D転置畳み込み層
//!
//! ## プーリング層（パラメータなし）
//! - [`MaxPool1d`] - 1D最大プーリング
//! - [`MaxPool2d`] - 2D最大プーリング
//! - [`MaxPool3d`] - 3D最大プーリング
//! - [`AvgPool1d`] - 1D平均プーリング
//! - [`AvgPool2d`] - 2D平均プーリング
//! - [`AvgPool3d`] - 3D平均プーリング
//! - [`AdaptiveAvgPool2d`] - 2D適応的平均プーリング
//! - [`AdaptiveMaxPool2d`] - 2D適応的最大プーリング
//! - [`GlobalAvgPool2d`] - 2Dグローバル平均プーリング
//! - [`GlobalMaxPool2d`] - 2Dグローバル最大プーリング
//!
//! ## 活性化関数層（パラメータなし）
//! - [`ReLU`] - ReLU 活性化関数
//! - [`LeakyReLU`] - Leaky ReLU 活性化関数
//! - [`Sigmoid`] - Sigmoid 活性化関数
//! - [`Tanh`] - Tanh 活性化関数
//! - [`GELU`] - GELU 活性化関数（高速近似）
//! - [`SiLU`] - SiLU (Swish) 活性化関数
//! - [`Softplus`] - Softplus 活性化関数
//! - [`Mish`] - Mish 活性化関数
//! - [`ELU`] - ELU 活性化関数
//! - [`Activation`] - 汎用活性化層

mod activation;
mod conv;
mod conv_transpose;
mod linear;
mod pooling;

pub use activation::{ELU, GELU, LeakyReLU, Mish, ReLU, SiLU, Sigmoid, Softplus, Swish, Tanh};
pub use conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig};
pub use conv_transpose::{
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig,
    ConvTranspose3d, ConvTranspose3dConfig,
};
pub use linear::Linear;
pub use pooling::{
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool1d, AvgPool2d, AvgPool3d, GlobalAvgPool2d,
    GlobalMaxPool2d, MaxPool1d, MaxPool2d, MaxPool3d,
};
