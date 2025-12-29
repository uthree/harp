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
mod linear;

pub use activation::{ELU, GELU, LeakyReLU, Mish, ReLU, SiLU, Sigmoid, Softplus, Swish, Tanh};
pub use conv::{
    Conv1d, Conv1dBuilder, Conv2d, Conv2dBuilder, Conv3d, Conv3dBuilder, ConvTranspose1d,
    ConvTranspose1dBuilder, ConvTranspose2d, ConvTranspose2dBuilder, ConvTranspose3d,
    ConvTranspose3dBuilder,
};
pub use linear::Linear;
