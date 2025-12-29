//! ニューラルネットワーク層
//!
//! 基本的な層の実装を提供します。
//!
//! # 層の種類
//!
//! ## パラメータを持つ層
//! - [`Linear`] - 全結合層
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
mod linear;

pub use activation::{ELU, GELU, LeakyReLU, Mish, ReLU, SiLU, Sigmoid, Softplus, Swish, Tanh};
pub use linear::Linear;
