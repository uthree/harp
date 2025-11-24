//! 勾配関数（GradFn）の実装
//!
//! 各演算に対する勾配計算を実装したモジュール群です。
//!
//! # モジュール構成
//!
//! - `basic`: 基本演算（加算、乗算、否定、逆数等）の勾配
//! - `math`: 数学関数（対数、指数、三角関数、平方根等）の勾配
//! - `memory`: メモリ操作（パディング、スライス）の勾配
//! - `conv`: 畳み込み演算（Conv1d, Conv2d, Conv3d）の勾配

use crate::autograd::Tensor;

/// 勾配関数のトレイト
///
/// 各演算に対する微分を定義します。
/// `backward`メソッドで上流からの勾配を受け取り、入力に対する勾配を計算します。
pub trait GradFn: std::fmt::Debug {
    /// 勾配を計算
    ///
    /// # 引数
    /// - `grad_output`: 出力に対する勾配
    /// - `inputs`: 元の入力テンソル
    ///
    /// # 戻り値
    /// 各入力に対する勾配（`None`は勾配不要を意味する）
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>>;
}

// 基本演算の勾配関数
mod basic;
pub use basic::*;

// 数学関数の勾配関数
mod math;
pub use math::*;

// メモリ操作の勾配関数
mod memory;
pub use memory::*;

// 畳み込み演算の勾配関数
mod conv;
pub use conv::*;
