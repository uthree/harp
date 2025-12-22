//! 初期化の高級演算 (Initialization HLOPs)
//!
//! primops の組み合わせで実装される初期化演算:
//! - Randn = Box-Muller変換 (Rand + Log2 + Sqrt + Sin/Cos)

use std::f32::consts::PI;
use std::ops;

use crate::differentiable::Differentiable;
use crate::primops::{Cos, Log2, Rand, Randn, Sqrt};
use crate::shape::IntoShape;

// ============================================================================
// Randn デフォルト実装 (Box-Muller法)
// ============================================================================

/// ボックス・ミュラー法を使用した正規乱数生成のためのヘルパートレイト
///
/// 内部でのみ使用される定数を提供します。
pub trait BoxMullerConstants {
    /// 2.0の定数
    fn two() -> Self;
    /// 2πの定数
    fn two_pi() -> Self;
    /// ln(2)の定数（log2からlnへの変換に使用）
    fn ln_2() -> Self;
}

impl BoxMullerConstants for f32 {
    fn two() -> Self {
        2.0
    }
    fn two_pi() -> Self {
        2.0 * PI
    }
    fn ln_2() -> Self {
        std::f32::consts::LN_2
    }
}

impl BoxMullerConstants for f64 {
    fn two() -> Self {
        2.0
    }
    fn two_pi() -> Self {
        2.0 * std::f64::consts::PI
    }
    fn ln_2() -> Self {
        std::f64::consts::LN_2
    }
}

/// Randn のデフォルト実装を提供するトレイト
///
/// ボックス・ミュラー法を使用して一様乱数から正規乱数を生成します:
/// Z = sqrt(-2 * ln(U1)) * cos(2π * U2)
///
/// ここで:
/// - U1, U2 は独立な一様乱数 [0, 1)
/// - ln(x) = log2(x) * ln(2) として計算
pub trait RandnDefault: Sized {
    fn randn_default(shape: &[usize]) -> Self;
}

impl<T> RandnDefault for T
where
    T: Rand
        + Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Neg<Output = T>
        + Log2
        + Sqrt
        + Cos
        + BoxMullerConstants,
{
    fn randn_default(shape: &[usize]) -> Self {
        // 2つの独立した一様乱数を生成
        let u1 = T::rand(shape);
        let u2 = T::rand(shape);

        // ln(u1) = log2(u1) * ln(2)
        let ln_u1 = u1.log2() * T::ln_2();

        // sqrt(-2 * ln(u1))
        let r = ((-ln_u1) * T::two()).sqrt();

        // 2π * u2
        let theta = u2 * T::two_pi();

        // r * cos(theta)
        r * theta.cos()
    }
}

// ============================================================================
// Differentiable<T> への Randn 実装
// ============================================================================

impl<T> Differentiable<T>
where
    T: Randn + 'static,
{
    /// 指定した形状の標準正規分布 N(0, 1) 変数を作成
    ///
    /// 勾配追跡なしで作成される。
    pub fn randn(shape: &[usize]) -> Differentiable<T> {
        Differentiable::new_no_grad(T::randn(shape))
    }

    /// 指定した形状の標準正規分布 N(0, 1) 変数を作成（IntoShape版）
    pub fn randn_shape<S: IntoShape>(shape: S) -> Differentiable<T> {
        Differentiable::new_no_grad(T::randn(&shape.into_shape()))
    }
}
