//! 高級四則演算（Sub, Div）
//!
//! これらはprimopsの組み合わせで実装されます：
//! - Sub = Add + Neg
//! - Div = Mul + Recip

use std::ops;

use crate::differentiable::Differentiable;
use crate::primops::RecipBackward;
use crate::traits::GradNode;

// ============================================================================
// Sub演算子の実装 (Add + Neg)
// ============================================================================

/// Sub演算子の全組み合わせを生成するマクロ
macro_rules! impl_sub_op {
    ($t:ident, [$($bounds:tt)*]) => {
        // &Differentiable<T> - &Differentiable<T> -> Differentiable<T> (基本実装: Neg + Add)
        impl<$t> ops::Sub<&Differentiable<$t>> for &Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;

            fn sub(self, rhs: &Differentiable<$t>) -> Differentiable<$t> {
                let neg_rhs = -rhs;
                self + &neg_rhs
            }
        }

        // Differentiable<T> - Differentiable<T>
        impl<$t> ops::Sub<Differentiable<$t>> for Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;
            fn sub(self, rhs: Differentiable<$t>) -> Differentiable<$t> {
                &self - &rhs
            }
        }

        // Differentiable<T> - &Differentiable<T>
        impl<$t> ops::Sub<&Differentiable<$t>> for Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;
            fn sub(self, rhs: &Differentiable<$t>) -> Differentiable<$t> {
                &self - rhs
            }
        }

        // &Differentiable<T> - Differentiable<T>
        impl<$t> ops::Sub<Differentiable<$t>> for &Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;
            fn sub(self, rhs: Differentiable<$t>) -> Differentiable<$t> {
                self - &rhs
            }
        }
    };
}

// Sub: T - T -> T (Neg + Add で実装)
impl_sub_op!(T, [
    T: GradNode + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
    for<'a> &'a Differentiable<T>: ops::Neg<Output = Differentiable<T>>,
]);

// ============================================================================
// Div演算子の実装 (Mul + Recip)
// ============================================================================

/// Div演算子の全組み合わせを生成するマクロ
macro_rules! impl_div_op {
    ($t:ident, [$($bounds:tt)*]) => {
        // &Differentiable<T> / &Differentiable<T> -> Differentiable<T> (基本実装: Recip + Mul)
        impl<$t> ops::Div<&Differentiable<$t>> for &Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;

            fn div(self, rhs: &Differentiable<$t>) -> Differentiable<$t> {
                // 1/rhs を計算
                let rhs_val = rhs.value();
                let one = $t::one();
                let recip_val = one / rhs_val;
                let recip = Differentiable::with_grad_fn(
                    recip_val,
                    Box::new(RecipBackward::new(rhs.clone())),
                );
                // self * (1/rhs)
                self * &recip
            }
        }

        // Differentiable<T> / Differentiable<T>
        impl<$t> ops::Div<Differentiable<$t>> for Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;
            fn div(self, rhs: Differentiable<$t>) -> Differentiable<$t> {
                &self / &rhs
            }
        }

        // Differentiable<T> / &Differentiable<T>
        impl<$t> ops::Div<&Differentiable<$t>> for Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;
            fn div(self, rhs: &Differentiable<$t>) -> Differentiable<$t> {
                &self / rhs
            }
        }

        // &Differentiable<T> / Differentiable<T>
        impl<$t> ops::Div<Differentiable<$t>> for &Differentiable<$t>
        where
            $($bounds)*
        {
            type Output = Differentiable<$t>;
            fn div(self, rhs: Differentiable<$t>) -> Differentiable<$t> {
                self / &rhs
            }
        }
    };
}

/// 単位元を取得するトレイト
pub trait One {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

// Div: T / T -> T (Mul + Recip で実装)
impl_div_op!(T, [
    T: GradNode
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + ops::Neg<Output = T>
        + One
        + 'static,
]);
