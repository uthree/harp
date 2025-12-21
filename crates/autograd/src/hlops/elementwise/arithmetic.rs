//! 高級四則演算（Sub, Div）
//!
//! これらはprimopsの組み合わせで実装されます：
//! - Sub = Add + Neg
//! - Div = Mul + Recip

use std::ops;

use crate::primops::RecipBackward;
use crate::traits::GradNode;
use crate::variable::Variable;

// ============================================================================
// Sub演算子の実装 (Add + Neg)
// ============================================================================

/// Sub演算子の全組み合わせを生成するマクロ
macro_rules! impl_sub_op {
    ($t:ident, [$($bounds:tt)*]) => {
        // &Variable<T> - &Variable<T> -> Variable<T> (基本実装: Neg + Add)
        impl<$t> ops::Sub<&Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;

            fn sub(self, rhs: &Variable<$t>) -> Variable<$t> {
                let neg_rhs = -rhs;
                self + &neg_rhs
            }
        }

        // Variable<T> - Variable<T>
        impl<$t> ops::Sub<Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn sub(self, rhs: Variable<$t>) -> Variable<$t> {
                &self - &rhs
            }
        }

        // Variable<T> - &Variable<T>
        impl<$t> ops::Sub<&Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn sub(self, rhs: &Variable<$t>) -> Variable<$t> {
                &self - rhs
            }
        }

        // &Variable<T> - Variable<T>
        impl<$t> ops::Sub<Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn sub(self, rhs: Variable<$t>) -> Variable<$t> {
                self - &rhs
            }
        }
    };
}

// Sub: T - T -> T (Neg + Add で実装)
impl_sub_op!(T, [
    T: GradNode + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
    for<'a> &'a Variable<T>: ops::Neg<Output = Variable<T>>,
]);

// ============================================================================
// Div演算子の実装 (Mul + Recip)
// ============================================================================

/// Div演算子の全組み合わせを生成するマクロ
macro_rules! impl_div_op {
    ($t:ident, [$($bounds:tt)*]) => {
        // &Variable<T> / &Variable<T> -> Variable<T> (基本実装: Recip + Mul)
        impl<$t> ops::Div<&Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;

            fn div(self, rhs: &Variable<$t>) -> Variable<$t> {
                // 1/rhs を計算
                let rhs_val = rhs.value();
                let one = $t::one();
                let recip_val = one / rhs_val;
                let recip = Variable::with_grad_fn(
                    recip_val,
                    Box::new(RecipBackward::new(rhs.clone())),
                );
                // self * (1/rhs)
                self * &recip
            }
        }

        // Variable<T> / Variable<T>
        impl<$t> ops::Div<Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn div(self, rhs: Variable<$t>) -> Variable<$t> {
                &self / &rhs
            }
        }

        // Variable<T> / &Variable<T>
        impl<$t> ops::Div<&Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn div(self, rhs: &Variable<$t>) -> Variable<$t> {
                &self / rhs
            }
        }

        // &Variable<T> / Variable<T>
        impl<$t> ops::Div<Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn div(self, rhs: Variable<$t>) -> Variable<$t> {
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
