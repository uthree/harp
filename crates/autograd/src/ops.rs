use std::ops;

use crate::arithmetic::{Add, Mul, Neg};
use crate::traits::{GradNode, GradientInto};
use crate::variable::Variable;

// ============================================================================
// 2項演算子の4パターン生成マクロ
// ============================================================================

/// 2項演算子の全組み合わせを生成するマクロ
/// - &Variable op &Variable (基本実装)
/// - Variable op Variable
/// - Variable op &Variable
/// - &Variable op Variable
macro_rules! impl_binary_op {
    // Add演算子
    (Add, $lhs:ident, $rhs:ident, $out:ident, $method:ident, $grad_fn:ident, [$($bounds:tt)*]) => {
        // &Variable<L> + &Variable<R> -> Variable<O> (基本実装)
        impl<$lhs, $rhs, $out> ops::Add<&Variable<$rhs>> for &Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;

            fn $method(self, rhs: &Variable<$rhs>) -> Self::Output {
                let lhs_val = self.value();
                let rhs_val = rhs.value();
                Variable::with_grad_fn(
                    lhs_val + rhs_val,
                    Box::new($grad_fn::new(self.clone(), rhs.clone())),
                )
            }
        }

        // Variable<L> + Variable<R>
        impl<$lhs, $rhs, $out> ops::Add<Variable<$rhs>> for Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: Variable<$rhs>) -> Self::Output {
                &self + &rhs
            }
        }

        // Variable<L> + &Variable<R>
        impl<$lhs, $rhs, $out> ops::Add<&Variable<$rhs>> for Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: &Variable<$rhs>) -> Self::Output {
                &self + rhs
            }
        }

        // &Variable<L> + Variable<R>
        impl<$lhs, $rhs, $out> ops::Add<Variable<$rhs>> for &Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: Variable<$rhs>) -> Self::Output {
                self + &rhs
            }
        }
    };

    // Mul演算子
    (Mul, $lhs:ident, $rhs:ident, $out:ident, $method:ident, $grad_fn:ident, [$($bounds:tt)*]) => {
        // &Variable<L> * &Variable<R> -> Variable<O> (基本実装)
        impl<$lhs, $rhs, $out> ops::Mul<&Variable<$rhs>> for &Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;

            fn $method(self, rhs: &Variable<$rhs>) -> Self::Output {
                let lhs_val = self.value();
                let rhs_val = rhs.value();
                Variable::with_grad_fn(
                    lhs_val * rhs_val,
                    Box::new($grad_fn::new(self.clone(), rhs.clone())),
                )
            }
        }

        // Variable<L> * Variable<R>
        impl<$lhs, $rhs, $out> ops::Mul<Variable<$rhs>> for Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: Variable<$rhs>) -> Self::Output {
                &self * &rhs
            }
        }

        // Variable<L> * &Variable<R>
        impl<$lhs, $rhs, $out> ops::Mul<&Variable<$rhs>> for Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: &Variable<$rhs>) -> Self::Output {
                &self * rhs
            }
        }

        // &Variable<L> * Variable<R>
        impl<$lhs, $rhs, $out> ops::Mul<Variable<$rhs>> for &Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: Variable<$rhs>) -> Self::Output {
                self * &rhs
            }
        }
    };

    // Sub演算子
    (Sub, $lhs:ident, $rhs:ident, $out:ident, $method:ident, [$($bounds:tt)*]) => {
        // &Variable<L> - &Variable<R> -> Variable<O> (基本実装: Neg + Add)
        impl<$lhs, $rhs, $out> ops::Sub<&Variable<$rhs>> for &Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;

            fn $method(self, rhs: &Variable<$rhs>) -> Variable<$out> {
                let neg_rhs = -rhs;
                self + &neg_rhs
            }
        }

        // Variable<L> - Variable<R>
        impl<$lhs, $rhs, $out> ops::Sub<Variable<$rhs>> for Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: Variable<$rhs>) -> Variable<$out> {
                &self - &rhs
            }
        }

        // Variable<L> - &Variable<R>
        impl<$lhs, $rhs, $out> ops::Sub<&Variable<$rhs>> for Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: &Variable<$rhs>) -> Variable<$out> {
                &self - rhs
            }
        }

        // &Variable<L> - Variable<R>
        impl<$lhs, $rhs, $out> ops::Sub<Variable<$rhs>> for &Variable<$lhs>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self, rhs: Variable<$rhs>) -> Variable<$out> {
                self - &rhs
            }
        }
    };

    // Div演算子 (TODO: Mul + Recip を組み合わせた実装に変更する)
    (Div, $t:ident, [$($bounds:tt)*]) => {
        // &Variable<T> / &Variable<T> -> Variable<T> (基本実装)
        impl<$t> ops::Div<&Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;

            fn div(self, rhs: &Variable<$t>) -> Variable<$t> {
                let lhs_val = self.value();
                let rhs_val = rhs.value();
                Variable::new(lhs_val / rhs_val)
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

/// 単項演算子の全組み合わせを生成するマクロ
macro_rules! impl_unary_op {
    (Neg, $inp:ident, $out:ident, $method:ident, $grad_fn:ident, [$($bounds:tt)*]) => {
        // -&Variable<I> -> Variable<O> (基本実装)
        impl<$inp, $out> ops::Neg for &Variable<$inp>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;

            fn $method(self) -> Self::Output {
                let val = self.value();
                Variable::with_grad_fn(-val, Box::new($grad_fn::new(self.clone())))
            }
        }

        // -Variable<I> -> Variable<O>
        impl<$inp, $out> ops::Neg for Variable<$inp>
        where
            $($bounds)*
        {
            type Output = Variable<$out>;
            fn $method(self) -> Self::Output {
                -&self
            }
        }
    };
}

// ============================================================================
// 演算子の実装
// ============================================================================

// Add: L + R -> O
impl_binary_op!(Add, L, R, O, add, Add, [
    L: GradNode + ops::Add<L, Output = L> + ops::Add<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + 'static,
    O: Clone + 'static,
    Variable<O>: GradientInto<Variable<L>> + GradientInto<Variable<R>> + Clone,
]);

// Mul: L * R -> O
impl_binary_op!(Mul, L, R, O, mul, Mul, [
    L: GradNode + ops::Add<L, Output = L> + ops::Mul<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + 'static,
    O: Clone + ops::Mul<R, Output = L> + ops::Mul<L, Output = R> + 'static,
]);

// Sub: L - R -> O (Neg + Add で実装)
impl_binary_op!(Sub, L, R, O, sub, [
    L: GradNode + ops::Add<L, Output = L> + ops::Add<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + ops::Neg<Output = R> + 'static,
    O: Clone + 'static,
    Variable<O>: GradientInto<Variable<L>> + GradientInto<Variable<R>> + Clone,
    Variable<R>: Clone,
    for<'a> &'a Variable<R>: ops::Neg<Output = Variable<R>>,
]);

// Div: T / T -> T (TODO: Mul + Recip を組み合わせた実装に変更する)
impl_binary_op!(Div, T, [
    T: ops::Div<T, Output = T> + Clone + 'static,
]);

// Neg: -I -> O
impl_unary_op!(Neg, I, O, neg, Neg, [
    I: GradNode + ops::Add<I, Output = I> + ops::Neg<Output = O> + 'static,
    O: Clone + ops::Neg<Output = I> + 'static,
]);
