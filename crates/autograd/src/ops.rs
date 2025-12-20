use std::ops;

use crate::arithmetic::{AddBackward, MulBackward, NegBackward};
use crate::traits::GradNode;
use crate::variable::Variable;

// ============================================================================
// 2項演算子の4パターン生成マクロ
// ============================================================================

/// 2項演算子の全組み合わせを生成するマクロ（単一型パラメータ版）
macro_rules! impl_binary_op {
    // Add演算子
    (Add, $t:ident, $method:ident, $grad_fn:ident, [$($bounds:tt)*]) => {
        // &Variable<T> + &Variable<T> -> Variable<T> (基本実装)
        impl<$t> ops::Add<&Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;

            fn $method(self, rhs: &Variable<$t>) -> Self::Output {
                let lhs_val = self.value();
                let rhs_val = rhs.value();
                Variable::with_grad_fn(
                    lhs_val + rhs_val,
                    Box::new($grad_fn::new(self.clone(), rhs.clone())),
                )
            }
        }

        // Variable<T> + Variable<T>
        impl<$t> ops::Add<Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: Variable<$t>) -> Self::Output {
                &self + &rhs
            }
        }

        // Variable<T> + &Variable<T>
        impl<$t> ops::Add<&Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: &Variable<$t>) -> Self::Output {
                &self + rhs
            }
        }

        // &Variable<T> + Variable<T>
        impl<$t> ops::Add<Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: Variable<$t>) -> Self::Output {
                self + &rhs
            }
        }
    };

    // Mul演算子
    (Mul, $t:ident, $method:ident, $grad_fn:ident, [$($bounds:tt)*]) => {
        // &Variable<T> * &Variable<T> -> Variable<T> (基本実装)
        impl<$t> ops::Mul<&Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;

            fn $method(self, rhs: &Variable<$t>) -> Self::Output {
                let lhs_val = self.value();
                let rhs_val = rhs.value();
                Variable::with_grad_fn(
                    lhs_val * rhs_val,
                    Box::new($grad_fn::new(self.clone(), rhs.clone())),
                )
            }
        }

        // Variable<T> * Variable<T>
        impl<$t> ops::Mul<Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: Variable<$t>) -> Self::Output {
                &self * &rhs
            }
        }

        // Variable<T> * &Variable<T>
        impl<$t> ops::Mul<&Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: &Variable<$t>) -> Self::Output {
                &self * rhs
            }
        }

        // &Variable<T> * Variable<T>
        impl<$t> ops::Mul<Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: Variable<$t>) -> Self::Output {
                self * &rhs
            }
        }
    };

    // Sub演算子
    (Sub, $t:ident, $method:ident, [$($bounds:tt)*]) => {
        // &Variable<T> - &Variable<T> -> Variable<T> (基本実装: Neg + Add)
        impl<$t> ops::Sub<&Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;

            fn $method(self, rhs: &Variable<$t>) -> Variable<$t> {
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
            fn $method(self, rhs: Variable<$t>) -> Variable<$t> {
                &self - &rhs
            }
        }

        // Variable<T> - &Variable<T>
        impl<$t> ops::Sub<&Variable<$t>> for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: &Variable<$t>) -> Variable<$t> {
                &self - rhs
            }
        }

        // &Variable<T> - Variable<T>
        impl<$t> ops::Sub<Variable<$t>> for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self, rhs: Variable<$t>) -> Variable<$t> {
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

/// 単項演算子の全組み合わせを生成するマクロ（単一型パラメータ版）
macro_rules! impl_unary_op {
    (Neg, $t:ident, $method:ident, $grad_fn:ident, [$($bounds:tt)*]) => {
        // -&Variable<T> -> Variable<T> (基本実装)
        impl<$t> ops::Neg for &Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;

            fn $method(self) -> Self::Output {
                let val = self.value();
                Variable::with_grad_fn(-val, Box::new($grad_fn::new(self.clone())))
            }
        }

        // -Variable<T> -> Variable<T>
        impl<$t> ops::Neg for Variable<$t>
        where
            $($bounds)*
        {
            type Output = Variable<$t>;
            fn $method(self) -> Self::Output {
                -&self
            }
        }
    };
}

// ============================================================================
// 演算子の実装
// ============================================================================

// Add: T + T -> T
impl_binary_op!(Add, T, add, AddBackward, [
    T: GradNode + ops::Add<T, Output = T> + 'static,
]);

// Mul: T * T -> T
impl_binary_op!(Mul, T, mul, MulBackward, [
    T: GradNode + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + 'static,
]);

// Sub: T - T -> T (Neg + Add で実装)
impl_binary_op!(Sub, T, sub, [
    T: GradNode + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
    for<'a> &'a Variable<T>: ops::Neg<Output = Variable<T>>,
]);

// Div: T / T -> T (TODO: Mul + Recip を組み合わせた実装に変更する)
impl_binary_op!(Div, T, [
    T: ops::Div<T, Output = T> + Clone + 'static,
]);

// Neg: -T -> T
impl_unary_op!(Neg, T, neg, NegBackward, [
    T: GradNode + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
]);
