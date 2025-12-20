use std::ops;

use crate::traits::{GradFn, GradNode};
use crate::variable::Variable;

// ============================================================================
// AddBackward (加算の逆伝播)
// ============================================================================

/// 加算の勾配関数
/// z = lhs + rhs の場合、∂L/∂lhs = ∂L/∂z, ∂L/∂rhs = ∂L/∂z
pub struct AddBackward<T: 'static> {
    lhs: Variable<T>,
    rhs: Variable<T>,
}

impl<T: 'static> AddBackward<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T> GradFn<Variable<T>> for AddBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 加算の勾配: 両方に同じ勾配を伝播
        self.lhs.backward_with(grad_y.clone());
        self.rhs.backward_with(grad_y);
    }
}

// ============================================================================
// MulBackward (乗算の逆伝播)
// ============================================================================

/// 乗算の勾配関数
/// z = lhs * rhs の場合、∂L/∂lhs = ∂L/∂z * rhs, ∂L/∂rhs = ∂L/∂z * lhs
pub struct MulBackward<T: 'static> {
    lhs: Variable<T>,
    rhs: Variable<T>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    lhs_value: T,
    rhs_value: T,
}

impl<T: Clone + 'static> MulBackward<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
        let lhs_value = lhs.value();
        let rhs_value = rhs.value();
        Self {
            lhs,
            rhs,
            lhs_value,
            rhs_value,
        }
    }
}

impl<T> GradFn<Variable<T>> for MulBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 乗算の勾配: ∂L/∂lhs = ∂L/∂z * rhs, ∂L/∂rhs = ∂L/∂z * lhs
        let grad_lhs_val = grad_y.value() * self.rhs_value.clone();
        self.lhs.backward_with(Variable::new(grad_lhs_val));

        let grad_rhs_val = grad_y.value() * self.lhs_value.clone();
        self.rhs.backward_with(Variable::new(grad_rhs_val));
    }
}

// ============================================================================
// NegBackward (符号反転の逆伝播)
// ============================================================================

/// 符号反転の勾配関数
/// z = -x の場合、∂L/∂x = -∂L/∂z
pub struct NegBackward<T: 'static> {
    input: Variable<T>,
}

impl<T: 'static> NegBackward<T> {
    pub fn new(input: Variable<T>) -> Self {
        Self { input }
    }
}

impl<T> GradFn<Variable<T>> for NegBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 符号反転の勾配: ∂L/∂x = -∂L/∂z
        let grad_val = -grad_y.value();
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// RecipBackward (逆数の逆伝播)
// ============================================================================

/// 逆数の勾配関数
/// z = 1/x の場合、∂L/∂x = -∂L/∂z / x²
pub struct RecipBackward<T: 'static> {
    input: Variable<T>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    input_value: T,
}

impl<T: Clone + 'static> RecipBackward<T> {
    pub fn new(input: Variable<T>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
        let input_value = input.value();
        Self { input, input_value }
    }
}

impl<T> GradFn<Variable<T>> for RecipBackward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + ops::Neg<Output = T>
        + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 逆数の勾配: ∂L/∂x = -∂L/∂z / x²
        let x = self.input_value.clone();
        let x_squared = x.clone() * x;
        let grad_val = -(grad_y.value() / x_squared);
        self.input.backward_with(Variable::new(grad_val));
    }
}

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
