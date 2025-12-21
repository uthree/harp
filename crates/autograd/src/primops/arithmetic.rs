//! 四則演算のプリミティブ演算（Add, Mul, Neg, Recip）

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
        let requires_grad = grad_y.requires_grad();

        let grad_lhs_val = grad_y.value() * self.rhs_value.clone();
        self.lhs.backward_with(Variable::new_with_requires_grad(
            grad_lhs_val,
            requires_grad,
        ));

        let grad_rhs_val = grad_y.value() * self.lhs_value.clone();
        self.rhs.backward_with(Variable::new_with_requires_grad(
            grad_rhs_val,
            requires_grad,
        ));
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
        let requires_grad = grad_y.requires_grad();
        let grad_val = -grad_y.value();
        self.input
            .backward_with(Variable::new_with_requires_grad(grad_val, requires_grad));
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
        let requires_grad = grad_y.requires_grad();
        let x = self.input_value.clone();
        let x_squared = x.clone() * x;
        let grad_val = -(grad_y.value() / x_squared);
        self.input
            .backward_with(Variable::new_with_requires_grad(grad_val, requires_grad));
    }
}

// ============================================================================
// 2項演算子の生成マクロ（primops用）
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
                let result_val = lhs_val + rhs_val;

                // どちらかが requires_grad=true なら逆伝播パスを作成
                if self.requires_grad() || rhs.requires_grad() {
                    Variable::with_grad_fn(
                        result_val,
                        Box::new($grad_fn::new(self.clone(), rhs.clone())),
                    )
                } else {
                    Variable::new_no_grad(result_val)
                }
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
                let result_val = lhs_val * rhs_val;

                // どちらかが requires_grad=true なら逆伝播パスを作成
                if self.requires_grad() || rhs.requires_grad() {
                    Variable::with_grad_fn(
                        result_val,
                        Box::new($grad_fn::new(self.clone(), rhs.clone())),
                    )
                } else {
                    Variable::new_no_grad(result_val)
                }
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
                let result_val = -val;

                // requires_grad=true なら逆伝播パスを作成
                if self.requires_grad() {
                    Variable::with_grad_fn(result_val, Box::new($grad_fn::new(self.clone())))
                } else {
                    Variable::new_no_grad(result_val)
                }
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
// 演算子の実装（primopsのみ）
// ============================================================================

// Add: T + T -> T
impl_binary_op!(Add, T, add, AddBackward, [
    T: GradNode + ops::Add<T, Output = T> + 'static,
]);

// Mul: T * T -> T
impl_binary_op!(Mul, T, mul, MulBackward, [
    T: GradNode + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + 'static,
]);

// Neg: -T -> T
impl_unary_op!(Neg, T, neg, NegBackward, [
    T: GradNode + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
]);

// ============================================================================
// Rem (剰余演算) - fmod相当
// ============================================================================

/// 剰余演算を表すトレイト
pub trait RemOp: Sized {
    fn rem(&self, other: &Self) -> Self;
}

/// 床関数を表すトレイト（剰余の勾配計算に使用）
pub trait Floor: Sized {
    fn floor(&self) -> Self;
}

/// 剰余の勾配関数
/// z = x % y の場合:
/// - ∂L/∂x = ∂L/∂z（xの変化はそのまま出力に反映）
/// - ∂L/∂y = -∂L/∂z * floor(x/y)
pub struct RemBackward<T: 'static> {
    lhs: Variable<T>,
    rhs: Variable<T>,
    lhs_value: T,
    rhs_value: T,
}

impl<T: Clone + 'static> RemBackward<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
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

impl<T> GradFn<Variable<T>> for RemBackward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + ops::Neg<Output = T>
        + Floor
        + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        let requires_grad = grad_y.requires_grad();

        // ∂L/∂x = ∂L/∂z
        self.lhs.backward_with(grad_y.clone());

        // ∂L/∂y = -∂L/∂z * floor(x/y)
        let quotient = self.lhs_value.clone() / self.rhs_value.clone();
        let floor_quotient = quotient.floor();
        let grad_rhs_val = -(grad_y.value() * floor_quotient);
        self.rhs.backward_with(Variable::new_with_requires_grad(
            grad_rhs_val,
            requires_grad,
        ));
    }
}

// Variable<T> への Rem 実装
impl<T> Variable<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + ops::Neg<Output = T>
        + RemOp
        + Floor
        + 'static,
{
    /// 剰余を計算
    pub fn rem(&self, other: &Variable<T>) -> Variable<T> {
        let output = self.value().rem(&other.value());
        if self.requires_grad() || other.requires_grad() {
            Variable::with_grad_fn(
                output,
                Box::new(RemBackward::new(self.clone(), other.clone())),
            )
        } else {
            Variable::new_no_grad(output)
        }
    }
}

// f32/f64 への RemOp と Floor 実装
impl RemOp for f32 {
    fn rem(&self, other: &Self) -> Self {
        self % other
    }
}

impl RemOp for f64 {
    fn rem(&self, other: &Self) -> Self {
        self % other
    }
}

impl Floor for f32 {
    fn floor(&self) -> Self {
        f32::floor(*self)
    }
}

impl Floor for f64 {
    fn floor(&self) -> Self {
        f64::floor(*self)
    }
}

// ============================================================================
// Maximum (二項max演算) - 2つの値のうち大きい方を返す
// ============================================================================

/// 二項max演算を表すトレイト
/// reduce::Max（軸に沿った縮約）とは区別される
pub trait Maximum<Rhs = Self>: Sized {
    type Output;
    /// 2つの値を比較して大きい方を返す
    fn maximum(&self, other: &Rhs) -> Self::Output;
    /// 左側オペランドに対する勾配を計算
    /// lhs >= rhs なら grad、そうでなければ 0
    fn maximum_grad_lhs(grad: &Self::Output, lhs: &Self, rhs: &Rhs) -> Self;
    /// 右側オペランドに対する勾配を計算
    /// rhs > lhs なら grad、そうでなければ 0
    fn maximum_grad_rhs(grad: &Self::Output, lhs: &Self, rhs: &Rhs) -> Rhs;
}

/// 二項max演算の勾配関数
/// z = maximum(x, y) の場合:
/// - ∂L/∂x = ∂L/∂z if x >= y, else 0
/// - ∂L/∂y = ∂L/∂z if y > x, else 0
pub struct MaximumBackward<T: 'static> {
    lhs: Variable<T>,
    rhs: Variable<T>,
    lhs_value: T,
    rhs_value: T,
}

impl<T: Clone + 'static> MaximumBackward<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
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

impl<T> GradFn<Variable<T>> for MaximumBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Maximum<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        let requires_grad = grad_y.requires_grad();

        // ∂L/∂lhs = grad if lhs >= rhs, else 0
        let grad_lhs_val = T::maximum_grad_lhs(&grad_y.value(), &self.lhs_value, &self.rhs_value);
        self.lhs.backward_with(Variable::new_with_requires_grad(
            grad_lhs_val,
            requires_grad,
        ));

        // ∂L/∂rhs = grad if rhs > lhs, else 0
        let grad_rhs_val = T::maximum_grad_rhs(&grad_y.value(), &self.lhs_value, &self.rhs_value);
        self.rhs.backward_with(Variable::new_with_requires_grad(
            grad_rhs_val,
            requires_grad,
        ));
    }
}

// Variable<T> への Maximum 実装
impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + Maximum<T, Output = T> + 'static,
{
    /// 2つの変数のうち大きい方を返す
    pub fn maximum(&self, other: &Variable<T>) -> Variable<T> {
        let output = self.value().maximum(&other.value());
        if self.requires_grad() || other.requires_grad() {
            Variable::with_grad_fn(
                output,
                Box::new(MaximumBackward::new(self.clone(), other.clone())),
            )
        } else {
            Variable::new_no_grad(output)
        }
    }
}

// f32 への Maximum 実装
impl Maximum for f32 {
    type Output = f32;

    fn maximum(&self, other: &f32) -> f32 {
        if *self >= *other { *self } else { *other }
    }

    fn maximum_grad_lhs(grad: &f32, lhs: &f32, rhs: &f32) -> f32 {
        if *lhs >= *rhs { *grad } else { 0.0 }
    }

    fn maximum_grad_rhs(grad: &f32, lhs: &f32, rhs: &f32) -> f32 {
        if *rhs > *lhs { *grad } else { 0.0 }
    }
}

// f64 への Maximum 実装
impl Maximum for f64 {
    type Output = f64;

    fn maximum(&self, other: &f64) -> f64 {
        if *self >= *other { *self } else { *other }
    }

    fn maximum_grad_lhs(grad: &f64, lhs: &f64, rhs: &f64) -> f64 {
        if *lhs >= *rhs { *grad } else { 0.0 }
    }

    fn maximum_grad_rhs(grad: &f64, lhs: &f64, rhs: &f64) -> f64 {
        if *rhs > *lhs { *grad } else { 0.0 }
    }
}
