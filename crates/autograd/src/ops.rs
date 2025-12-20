use std::ops;

use crate::grad_fns::{Add, Mul, Neg};
use crate::traits::GradNode;
use crate::variable::Variable;

// ============================================================================
// 演算子の実装（参照）
// ============================================================================

// &Variable<T> + &Variable<T> -> Variable<T>
impl<T> ops::Add<&Variable<T>> for &Variable<T>
where
    T: GradNode + ops::Add<T, Output = T> + 'static,
{
    type Output = Variable<T>;

    fn add(self, rhs: &Variable<T>) -> Self::Output {
        let lhs_val = self.value();
        let rhs_val = rhs.value();
        Variable::with_grad_fn(
            lhs_val + rhs_val,
            Box::new(Add::new(self.clone(), rhs.clone())),
        )
    }
}

// &Variable<T> * &Variable<T> -> Variable<T>
impl<T> ops::Mul<&Variable<T>> for &Variable<T>
where
    T: GradNode + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + 'static,
{
    type Output = Variable<T>;

    fn mul(self, rhs: &Variable<T>) -> Self::Output {
        let lhs_val = self.value();
        let rhs_val = rhs.value();
        Variable::with_grad_fn(
            lhs_val * rhs_val,
            Box::new(Mul::new(self.clone(), rhs.clone())),
        )
    }
}

// -&Variable<T> -> Variable<T>
impl<T> ops::Neg for &Variable<T>
where
    T: GradNode + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
{
    type Output = Variable<T>;

    fn neg(self) -> Self::Output {
        let val = self.value();
        Variable::with_grad_fn(-val, Box::new(Neg::new(self.clone())))
    }
}

// &Variable<T> - &Variable<T> -> Variable<T>
// TODO: Neg + Add を組み合わせた実装に変更する
impl<T> ops::Sub<&Variable<T>> for &Variable<T>
where
    T: ops::Sub<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;

    fn sub(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_val = self.value();
        let rhs_val = rhs.value();
        Variable::new(lhs_val - rhs_val)
    }
}

// &Variable<T> / &Variable<T> -> Variable<T>
// TODO: Mul + Recip を組み合わせた実装に変更する
impl<T> ops::Div<&Variable<T>> for &Variable<T>
where
    T: ops::Div<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;

    fn div(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_val = self.value();
        let rhs_val = rhs.value();
        Variable::new(lhs_val / rhs_val)
    }
}

// ============================================================================
// 演算子の実装（値を消費）
// ============================================================================

impl<T> ops::Add<Variable<T>> for Variable<T>
where
    T: GradNode + ops::Add<T, Output = T> + 'static,
{
    type Output = Variable<T>;
    fn add(self, rhs: Variable<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<T> ops::Mul<Variable<T>> for Variable<T>
where
    T: GradNode + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + 'static,
{
    type Output = Variable<T>;
    fn mul(self, rhs: Variable<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T> ops::Neg for Variable<T>
where
    T: GradNode + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
{
    type Output = Variable<T>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T> ops::Sub<Variable<T>> for Variable<T>
where
    T: ops::Sub<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;
    fn sub(self, rhs: Variable<T>) -> Variable<T> {
        &self - &rhs
    }
}

impl<T> ops::Div<Variable<T>> for Variable<T>
where
    T: ops::Div<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;
    fn div(self, rhs: Variable<T>) -> Variable<T> {
        &self / &rhs
    }
}
