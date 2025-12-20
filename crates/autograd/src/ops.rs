use std::ops;

use crate::grad_fns::{Add, Mul, Neg};
use crate::traits::{GradNode, GradientInto};
use crate::variable::Variable;

// ============================================================================
// 演算子の実装（参照）
// ============================================================================

// &Variable<L> + &Variable<R> -> Variable<O>
impl<L, R, O> ops::Add<&Variable<R>> for &Variable<L>
where
    L: GradNode + ops::Add<L, Output = L> + ops::Add<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + 'static,
    O: Clone + 'static,
    Variable<O>: GradientInto<Variable<L>> + GradientInto<Variable<R>> + Clone,
{
    type Output = Variable<O>;

    fn add(self, rhs: &Variable<R>) -> Self::Output {
        let lhs_val = self.value();
        let rhs_val = rhs.value();
        Variable::with_grad_fn(
            lhs_val + rhs_val,
            Box::new(Add::new(self.clone(), rhs.clone())),
        )
    }
}

// &Variable<L> * &Variable<R> -> Variable<O>
impl<L, R, O> ops::Mul<&Variable<R>> for &Variable<L>
where
    L: GradNode + ops::Add<L, Output = L> + ops::Mul<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + 'static,
    O: Clone + ops::Mul<R, Output = L> + ops::Mul<L, Output = R> + 'static,
{
    type Output = Variable<O>;

    fn mul(self, rhs: &Variable<R>) -> Self::Output {
        let lhs_val = self.value();
        let rhs_val = rhs.value();
        Variable::with_grad_fn(
            lhs_val * rhs_val,
            Box::new(Mul::new(self.clone(), rhs.clone())),
        )
    }
}

// -&Variable<I> -> Variable<O>
impl<I, O> ops::Neg for &Variable<I>
where
    I: GradNode + ops::Add<I, Output = I> + ops::Neg<Output = O> + 'static,
    O: Clone + ops::Neg<Output = I> + 'static,
{
    type Output = Variable<O>;

    fn neg(self) -> Self::Output {
        let val = self.value();
        Variable::with_grad_fn(-val, Box::new(Neg::new(self.clone())))
    }
}

// &Variable<L> - &Variable<R> -> Variable<O>
// Neg + Add を組み合わせた実装
impl<L, R, O> ops::Sub<&Variable<R>> for &Variable<L>
where
    L: GradNode + ops::Add<L, Output = L> + ops::Add<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + ops::Neg<Output = R> + 'static,
    O: Clone + 'static,
    Variable<O>: GradientInto<Variable<L>> + GradientInto<Variable<R>> + Clone,
    Variable<R>: Clone,
    for<'a> &'a Variable<R>: ops::Neg<Output = Variable<R>>,
{
    type Output = Variable<O>;

    fn sub(self, rhs: &Variable<R>) -> Variable<O> {
        let neg_rhs = -rhs;
        self + &neg_rhs
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

impl<L, R, O> ops::Add<Variable<R>> for Variable<L>
where
    L: GradNode + ops::Add<L, Output = L> + ops::Add<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + 'static,
    O: Clone + 'static,
    Variable<O>: GradientInto<Variable<L>> + GradientInto<Variable<R>> + Clone,
{
    type Output = Variable<O>;
    fn add(self, rhs: Variable<R>) -> Self::Output {
        &self + &rhs
    }
}

impl<L, R, O> ops::Mul<Variable<R>> for Variable<L>
where
    L: GradNode + ops::Add<L, Output = L> + ops::Mul<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + 'static,
    O: Clone + ops::Mul<R, Output = L> + ops::Mul<L, Output = R> + 'static,
{
    type Output = Variable<O>;
    fn mul(self, rhs: Variable<R>) -> Self::Output {
        &self * &rhs
    }
}

impl<I, O> ops::Neg for Variable<I>
where
    I: GradNode + ops::Add<I, Output = I> + ops::Neg<Output = O> + 'static,
    O: Clone + ops::Neg<Output = I> + 'static,
{
    type Output = Variable<O>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<L, R, O> ops::Sub<Variable<R>> for Variable<L>
where
    L: GradNode + ops::Add<L, Output = L> + ops::Add<R, Output = O> + 'static,
    R: GradNode + ops::Add<R, Output = R> + ops::Neg<Output = R> + 'static,
    O: Clone + 'static,
    Variable<O>: GradientInto<Variable<L>> + GradientInto<Variable<R>> + Clone,
    Variable<R>: Clone,
    for<'a> &'a Variable<R>: ops::Neg<Output = Variable<R>>,
{
    type Output = Variable<O>;
    fn sub(self, rhs: Variable<R>) -> Variable<O> {
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
