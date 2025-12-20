use std::ops;

use crate::Variable;
use crate::arithmetic::{Add, Mul, Neg};

// ============================================================================
// 演算子の実装（参照）
// ============================================================================

// &Variable<T> + &Variable<T> -> Variable<T>
impl<T> ops::Add<&Variable<T>> for &Variable<T>
where
    T: ops::Add<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;

    fn add(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_val = self.0.lock().unwrap().value.clone();
        let rhs_val = rhs.0.lock().unwrap().value.clone();
        let result = Variable::new(lhs_val + rhs_val);
        result.set_grad_fn(Box::new(Add::new(self.clone(), rhs.clone())));
        result
    }
}

// &Variable<T> * &Variable<T> -> Variable<T>
impl<T> ops::Mul<&Variable<T>> for &Variable<T>
where
    T: ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;

    fn mul(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_val = self.0.lock().unwrap().value.clone();
        let rhs_val = rhs.0.lock().unwrap().value.clone();
        let result = Variable::new(lhs_val * rhs_val);
        result.set_grad_fn(Box::new(Mul::new(self.clone(), rhs.clone())));
        result
    }
}

// &Variable<T> / &Variable<T> -> Variable<T>
impl<T> ops::Div<&Variable<T>> for &Variable<T>
where
    T: ops::Div<T, Output = T> + Clone,
{
    type Output = Variable<T>;

    fn div(self, rhs: &Variable<T>) -> Variable<T> {
        // x / y = x * (1/y) として実装
        // TODO: Recip を使った実装に変更する
        let lhs_val = self.0.lock().unwrap().value.clone();
        let rhs_val = rhs.0.lock().unwrap().value.clone();
        Variable::new(lhs_val / rhs_val)
    }
}

// &Variable<T> - &Variable<T> -> Variable<T>
impl<T> ops::Sub<&Variable<T>> for &Variable<T>
where
    T: ops::Sub<T, Output = T> + Clone,
{
    type Output = Variable<T>;

    fn sub(self, rhs: &Variable<T>) -> Variable<T> {
        // x - y = x + (-y) として実装
        // TODO: Neg + Add を使った実装に変更する
        let lhs_val = self.0.lock().unwrap().value.clone();
        let rhs_val = rhs.0.lock().unwrap().value.clone();
        Variable::new(lhs_val - rhs_val)
    }
}

// -&Variable<T> -> Variable<T>
impl<T> ops::Neg for &Variable<T>
where
    T: ops::Add<T, Output = T> + ops::Neg<Output = T> + Clone + 'static,
{
    type Output = Variable<T>;

    fn neg(self) -> Variable<T> {
        let val = self.0.lock().unwrap().value.clone();
        let result = Variable::new(-val);
        result.set_grad_fn(Box::new(Neg::new(self.clone())));
        result
    }
}

// ============================================================================
// 演算子の実装（値を消費）
// ============================================================================

impl<T> ops::Add<Variable<T>> for Variable<T>
where
    T: ops::Add<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;
    fn add(self, rhs: Variable<T>) -> Variable<T> {
        &self + &rhs
    }
}

impl<T> ops::Mul<Variable<T>> for Variable<T>
where
    T: ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Clone + 'static,
{
    type Output = Variable<T>;
    fn mul(self, rhs: Variable<T>) -> Variable<T> {
        &self * &rhs
    }
}

impl<T> ops::Div<Variable<T>> for Variable<T>
where
    T: ops::Div<T, Output = T> + Clone,
{
    type Output = Variable<T>;
    fn div(self, rhs: Variable<T>) -> Variable<T> {
        &self / &rhs
    }
}

impl<T> ops::Sub<Variable<T>> for Variable<T>
where
    T: ops::Sub<T, Output = T> + Clone,
{
    type Output = Variable<T>;
    fn sub(self, rhs: Variable<T>) -> Variable<T> {
        &self - &rhs
    }
}

impl<T> ops::Neg for Variable<T>
where
    T: ops::Add<T, Output = T> + ops::Neg<Output = T> + Clone + 'static,
{
    type Output = Variable<T>;
    fn neg(self) -> Variable<T> {
        -&self
    }
}
