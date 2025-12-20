use std::ops;
use std::sync::{Arc, RwLock};

// 内部データ構造
struct VariableInner<T> {
    value: T,
    requires_grad: bool,
    grad: Option<Variable<T>>,
    grad_fn: Option<Box<dyn Backward<T>>>,
}

// 自動微分を適用する変数（Arc<RwLock<...>> のハンドル）
pub struct Variable<T>(Arc<RwLock<VariableInner<T>>>);

// Clone は Arc::clone のみ（軽量）
impl<T> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable(Arc::clone(&self.0))
    }
}

impl<T> Variable<T> {
    /// 新しい Variable を作成
    pub fn new(value: T) -> Variable<T> {
        Variable(Arc::new(RwLock::new(VariableInner {
            value,
            requires_grad: true,
            grad: None,
            grad_fn: None,
        })))
    }

    /// 値への参照を取得してクロージャを実行
    pub fn with_value<R, F: FnOnce(&T) -> R>(&self, f: F) -> R {
        let inner = self.0.read().unwrap();
        f(&inner.value)
    }

    /// 値を変更
    pub fn with_value_mut<R, F: FnOnce(&mut T) -> R>(&self, f: F) -> R {
        let mut inner = self.0.write().unwrap();
        f(&mut inner.value)
    }

    /// grad_fn を設定
    pub fn set_grad_fn(&self, grad_fn: Box<dyn Backward<T>>) {
        let mut inner = self.0.write().unwrap();
        inner.grad_fn = Some(grad_fn);
    }

    /// grad_fn を None にして勾配の伝搬を遮断
    pub fn detach(&self) {
        let mut inner = self.0.write().unwrap();
        inner.grad_fn = None;
    }

    /// grad を None にして勾配を初期化
    pub fn zero_grad(&self) {
        let mut inner = self.0.write().unwrap();
        inner.grad = None;
    }

    /// 勾配を取得
    pub fn grad(&self) -> Option<Variable<T>> {
        let inner = self.0.read().unwrap();
        inner.grad.clone()
    }
}

// T: Clone の場合、値のコピーを取得可能
impl<T: Clone> Variable<T> {
    /// 値のコピーを取得
    pub fn value(&self) -> T {
        let inner = self.0.read().unwrap();
        inner.value.clone()
    }
}

// 逆伝播
impl<T> Variable<T>
where
    for<'a> &'a T: ops::Add<Output = T>,
{
    pub fn backward(&self, grad_y: Variable<T>) {
        let mut inner = self.0.write().unwrap();

        // 自身の勾配を累積する
        inner.grad = Some(if let Some(existing) = inner.grad.take() {
            // 既存の勾配と新しい勾配を加算
            &existing + &grad_y
        } else {
            grad_y.clone()
        });

        // 勾配を伝播する
        if let Some(mut grad_fn) = inner.grad_fn.take() {
            // ロックを解放してから backward を呼ぶ（デッドロック防止）
            drop(inner);
            grad_fn.backward(grad_y);
        }
    }
}

// 微分可能な関数を表現する
pub trait Backward<T> {
    fn backward(&mut self, grad_y: Variable<T>);
}

// ============================================================================
// 演算子の実装
// ============================================================================

// &Variable<T> + &Variable<T> -> Variable<T>
impl<T> ops::Add<&Variable<T>> for &Variable<T>
where
    for<'a> &'a T: ops::Add<Output = T>,
{
    type Output = Variable<T>;

    fn add(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_guard = self.0.read().unwrap();
        let rhs_guard = rhs.0.read().unwrap();
        Variable::new(&lhs_guard.value + &rhs_guard.value)
    }
}

// &Variable<T> * &Variable<T> -> Variable<T>
impl<T> ops::Mul<&Variable<T>> for &Variable<T>
where
    for<'a> &'a T: ops::Mul<Output = T>,
{
    type Output = Variable<T>;

    fn mul(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_guard = self.0.read().unwrap();
        let rhs_guard = rhs.0.read().unwrap();
        Variable::new(&lhs_guard.value * &rhs_guard.value)
    }
}

// &Variable<T> / &Variable<T> -> Variable<T>
impl<T> ops::Div<&Variable<T>> for &Variable<T>
where
    for<'a> &'a T: ops::Div<Output = T>,
{
    type Output = Variable<T>;

    fn div(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_guard = self.0.read().unwrap();
        let rhs_guard = rhs.0.read().unwrap();
        Variable::new(&lhs_guard.value / &rhs_guard.value)
    }
}

// &Variable<T> - &Variable<T> -> Variable<T>
impl<T> ops::Sub<&Variable<T>> for &Variable<T>
where
    for<'a> &'a T: ops::Sub<Output = T>,
{
    type Output = Variable<T>;

    fn sub(self, rhs: &Variable<T>) -> Variable<T> {
        let lhs_guard = self.0.read().unwrap();
        let rhs_guard = rhs.0.read().unwrap();
        Variable::new(&lhs_guard.value - &rhs_guard.value)
    }
}

// -&Variable<T> -> Variable<T>
impl<T> ops::Neg for &Variable<T>
where
    for<'a> &'a T: ops::Neg<Output = T>,
{
    type Output = Variable<T>;

    fn neg(self) -> Variable<T> {
        let guard = self.0.read().unwrap();
        Variable::new(-&guard.value)
    }
}

// ============================================================================
// 値を消費する演算子（利便性のため）
// ============================================================================

impl<T> ops::Add<Variable<T>> for Variable<T>
where
    for<'a> &'a T: ops::Add<Output = T>,
{
    type Output = Variable<T>;
    fn add(self, rhs: Variable<T>) -> Variable<T> {
        &self + &rhs
    }
}

impl<T> ops::Mul<Variable<T>> for Variable<T>
where
    for<'a> &'a T: ops::Mul<Output = T>,
{
    type Output = Variable<T>;
    fn mul(self, rhs: Variable<T>) -> Variable<T> {
        &self * &rhs
    }
}

impl<T> ops::Div<Variable<T>> for Variable<T>
where
    for<'a> &'a T: ops::Div<Output = T>,
{
    type Output = Variable<T>;
    fn div(self, rhs: Variable<T>) -> Variable<T> {
        &self / &rhs
    }
}

impl<T> ops::Sub<Variable<T>> for Variable<T>
where
    for<'a> &'a T: ops::Sub<Output = T>,
{
    type Output = Variable<T>;
    fn sub(self, rhs: Variable<T>) -> Variable<T> {
        &self - &rhs
    }
}

impl<T> ops::Neg for Variable<T>
where
    for<'a> &'a T: ops::Neg<Output = T>,
{
    type Output = Variable<T>;
    fn neg(self) -> Variable<T> {
        -&self
    }
}
