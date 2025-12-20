use num_traits::One;
use std::ops;
use std::sync::{Arc, Mutex};

// 内部データ構造
pub(crate) struct VariableInner<T> {
    pub(crate) value: T,
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Variable<T>>,
    pub(crate) grad_fn: Option<Box<dyn Backward<T>>>,
}

// 自動微分を適用する変数（Arc<Mutex<...>> のハンドル）
pub struct Variable<T>(pub(crate) Arc<Mutex<VariableInner<T>>>);

// Clone は Arc::clone のみ（軽量）
impl<T> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable(Arc::clone(&self.0))
    }
}

impl<T> Variable<T> {
    /// 新しい Variable を作成
    pub fn new(value: T) -> Variable<T> {
        Variable(Arc::new(Mutex::new(VariableInner {
            value,
            requires_grad: true,
            grad: None,
            grad_fn: None,
        })))
    }

    /// 値への参照を取得してクロージャを実行
    pub fn with_value<R, F: FnOnce(&T) -> R>(&self, f: F) -> R {
        let inner = self.0.lock().unwrap();
        f(&inner.value)
    }

    /// 値を変更
    pub fn with_value_mut<R, F: FnOnce(&mut T) -> R>(&self, f: F) -> R {
        let mut inner = self.0.lock().unwrap();
        f(&mut inner.value)
    }

    /// grad_fn を設定
    pub fn set_grad_fn(&self, grad_fn: Box<dyn Backward<T>>) {
        let mut inner = self.0.lock().unwrap();
        inner.grad_fn = Some(grad_fn);
    }

    /// grad_fn を None にして勾配の伝搬を遮断
    pub fn detach(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.grad_fn = None;
    }

    /// grad を None にして勾配を初期化
    pub fn zero_grad(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.grad = None;
    }

    /// 勾配を取得
    pub fn grad(&self) -> Option<Variable<T>> {
        let inner = self.0.lock().unwrap();
        inner.grad.clone()
    }
}

// T: Clone の場合、値のコピーを取得可能
impl<T: Clone> Variable<T> {
    /// 値のコピーを取得
    pub fn value(&self) -> T {
        let inner = self.0.lock().unwrap();
        inner.value.clone()
    }
}

// 逆伝播
impl<T> Variable<T>
where
    T: ops::Add<T, Output = T> + Clone + 'static,
{
    /// 指定した勾配で逆伝播を実行
    pub fn backward_with(&self, grad_y: Variable<T>) {
        let mut inner = self.0.lock().unwrap();

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

// 引数なしの逆伝播（初期勾配 = 1）
impl<T> Variable<T>
where
    T: ops::Add<T, Output = T> + Clone + One + 'static,
{
    /// 初期勾配 1 で逆伝播を実行
    pub fn backward(&self) {
        self.backward_with(Variable::new(T::one()));
    }
}

// 微分可能な関数を表現する
pub trait Backward<T> {
    fn backward(&mut self, grad_y: Variable<T>);
}
