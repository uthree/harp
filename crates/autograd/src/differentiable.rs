use std::ops;
use std::sync::{Arc, Mutex};

use crate::traits::{GradFn, GradNode, GradRoot};

// ============================================================================
// Variable (統合された変数型)
// ============================================================================

/// 変数の内部データ
struct DifferentiableInner<T: 'static> {
    value: T,
    grad: Option<Differentiable<T>>,
    grad_fn: Option<Box<dyn GradFn<Differentiable<T>>>>,
    requires_grad: bool,
}

/// 変数（リーフまたは計算結果）
pub struct Differentiable<T: 'static>(Arc<Mutex<DifferentiableInner<T>>>);

impl<T: 'static> Clone for Differentiable<T> {
    fn clone(&self) -> Self {
        Differentiable(Arc::clone(&self.0))
    }
}

impl<T: 'static> Differentiable<T> {
    /// 新しいリーフ変数を作成（requires_grad = true）
    pub fn new(value: T) -> Differentiable<T> {
        Differentiable(Arc::new(Mutex::new(DifferentiableInner {
            value,
            grad: None,
            grad_fn: None,
            requires_grad: true,
        })))
    }

    /// 新しいリーフ変数を作成（requires_grad = false）
    /// 高階微分を行わない場合に使用
    pub fn new_no_grad(value: T) -> Differentiable<T> {
        Differentiable(Arc::new(Mutex::new(DifferentiableInner {
            value,
            grad: None,
            grad_fn: None,
            requires_grad: false,
        })))
    }

    /// 新しいリーフ変数を作成（requires_grad を指定）
    pub fn new_with_requires_grad(value: T, requires_grad: bool) -> Differentiable<T> {
        Differentiable(Arc::new(Mutex::new(DifferentiableInner {
            value,
            grad: None,
            grad_fn: None,
            requires_grad,
        })))
    }

    /// 勾配関数付きの変数を作成（演算結果用、requires_grad = true）
    pub fn with_grad_fn(
        value: T,
        grad_fn: Box<dyn GradFn<Differentiable<T>>>,
    ) -> Differentiable<T> {
        Differentiable(Arc::new(Mutex::new(DifferentiableInner {
            value,
            grad: None,
            grad_fn: Some(grad_fn),
            requires_grad: true,
        })))
    }

    /// requires_grad の値を取得
    pub fn requires_grad(&self) -> bool {
        let inner = self.0.lock().unwrap();
        inner.requires_grad
    }

    /// requires_grad の値を設定
    pub fn set_requires_grad(&self, requires_grad: bool) {
        let mut inner = self.0.lock().unwrap();
        inner.requires_grad = requires_grad;
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

    /// grad_fn を取り除いて勾配の伝播を遮断
    pub fn detach(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.grad_fn = None;
    }
}

impl<T: Clone + 'static> Differentiable<T> {
    /// 値のコピーを取得
    pub fn value(&self) -> T {
        let inner = self.0.lock().unwrap();
        inner.value.clone()
    }
}

// T: GradNode の場合の実装
impl<T> Differentiable<T>
where
    T: GradNode + ops::Add<T, Output = T> + 'static,
{
    /// 累積された勾配を取得
    pub fn grad(&self) -> Option<Differentiable<T>> {
        let inner = self.0.lock().unwrap();
        inner.grad.clone()
    }

    /// 勾配をリセット
    pub fn zero_grad(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.grad = None;
    }

    /// 勾配を伝播
    pub fn backward_with(&self, grad: Differentiable<T>) {
        let mut inner = self.0.lock().unwrap();

        // requires_grad が true の場合のみ自身の勾配を累積する
        if inner.requires_grad {
            inner.grad = Some(if let Some(existing) = inner.grad.take() {
                let existing_val = existing.value();
                let grad_val = grad.value();
                Differentiable::new(existing_val + grad_val)
            } else {
                grad.clone()
            });
        }

        // 勾配を伝播する（grad_fn があれば）
        if let Some(mut grad_fn) = inner.grad_fn.take() {
            // ロックを解放してから backward を呼ぶ（デッドロック防止）
            drop(inner);
            grad_fn.backward(grad);
        }
    }
}

// T: GradRoot の場合の追加実装
impl<T> Differentiable<T>
where
    T: GradRoot + ops::Add<T, Output = T> + 'static,
{
    /// 初期勾配 1 で逆伝播を開始（高階微分なし）
    pub fn backward(&self) {
        // 高階微分を行わないため requires_grad = false で作成
        self.backward_with(Differentiable::new_no_grad(T::unit_grad()));
    }
}
