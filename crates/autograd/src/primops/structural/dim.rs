//! 次元操作のプリミティブ演算（Squeeze, Unsqueeze）
//!
//! テンソルの次元を追加・削除する操作を提供します。
//! - Squeeze: size=1 の軸を削除
//! - Unsqueeze: 新しい size=1 の軸を挿入
//!
//! これらは互いに逆操作であり、Sum/Expand のペアと同様に実装されます。

use std::ops;

use crate::traits::GradFn;
use crate::variable::Variable;

// ============================================================================
// Squeeze トレイト
// ============================================================================

/// 次元を削除するトレイト
/// axis の位置にある size=1 の軸を削除する
pub trait Squeeze: Sized {
    type Output;
    fn squeeze(&self, axis: usize) -> Self::Output;
}

// ============================================================================
// Unsqueeze トレイト
// ============================================================================

/// 次元を追加するトレイト
/// axis の位置に size=1 の新しい軸を挿入する
pub trait Unsqueeze: Sized {
    type Output;
    fn unsqueeze(&self, axis: usize) -> Self::Output;
}

// ============================================================================
// スカラー型の実装（no-op）
// ============================================================================

impl Squeeze for f32 {
    type Output = Self;
    fn squeeze(&self, _axis: usize) -> Self::Output {
        *self
    }
}

impl Squeeze for f64 {
    type Output = Self;
    fn squeeze(&self, _axis: usize) -> Self::Output {
        *self
    }
}

impl Unsqueeze for f32 {
    type Output = Self;
    fn unsqueeze(&self, _axis: usize) -> Self::Output {
        *self
    }
}

impl Unsqueeze for f64 {
    type Output = Self;
    fn unsqueeze(&self, _axis: usize) -> Self::Output {
        *self
    }
}

// ============================================================================
// SqueezeBackward (次元削除の逆伝播)
// ============================================================================

/// Squeeze の勾配関数
/// y = squeeze(x, axis) の場合、∂L/∂x = unsqueeze(∂L/∂y, axis)
pub struct SqueezeBackward<T: 'static, U: 'static> {
    input: Variable<T>,
    axis: usize,
    _phantom: std::marker::PhantomData<U>,
}

impl<T: 'static, U: 'static> SqueezeBackward<T, U> {
    pub fn new(input: Variable<T>, axis: usize) -> Self {
        Self {
            input,
            axis,
            _phantom: std::marker::PhantomData,
        }
    }

    /// 削除した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl<T, U> GradFn<Variable<U>> for SqueezeBackward<T, U>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
    U: Clone + Unsqueeze<Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<U>) {
        // Squeeze の逆 = Unsqueeze
        let requires_grad = grad_y.requires_grad();
        let grad_x = grad_y.value().unsqueeze(self.axis);
        self.input
            .backward_with(Variable::new_with_requires_grad(grad_x, requires_grad));
    }
}

// ============================================================================
// UnsqueezeBackward (次元追加の逆伝播)
// ============================================================================

/// Unsqueeze の勾配関数
/// y = unsqueeze(x, axis) の場合、∂L/∂x = squeeze(∂L/∂y, axis)
pub struct UnsqueezeBackward<T: 'static, U: 'static> {
    input: Variable<T>,
    axis: usize,
    _phantom: std::marker::PhantomData<U>,
}

impl<T: 'static, U: 'static> UnsqueezeBackward<T, U> {
    pub fn new(input: Variable<T>, axis: usize) -> Self {
        Self {
            input,
            axis,
            _phantom: std::marker::PhantomData,
        }
    }

    /// 追加した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl<T, U> GradFn<Variable<U>> for UnsqueezeBackward<T, U>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
    U: Clone + Squeeze<Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<U>) {
        // Unsqueeze の逆 = Squeeze
        let requires_grad = grad_y.requires_grad();
        let grad_x = grad_y.value().squeeze(self.axis);
        self.input
            .backward_with(Variable::new_with_requires_grad(grad_x, requires_grad));
    }
}

// ============================================================================
// Variable<T> への Squeeze 実装
// ============================================================================

impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + Squeeze + 'static,
    <T as Squeeze>::Output: Clone
        + ops::Add<<T as Squeeze>::Output, Output = <T as Squeeze>::Output>
        + Unsqueeze<Output = T>
        + 'static,
{
    /// 指定した軸（size=1）を削除
    pub fn squeeze(&self, axis: usize) -> Variable<<T as Squeeze>::Output> {
        let output = Squeeze::squeeze(&self.value(), axis);
        if self.requires_grad() {
            Variable::with_grad_fn(
                output,
                Box::new(SqueezeBackward::<T, <T as Squeeze>::Output>::new(
                    self.clone(),
                    axis,
                )),
            )
        } else {
            Variable::new_no_grad(output)
        }
    }
}

// ============================================================================
// Variable<T> への Unsqueeze 実装
// ============================================================================

impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + Unsqueeze + 'static,
    <T as Unsqueeze>::Output: Clone
        + ops::Add<<T as Unsqueeze>::Output, Output = <T as Unsqueeze>::Output>
        + Squeeze<Output = T>
        + 'static,
{
    /// 指定した位置に新しい軸（size=1）を挿入
    pub fn unsqueeze(&self, axis: usize) -> Variable<<T as Unsqueeze>::Output> {
        let output = Unsqueeze::unsqueeze(&self.value(), axis);
        if self.requires_grad() {
            Variable::with_grad_fn(
                output,
                Box::new(UnsqueezeBackward::<T, <T as Unsqueeze>::Output>::new(
                    self.clone(),
                    axis,
                )),
            )
        } else {
            Variable::new_no_grad(output)
        }
    }
}
