//! 型変換演算（Cast）
//!
//! RustのInto/Fromトレイトを活用した型変換を提供します。
//! 勾配も型変換して逆伝播します。

use std::marker::PhantomData;
use std::ops;

use crate::traits::GradFn;
use crate::variable::Variable;

// ============================================================================
// CastBackward (型変換の逆伝播)
// ============================================================================

/// 型変換の勾配関数
/// y: To = x.into() の場合:
/// - ∂L/∂x = ∂L/∂y.into() （勾配も型変換して伝播）
pub struct CastBackward<From, To>
where
    From: 'static,
    To: 'static,
{
    input: Variable<From>,
    _phantom: PhantomData<To>,
}

impl<From, To> CastBackward<From, To>
where
    From: 'static,
    To: 'static,
{
    pub fn new(input: Variable<From>) -> Self {
        Self {
            input,
            _phantom: PhantomData,
        }
    }
}

impl<From, To> GradFn<Variable<To>> for CastBackward<From, To>
where
    From: Clone + ops::Add<From, Output = From> + 'static,
    To: Clone + Into<From> + 'static,
{
    fn backward(&mut self, grad_y: Variable<To>) {
        // 勾配も型変換して逆伝播
        let grad_from: From = grad_y.value().into();
        self.input.backward_with(Variable::new(grad_from));
    }
}

// ============================================================================
// Variable<From> への Cast 実装
// ============================================================================

impl<From> Variable<From>
where
    From: Clone + 'static,
{
    /// 型変換を行う
    /// 双方向Into必須: From: Into<To>, To: Into<From>
    pub fn cast<To>(&self) -> Variable<To>
    where
        From: Into<To> + ops::Add<From, Output = From>,
        To: Clone + Into<From> + ops::Add<To, Output = To> + 'static,
    {
        let output: To = self.value().into();
        Variable::with_grad_fn(
            output,
            Box::new(CastBackward::<From, To>::new(self.clone())),
        )
    }
}

// ============================================================================
// f32 <-> f64 の相互変換実装
// ============================================================================

// 注: f32 -> f64 と f64 -> f32 は Rust 標準の From/Into で提供されていないため、
// 独自に実装が必要な場合があります。
// ただし、基本的な数値型については as キャストを使用することが多いため、
// 必要に応じてラッパー型やカスタム変換を実装してください。
