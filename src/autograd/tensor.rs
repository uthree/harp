//! Tensor API
//!
//! PyTorchライクな自動微分対応のTensor型を提供します。

use super::grad_fn::{
    AddBackward, AddConstBackward, GradFn, MulBackward, MulConstBackward, NegBackward,
    RecipBackward, ReduceSumBackward,
};
use crate::graph::GraphNode;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// 自動微分対応のTensor
///
/// GraphNodeをラップし、勾配計算機能を追加します。
#[derive(Clone)]
pub struct Tensor {
    /// 計算グラフのノード
    pub data: GraphNode,

    /// 勾配を計算するかどうか
    requires_grad: bool,

    /// 累積された勾配（backwardで計算される）
    grad: Rc<RefCell<Option<GraphNode>>>,

    /// backward時に使用する勾配計算関数
    grad_fn: Option<Rc<GradFnWrapper>>,
}

/// GradFnと入力テンソルをまとめて保持
#[derive(Clone)]
pub(super) struct GradFnWrapper {
    pub grad_fn: Rc<dyn GradFn>,
    pub inputs: Vec<Tensor>,
}

impl Tensor {
    /// GraphNodeから新しいTensorを作成
    ///
    /// # 引数
    /// - `data`: 計算グラフノード
    /// - `requires_grad`: 勾配を計算するかどうか
    pub fn from_graph_node(data: GraphNode, requires_grad: bool) -> Self {
        Self {
            data,
            requires_grad,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
        }
    }

    /// 勾配を計算するかどうかを取得
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// 勾配を取得
    ///
    /// backwardを実行した後に勾配が利用可能になります。
    pub fn grad(&self) -> Option<GraphNode> {
        self.grad.borrow().clone()
    }

    /// 勾配をゼロクリア
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// 逆伝播を実行
    ///
    /// スカラー（ndim=0）のテンソルに対してのみ呼び出し可能です。
    pub fn backward(&self) {
        assert_eq!(
            self.data.view.ndim(),
            0,
            "backward can only be called on scalar tensors"
        );

        // 勾配を1で初期化
        let grad_output = GraphNode::constant(1.0f32);

        // backward実行
        super::backward::backward(self, grad_output);
    }

    /// 前向き計算の結果から新しいTensorを作成（内部用）
    pub(super) fn from_forward(
        data: GraphNode,
        inputs: Vec<Tensor>,
        grad_fn: impl GradFn + 'static,
    ) -> Self {
        let requires_grad = inputs.iter().any(|t| t.requires_grad);

        let grad_fn_wrapper = if requires_grad {
            Some(Rc::new(GradFnWrapper {
                grad_fn: Rc::new(grad_fn),
                inputs,
            }))
        } else {
            None
        };

        Self {
            data,
            requires_grad,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: grad_fn_wrapper,
        }
    }

    /// grad_fnを取得（backward用）
    pub(super) fn grad_fn(&self) -> Option<&GradFnWrapper> {
        self.grad_fn.as_ref().map(|rc| rc.as_ref())
    }

    /// 勾配を累積（backward用）
    pub(super) fn accumulate_grad(&self, grad: GraphNode) {
        let mut grad_ref = self.grad.borrow_mut();
        *grad_ref = Some(if let Some(existing) = grad_ref.take() {
            existing + grad
        } else {
            grad
        });
    }

    // === Tensor演算メソッド ===

    /// 逆数を計算（1/x）
    pub fn recip(&self) -> Tensor {
        let result = self.data.clone().recip();
        Tensor::from_forward(result, vec![self.clone()], RecipBackward)
    }

    /// 指定軸の合計
    pub fn sum(&self, axis: usize) -> Tensor {
        let result = self.data.reduce_sum(axis);
        Tensor::from_forward(result, vec![self.clone()], ReduceSumBackward { axis })
    }

    /// 要素ごとの最大値
    pub fn max(&self, other: &Tensor) -> Tensor {
        let result = self.data.clone().max(other.data.clone());
        Tensor::from_forward(
            result,
            vec![self.clone(), other.clone()],
            super::grad_fn::MaxBackward,
        )
    }
}

// === 演算子オーバーロード ===

// Add: Tensor + Tensor
impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        let result = self.data.clone() + rhs.data.clone();
        Tensor::from_forward(result, vec![self, rhs], AddBackward)
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        let result = &self.data + &rhs.data;
        Tensor::from_forward(result, vec![self.clone(), rhs.clone()], AddBackward)
    }
}

// Add: Tensor + f32
impl Add<f32> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Tensor {
        let result = self.data.clone() + rhs;
        Tensor::from_forward(result, vec![self], AddConstBackward)
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Tensor {
        let result = &self.data + rhs;
        Tensor::from_forward(result, vec![self.clone()], AddConstBackward)
    }
}

// Add: f32 + Tensor
impl Add<Tensor> for f32 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        rhs + self
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        rhs + self
    }
}

// Mul: Tensor * Tensor
impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        let result = self.data.clone() * rhs.data.clone();
        Tensor::from_forward(result, vec![self, rhs], MulBackward)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        let result = &self.data * &rhs.data;
        Tensor::from_forward(result, vec![self.clone(), rhs.clone()], MulBackward)
    }
}

// Tensor * &Tensor
impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        let result = self.data.clone() * &rhs.data;
        Tensor::from_forward(result, vec![self, rhs.clone()], MulBackward)
    }
}

// Mul: Tensor * f32
impl Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        let result = self.data.clone() * rhs;
        Tensor::from_forward(result, vec![self], MulConstBackward { constant: rhs })
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        let result = &self.data * rhs;
        Tensor::from_forward(
            result,
            vec![self.clone()],
            MulConstBackward { constant: rhs },
        )
    }
}

// Mul: f32 * Tensor
impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        rhs * self
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        rhs * self
    }
}

// Neg: -Tensor
impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        let result = -self.data.clone();
        Tensor::from_forward(result, vec![self], NegBackward)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        let result = -&self.data;
        Tensor::from_forward(result, vec![self.clone()], NegBackward)
    }
}

// Sub: a - b = a + (-b)
impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self + (-rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self + &(-rhs)
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Tensor {
        self + (-rhs)
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Tensor {
        self + (-rhs)
    }
}

impl Sub<Tensor> for f32 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self + (-rhs)
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self + &(-rhs)
    }
}

// Div: a / b = a * recip(b)
impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self * rhs.recip()
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self * &rhs.recip()
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f32) -> Tensor {
        self * (1.0 / rhs)
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f32) -> Tensor {
        self * (1.0 / rhs)
    }
}

impl Div<Tensor> for f32 {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self * rhs.recip()
    }
}

impl Div<&Tensor> for f32 {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self * &rhs.recip()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.data.view.shape())
            .field("dtype", &self.data.dtype)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.borrow().is_some())
            .finish()
    }
}
