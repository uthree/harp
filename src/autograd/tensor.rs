//! Tensor API
//!
//! PyTorchライクな自動微分対応のTensor型を提供します。

use super::grad_fn::{
    AddBackward, AddConstBackward, Exp2Backward, GradFn, Log2Backward, MulBackward,
    MulConstBackward, NegBackward, PadBackward, RecipBackward, ReduceSumBackward, SinBackward,
    SliceBackward, SqrtBackward,
};
use crate::graph::{GraphNode, ops::ElementwiseOp, ops::GraphOp};
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

    // === 基本数学関数 ===

    /// 底が2の対数: log2(x)
    pub fn log2(&self) -> Tensor {
        let dtype = self.data.dtype.clone();
        let view = self.data.view.clone();
        let result = GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Log2,
                elementwise_strategies: None,
            },
            vec![self.data.clone()],
            view,
        );
        Tensor::from_forward(result, vec![self.clone()], Log2Backward)
    }

    /// 2の累乗: 2^x
    pub fn exp2(&self) -> Tensor {
        let dtype = self.data.dtype.clone();
        let view = self.data.view.clone();
        let result = GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Exp2,
                elementwise_strategies: None,
            },
            vec![self.data.clone()],
            view,
        );
        Tensor::from_forward(result, vec![self.clone()], Exp2Backward)
    }

    /// 自然対数: ln(x)
    ///
    /// log2を使って実装: log(x) = log2(x) / log2(e)
    pub fn log(&self) -> Tensor {
        const INV_LOG2_E: f32 = 1.0 / std::f32::consts::LOG2_E;
        &self.log2() * INV_LOG2_E
    }

    /// 指数関数: e^x
    ///
    /// exp2を使って実装: exp(x) = 2^(x * log2(e))
    pub fn exp(&self) -> Tensor {
        const LOG2_E: f32 = std::f32::consts::LOG2_E;
        (self * LOG2_E).exp2()
    }

    /// 正弦: sin(x)
    pub fn sin(&self) -> Tensor {
        let dtype = self.data.dtype.clone();
        let view = self.data.view.clone();
        let result = GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Sin,
                elementwise_strategies: None,
            },
            vec![self.data.clone()],
            view,
        );
        Tensor::from_forward(result, vec![self.clone()], SinBackward)
    }

    /// 余弦: cos(x) = sin(x + π/2)
    pub fn cos(&self) -> Tensor {
        const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;
        (self + HALF_PI).sin()
    }

    /// 平方根: sqrt(x)
    pub fn sqrt(&self) -> Tensor {
        let dtype = self.data.dtype.clone();
        let view = self.data.view.clone();
        let result = GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Sqrt,
                elementwise_strategies: None,
            },
            vec![self.data.clone()],
            view,
        );
        Tensor::from_forward(result, vec![self.clone()], SqrtBackward)
    }

    /// 平方根の逆数: rsqrt(x) = 1/sqrt(x)
    pub fn rsqrt(&self) -> Tensor {
        self.sqrt().recip()
    }

    // === 高レベル演算 ===

    /// 二乗: x^2
    pub fn square(&self) -> Tensor {
        self * self
    }

    /// 累乗: x^n (正の整数のみ)
    pub fn powi(&self, n: u32) -> Tensor {
        assert!(n > 0, "powi: n must be positive");

        if n == 1 {
            return self.clone();
        }

        let mut result = self.clone();
        for _ in 1..n {
            result = &result * self;
        }
        result
    }

    /// 絶対値の二乗: x^2 (常に非負)
    pub fn abs_square(&self) -> Tensor {
        self.square()
    }

    /// 2つのテンソルの要素ごとの最小値: min(a, b) = -max(-a, -b)
    pub fn min(&self, other: &Tensor) -> Tensor {
        let neg_self = -self;
        let neg_other = -other;
        -neg_self.max(&neg_other)
    }

    /// クランプ: min_val <= x <= max_val に制限
    pub fn clamp(&self, min_val: &Tensor, max_val: &Tensor) -> Tensor {
        self.max(min_val).min(max_val)
    }

    /// 平均を計算: mean(x, axis)
    ///
    /// 指定された軸に沿った平均を計算します。
    pub fn mean(&self, axis: usize) -> Tensor {
        use crate::graph::shape::Expr;

        let shape = self.data.view.shape();
        if axis >= shape.len() {
            panic!("mean: axis {} is out of bounds for shape {:?}", axis, shape);
        }

        // 軸のサイズを取得
        let axis_size = &shape[axis];

        // 軸サイズが定数の場合のみ処理
        let size_value = match axis_size {
            Expr::Const(n) => *n as f32,
            _ => panic!(
                "mean: axis size must be constant, got symbolic expression: {:?}",
                axis_size
            ),
        };

        // 合計を計算し、サイズで割る
        &self.sum(axis) * (1.0f32 / size_value)
    }

    /// 分散を計算: var(x, axis)
    ///
    /// 不偏分散を計算します: E[(x - mean(x))^2]
    pub fn variance(&self, axis: usize) -> Tensor {
        // 平均を計算
        let x_mean = self.mean(axis);

        // meanの次元を復元してbroadcast可能にする
        let x_mean_view = x_mean
            .data
            .view
            .clone()
            .unsqueeze(axis)
            .expand(self.data.view.shape().to_vec());
        let x_mean_expanded = Tensor::from_graph_node(x_mean.data.view(x_mean_view), false);

        // (x - mean)^2 の平均を計算
        (self - &x_mean_expanded).square().mean(axis)
    }

    /// パディング
    pub fn pad(&self, padding: Vec<(usize, usize)>, value: f32) -> Tensor {
        let result = self.data.pad(padding.clone(), value);
        Tensor::from_forward(result, vec![self.clone()], PadBackward { padding })
    }

    /// スライス
    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Tensor {
        let result = self.data.slice(ranges.clone());
        Tensor::from_forward(result, vec![self.clone()], SliceBackward { ranges })
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
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self * rhs.recip()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
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

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<Tensor> for f32 {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self * rhs.recip()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
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
