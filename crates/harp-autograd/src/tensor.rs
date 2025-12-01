//! Tensor API
//!
//! PyTorchライクな自動微分対応のTensor型を提供します。

use super::grad_fn::{
    AddBackward, AddConstBackward, Conv1dBackward, Conv2dBackward, Conv3dBackward,
    ConvTranspose1dBackward, ConvTranspose2dBackward, ConvTranspose3dBackward, Exp2Backward,
    GradFn, Log2Backward, MulBackward, MulConstBackward, NegBackward, PadBackward, RecipBackward,
    ReduceSumBackward, SinBackward, SliceBackward, SqrtBackward,
};
use harp::backend::Device;
use harp::graph::{Graph, GraphNode, ops::ElementwiseOp, ops::GraphOp};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// 自動微分対応のTensor
///
/// GraphNodeをラップし、勾配計算機能を追加します。
///
/// PyTorch風にデバイス情報を保持しますが、遅延評価を維持します。
/// realize()を呼ぶまで実際の計算は行われません。
#[derive(Clone)]
pub struct Tensor {
    /// 計算グラフのノード
    pub data: GraphNode,

    /// このTensorを実行する予定のデバイス
    device: Device,

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
    /// GraphNodeから新しいTensorを作成（デフォルトデバイス）
    ///
    /// # 引数
    /// - `data`: 計算グラフノード
    /// - `requires_grad`: 勾配を計算するかどうか
    pub fn from_graph_node(data: GraphNode, requires_grad: bool) -> Self {
        Self::from_graph_node_with_device(data, requires_grad, Device::default())
    }

    /// GraphNodeから新しいTensorを作成（デバイス指定版）
    ///
    /// # 引数
    /// - `data`: 計算グラフノード
    /// - `requires_grad`: 勾配を計算するかどうか
    /// - `device`: 使用するデバイス
    pub fn from_graph_node_with_device(
        data: GraphNode,
        requires_grad: bool,
        device: Device,
    ) -> Self {
        Self {
            data,
            device,
            requires_grad,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
        }
    }

    /// 指定した形状のゼロテンソルを作成（デフォルトデバイス）
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    ///
    /// let zeros = Tensor::zeros(vec![2, 3]); // 2x3のゼロ行列
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::full(shape, 0.0)
    }

    /// 指定した形状のゼロテンソルを作成（デバイス指定版）
    pub fn zeros_on(shape: Vec<usize>, device: Device) -> Self {
        Self::full_on(shape, 0.0, device)
    }

    /// 指定した形状の1で埋められたテンソルを作成（デフォルトデバイス）
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    ///
    /// let ones = Tensor::ones(vec![2, 3]); // 2x3の1行列
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::full(shape, 1.0)
    }

    /// 指定した形状の1で埋められたテンソルを作成（デバイス指定版）
    pub fn ones_on(shape: Vec<usize>, device: Device) -> Self {
        Self::full_on(shape, 1.0, device)
    }

    /// 指定した形状の一様乱数テンソルを作成（デフォルトデバイス）
    ///
    /// 各要素は[0, 1)の範囲の一様乱数で初期化されます。
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    ///
    /// let rand_tensor = Tensor::rand(vec![2, 3]); // 2x3の乱数テンソル
    /// ```
    pub fn rand(shape: Vec<usize>) -> Self {
        Self::rand_on(shape, Device::default())
    }

    /// 指定した形状の一様乱数テンソルを作成（デバイス指定版）
    pub fn rand_on(shape: Vec<usize>, device: Device) -> Self {
        let node = GraphNode::rand(shape);
        Self::from_graph_node_with_device(node, false, device)
    }

    /// 指定した形状の標準正規分布乱数テンソルを作成（デフォルトデバイス）
    ///
    /// 各要素は平均0、標準偏差1の正規分布に従う乱数で初期化されます。
    /// Box-Muller法を使用して一様乱数から正規乱数を生成します。
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    ///
    /// let randn_tensor = Tensor::randn(vec![2, 3]); // 2x3の正規乱数テンソル
    /// ```
    pub fn randn(shape: Vec<usize>) -> Self {
        Self::randn_on(shape, Device::default())
    }

    /// 指定した形状の標準正規分布乱数テンソルを作成（デバイス指定版）
    pub fn randn_on(shape: Vec<usize>, device: Device) -> Self {
        let node = GraphNode::randn(shape);
        Self::from_graph_node_with_device(node, false, device)
    }

    /// 指定した形状と値で埋められたテンソルを作成（デフォルトデバイス）
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    /// - `value`: 埋める値
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    ///
    /// let tensor = Tensor::full(vec![2, 3], 5.0); // 2x3の5で埋められた行列
    /// ```
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        Self::full_on(shape, value, Device::default())
    }

    /// 指定した形状と値で埋められたテンソルを作成（デバイス指定版）
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    /// - `value`: 埋める値
    /// - `device`: 使用するデバイス
    pub fn full_on(shape: Vec<usize>, value: f32, device: Device) -> Self {
        use harp::graph::shape::Expr;

        // スカラー定数を作成
        let mut node = GraphNode::constant(value);

        // 各次元に対してunsqueezeしてから、expandで目的の形状にする
        if !shape.is_empty() {
            // まず全次元をunsqueezeして [1, 1, ..., 1] の形状にする
            let mut view = node.view.clone();
            for _ in 0..shape.len() {
                view = view.unsqueeze(0);
            }
            node = node.view(view);

            // 次にexpandで目的の形状にブロードキャスト
            let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as isize)).collect();
            node = node.expand(shape_exprs);
        }

        Self::from_graph_node_with_device(node, false, device)
    }

    /// 勾配を計算するかどうかを取得
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// 勾配グラフから切り離した新しいTensorを作成
    ///
    /// 返されるTensorは `requires_grad = false` となり、
    /// backward時にこのTensorより先に勾配が伝播されなくなります。
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    /// use harp::graph::GraphNode;
    ///
    /// let a = Tensor::from_graph_node(GraphNode::constant(1.0f32), true);
    /// assert!(a.requires_grad());
    /// let b = a.detach(); // 勾配グラフから切り離し
    /// assert!(!b.requires_grad());
    /// ```
    pub fn detach(&self) -> Self {
        Self {
            data: self.data.clone(),
            device: self.device,
            requires_grad: false,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
        }
    }

    /// このTensorが使用するデバイスを取得
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    /// use harp::backend::Device;
    ///
    /// let tensor = Tensor::zeros(vec![2, 3]);
    /// println!("Device: {}", tensor.device());
    /// ```
    pub fn device(&self) -> Device {
        self.device
    }

    /// デバイスを変更した新しいTensorを作成
    ///
    /// 計算グラフは共有され、デバイス情報のみが変更されます。
    /// 遅延評価のため、実際のデータ転送は行われません。
    ///
    /// # 引数
    /// - `device`: 新しいデバイス
    ///
    /// # 例
    /// ```
    /// use harp_autograd::Tensor;
    /// use harp::backend::Device;
    ///
    /// let tensor = Tensor::zeros(vec![2, 3]);
    /// // 利用可能な最適なデバイスを自動選択
    /// let tensor_gpu = tensor.to(Device::auto_select());
    /// ```
    pub fn to(&self, device: Device) -> Self {
        if self.device == device {
            // 同じデバイスなら自分自身をクローン
            self.clone()
        } else {
            // デバイス情報のみ変更
            Self {
                data: self.data.clone(),
                device,
                requires_grad: self.requires_grad,
                grad: self.grad.clone(),
                grad_fn: self.grad_fn.clone(),
            }
        }
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
    ///
    /// デバイス情報は入力テンソルから引き継がれます。
    /// 複数の入力がある場合は、最初のテンソルのデバイスを使用します。
    pub(super) fn from_forward(
        data: GraphNode,
        inputs: Vec<Tensor>,
        grad_fn: impl GradFn + 'static,
    ) -> Self {
        let requires_grad = inputs.iter().any(|t| t.requires_grad);

        // 入力テンソルからデバイスを取得（最初のテンソルのデバイスを使用）
        let device = inputs.first().map(|t| t.device).unwrap_or_default();

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
            device,
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
            },
            vec![self.data.clone()],
            view,
        );
        Tensor::from_forward(result, vec![self.clone()], Exp2Backward)
    }

    /// 正弦: sin(x)
    pub fn sin(&self) -> Tensor {
        let dtype = self.data.dtype.clone();
        let view = self.data.view.clone();
        let result = GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Sin,
            },
            vec![self.data.clone()],
            view,
        );
        Tensor::from_forward(result, vec![self.clone()], SinBackward)
    }

    /// 平方根: sqrt(x)
    pub fn sqrt(&self) -> Tensor {
        let dtype = self.data.dtype.clone();
        let view = self.data.view.clone();
        let result = GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Sqrt,
            },
            vec![self.data.clone()],
            view,
        );
        Tensor::from_forward(result, vec![self.clone()], SqrtBackward)
    }

    // === パディング・スライス演算 ===

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

    // === 畳み込み演算 ===

    /// 1D畳み込み
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, k)
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数（1=通常、C_in=depthwise）
    ///
    /// # 入出力
    /// - 入力: (C_in, L)
    /// - カーネル: (C_out, C_in/groups, k)
    /// - 出力: (C_out, L')
    ///
    /// # 注意
    /// 現在、勾配計算は未実装です。backward()を呼ぶとpanicします。
    /// fold/col2im演算の実装が必要です。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![2, 5]); // (C_in=2, L=5)
    /// let kernel = Tensor::ones(vec![3, 2, 3]); // (C_out=3, C_in=2, k=3)
    /// let output = x.conv1d(&kernel, 1, 1, 1); // (3, 3)
    /// ```
    pub fn conv1d(&self, kernel: &Tensor, stride: usize, dilation: usize, groups: usize) -> Tensor {
        let result = self
            .data
            .clone()
            .conv1d(kernel.data.clone(), stride, dilation, groups);
        Tensor::from_forward(
            result,
            vec![self.clone(), kernel.clone()],
            Conv1dBackward {
                stride,
                dilation,
                groups,
            },
        )
    }

    /// 2D畳み込み
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, kH, kW)
    /// - `stride`: ストライド (sH, sW)
    /// - `dilation`: 膨張率 (dH, dW)
    /// - `groups`: グループ数（1=通常、C_in=depthwise）
    ///
    /// # 入出力
    /// - 入力: (C_in, H, W)
    /// - カーネル: (C_out, C_in/groups, kH, kW)
    /// - 出力: (C_out, H', W')
    ///
    /// # 注意
    /// 現在、勾配計算は未実装です。backward()を呼ぶとpanicします。
    /// fold/col2im演算の実装が必要です。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![3, 32, 32]); // (C_in=3, H=32, W=32)
    /// let kernel = Tensor::ones(vec![16, 3, 3, 3]); // (C_out=16, C_in=3, kH=3, kW=3)
    /// let output = x.conv2d(&kernel, (1, 1), (1, 1), 1); // (16, 30, 30)
    /// ```
    pub fn conv2d(
        &self,
        kernel: &Tensor,
        stride: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Tensor {
        let result = self
            .data
            .clone()
            .conv2d(kernel.data.clone(), stride, dilation, groups);
        Tensor::from_forward(
            result,
            vec![self.clone(), kernel.clone()],
            Conv2dBackward {
                stride,
                dilation,
                groups,
            },
        )
    }

    /// 3D畳み込み
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, kD, kH, kW)
    /// - `stride`: ストライド (sD, sH, sW)
    /// - `dilation`: 膨張率 (dD, dH, dW)
    /// - `groups`: グループ数（1=通常、C_in=depthwise）
    ///
    /// # 入出力
    /// - 入力: (C_in, D, H, W)
    /// - カーネル: (C_out, C_in/groups, kD, kH, kW)
    /// - 出力: (C_out, D', H', W')
    ///
    /// # 注意
    /// 現在、勾配計算は未実装です。backward()を呼ぶとpanicします。
    /// fold/col2im演算の実装が必要です。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![2, 16, 16, 16]); // (C_in=2, D=16, H=16, W=16)
    /// let kernel = Tensor::ones(vec![8, 2, 3, 3, 3]); // (C_out=8, C_in=2, kD=3, kH=3, kW=3)
    /// let output = x.conv3d(&kernel, (1, 1, 1), (1, 1, 1), 1); // (8, 14, 14, 14)
    /// ```
    pub fn conv3d(
        &self,
        kernel: &Tensor,
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Tensor {
        let result = self
            .data
            .clone()
            .conv3d(kernel.data.clone(), stride, dilation, groups);
        Tensor::from_forward(
            result,
            vec![self.clone(), kernel.clone()],
            Conv3dBackward {
                stride,
                dilation,
                groups,
            },
        )
    }

    /// 1D転置畳み込み（deconvolution / transposed convolution）
    ///
    /// 畳み込みの逆操作を行います。主にアップサンプリングに使用されます。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, k)
    /// - `stride`: ストライド
    /// - `padding`: パディング - 出力から削られるサイズ
    /// - `output_padding`: 出力パディング - 出力に追加されるサイズ
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: (C_in, L_in)
    /// - カーネル: (C_in, C_out/groups, k)
    /// - 出力: (C_out, L_out)
    ///   - L_out = (L_in - 1) * s - 2 * p + d * (k - 1) + op + 1
    pub fn conv_transpose1d(
        &self,
        kernel: &Tensor,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Tensor {
        let result = self.data.clone().conv_transpose1d(
            kernel.data.clone(),
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        );
        Tensor::from_forward(
            result,
            vec![self.clone(), kernel.clone()],
            ConvTranspose1dBackward {
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            },
        )
    }

    /// 2D転置畳み込み（deconvolution / transposed convolution）
    ///
    /// 畳み込みの逆操作を行います。主にアップサンプリングに使用されます。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, kH, kW)
    /// - `stride`: ストライド (sH, sW)
    /// - `padding`: パディング (pH, pW) - 出力から削られるサイズ
    /// - `output_padding`: 出力パディング (opH, opW) - 出力に追加されるサイズ
    /// - `dilation`: 膨張率 (dH, dW)
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: (C_in, H_in, W_in)
    /// - カーネル: (C_in, C_out/groups, kH, kW)
    /// - 出力: (C_out, H_out, W_out)
    ///   - H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
    ///   - W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![16, 4, 4]); // (C_in=16, H=4, W=4)
    /// let kernel = Tensor::ones(vec![16, 3, 3, 3]); // (C_in=16, C_out=3, kH=3, kW=3)
    /// let output = x.conv_transpose2d(&kernel, (2, 2), (0, 0), (0, 0), (1, 1), 1); // (3, 9, 9)
    /// ```
    pub fn conv_transpose2d(
        &self,
        kernel: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Tensor {
        let result = self.data.clone().conv_transpose2d(
            kernel.data.clone(),
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        );
        Tensor::from_forward(
            result,
            vec![self.clone(), kernel.clone()],
            ConvTranspose2dBackward {
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            },
        )
    }

    /// 3D転置畳み込み（deconvolution / transposed convolution）
    ///
    /// 畳み込みの逆操作を行います。主にアップサンプリングに使用されます。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, kD, kH, kW)
    /// - `stride`: ストライド (sD, sH, sW)
    /// - `padding`: パディング (pD, pH, pW) - 出力から削られるサイズ
    /// - `output_padding`: 出力パディング (opD, opH, opW) - 出力に追加されるサイズ
    /// - `dilation`: 膨張率 (dD, dH, dW)
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: (C_in, D_in, H_in, W_in)
    /// - カーネル: (C_in, C_out/groups, kD, kH, kW)
    /// - 出力: (C_out, D_out, H_out, W_out)
    pub fn conv_transpose3d(
        &self,
        kernel: &Tensor,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Tensor {
        let result = self.data.clone().conv_transpose3d(
            kernel.data.clone(),
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        );
        Tensor::from_forward(
            result,
            vec![self.clone(), kernel.clone()],
            ConvTranspose3dBackward {
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            },
        )
    }

    // === 実行（realize）メソッド ===

    /// このTensorをこのTensorが持つデバイス上で実行（tinygradスタイル）
    ///
    /// 入力データを提供してこのTensorの計算を実行します。
    /// このTensorが持つデバイス情報を使用します。
    ///
    /// # 引数
    /// - `inputs`: 入力ノード名 -> データのマッピング
    ///
    /// # 戻り値
    /// 計算結果のバイト列
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    /// use harp::backend::Device;
    /// use std::collections::HashMap;
    ///
    /// // 利用可能な最適なデバイスで実行するTensorを作成
    /// let x = Tensor::ones(vec![3]).to(Device::auto_select());
    /// let y = &x * 2.0 + 1.0;
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("x".to_string(), vec![1.0f32, 2.0, 3.0]);
    ///
    /// // yが持つデバイスで実行される
    /// let result = y.realize(inputs).unwrap();
    /// // result = [3.0, 5.0, 7.0]
    /// ```
    pub fn realize(&self, inputs: HashMap<String, Vec<f32>>) -> Result<Vec<f32>, String> {
        // このTensorが持つデバイスを使用
        self.realize_on(inputs, self.device)
    }

    /// このTensorを指定されたデバイス上で実行
    ///
    /// # 引数
    /// - `inputs`: 入力ノード名 -> データのマッピング
    /// - `device`: 使用するデバイス
    pub fn realize_on(
        &self,
        inputs: HashMap<String, Vec<f32>>,
        device: Device,
    ) -> Result<Vec<f32>, String> {
        // 1. このTensorを出力とするGraphを構築
        let mut graph = Graph::new();

        // 入力ノードを登録（GraphNodeから逆算する必要がある）
        // 簡略化のため、入力データから推論
        for (name, data) in &inputs {
            let shape = vec![data.len()];
            let _input_node = graph.input(name, harp::graph::DType::F32, shape);
        }

        // 出力ノードを登録
        graph.output("result", self.data.clone());

        // 2. Graphのrealize_with_deviceを使用
        let mut results = graph.realize_with_device(inputs, Some(device))?;

        // 3. "result"の出力を取得
        results
            .remove("result")
            .ok_or_else(|| "Result not found in outputs".to_string())
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
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.borrow().is_some())
            .finish()
    }
}

// ndarray feature が有効な場合の変換機能
#[cfg(feature = "ndarray")]
impl Tensor {
    /// ndarrayから形状情報を使ってゼロテンソルを作成
    ///
    /// 注意: この関数は形状のみをコピーし、データはコピーしません。
    /// データを含むテンソルを作成する場合は、realize()を使用して
    /// データを渡す必要があります。
    ///
    /// # 引数
    /// - `array`: 参照するndarray
    ///
    /// # 例
    /// ```ignore
    /// use ndarray::Array2;
    /// use harp_autograd::Tensor;
    ///
    /// let array = Array2::<f32>::zeros((2, 3));
    /// let tensor = Tensor::from_ndarray_shape(&array.into_dyn());
    /// // tensorは2x3のゼロテンソルとして作成される
    /// ```
    pub fn from_ndarray_shape(array: &ndarray::ArrayD<f32>) -> Self {
        let shape: Vec<usize> = array.shape().to_vec();
        Self::zeros(shape)
    }

    /// ndarrayのデータを使ってテンソルを作成
    ///
    /// 注意: 現在の実装では、小さな配列のみを推奨します。
    /// 大きな配列の場合、計算グラフが肥大化する可能性があります。
    ///
    /// # 引数
    /// - `array`: ndarrayの配列
    ///
    /// # 例
    /// ```ignore
    /// use ndarray::{Array2, arr2};
    /// use harp_autograd::Tensor;
    ///
    /// let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let tensor = Tensor::from_ndarray(array.into_dyn());
    /// ```
    pub fn from_ndarray(array: ndarray::ArrayD<f32>) -> Self {
        // 現在の実装では形状のみを使用
        // TODO: 将来的にはデータを計算グラフに埋め込む実装を検討
        Self::from_ndarray_shape(&array)
    }
}

/// ndarrayからTensorへの変換
///
/// 注意: 現在の実装では、形状のみをコピーします。
/// データは含まれません（ゼロ初期化されます）。
#[cfg(feature = "ndarray")]
impl From<ndarray::ArrayD<f32>> for Tensor {
    fn from(array: ndarray::ArrayD<f32>) -> Self {
        Self::from_ndarray(array)
    }
}

/// ndarray::Array1からの変換
#[cfg(feature = "ndarray")]
impl From<ndarray::Array1<f32>> for Tensor {
    fn from(array: ndarray::Array1<f32>) -> Self {
        Self::from_ndarray(array.into_dyn())
    }
}

/// ndarray::Array2からの変換
#[cfg(feature = "ndarray")]
impl From<ndarray::Array2<f32>> for Tensor {
    fn from(array: ndarray::Array2<f32>) -> Self {
        Self::from_ndarray(array.into_dyn())
    }
}

/// ndarray::Array3からの変換
#[cfg(feature = "ndarray")]
impl From<ndarray::Array3<f32>> for Tensor {
    fn from(array: ndarray::Array3<f32>) -> Self {
        Self::from_ndarray(array.into_dyn())
    }
}

/// ndarray::Array4からの変換
#[cfg(feature = "ndarray")]
impl From<ndarray::Array4<f32>> for Tensor {
    fn from(array: ndarray::Array4<f32>) -> Self {
        Self::from_ndarray(array.into_dyn())
    }
}
