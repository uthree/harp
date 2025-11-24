//! 勾配計算関数
//!
//! 各演算に対する勾配計算の規則を定義します。
//! 設計方針: 演算の種類を最小限に抑え、複雑な演算は基本演算の組み合わせで表現します。

use super::tensor::Tensor;
use std::fmt::Debug;

/// 勾配計算関数のtrait
///
/// 各演算のbackward規則を実装します。
pub trait GradFn: Debug {
    /// 出力の勾配から入力の勾配を計算
    ///
    /// # 引数
    /// - `grad_output`: 出力に対する勾配
    /// - `inputs`: 前向き計算時の入力テンソル
    ///
    /// # 戻り値
    /// 各入力に対する勾配（requires_grad=falseの入力にはNoneを返す）
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>>;
}

// === 基本演算の勾配関数 ===

/// Add演算の勾配: ∂L/∂a = ∂L/∂out, ∂L/∂b = ∂L/∂out
#[derive(Debug)]
pub struct AddBackward;

impl GradFn for AddBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Add requires 2 inputs");
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// Mul演算の勾配: ∂L/∂a = ∂L/∂out * b, ∂L/∂b = ∂L/∂out * a
#[derive(Debug)]
pub struct MulBackward;

impl GradFn for MulBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Mul requires 2 inputs");
        let a = &inputs[0];
        let b = &inputs[1];
        vec![Some(grad_output * b), Some(grad_output * a)]
    }
}

/// Neg演算の勾配: ∂L/∂a = -∂L/∂out
#[derive(Debug)]
pub struct NegBackward;

impl GradFn for NegBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Neg requires 1 input");
        vec![Some(-grad_output)]
    }
}

/// Recip演算の勾配: ∂L/∂a = -∂L/∂out / (a²)
#[derive(Debug)]
pub struct RecipBackward;

impl GradFn for RecipBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Recip requires 1 input");
        let a = &inputs[0];
        // -grad_output / (a²) = -grad_output * recip(a²) = -grad_output * recip(a) * recip(a)
        let recip_a = a.recip();
        let grad = -grad_output * &recip_a;
        let grad = &grad * &recip_a;
        vec![Some(grad)]
    }
}

/// ReduceSum演算の勾配: ∂L/∂a = expand(∂L/∂out)
///
/// 縮約した軸を元のサイズに展開します。
#[derive(Debug)]
pub struct ReduceSumBackward {
    pub axis: usize,
}

impl GradFn for ReduceSumBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "ReduceSum requires 1 input");
        let input_shape = inputs[0].data.view.shape();

        // 縮約された軸を復元（サイズ1で追加）
        let grad_unsqueezed = grad_output.data.view.clone().unsqueeze(self.axis);

        // 元のshapeに展開
        let grad_expanded_view = grad_unsqueezed.expand(input_shape.to_vec());

        // Viewを適用したGraphNodeを作成
        let grad_expanded = grad_output.data.view(grad_expanded_view);

        vec![Some(Tensor::from_graph_node(grad_expanded, false))]
    }
}

/// Max演算（要素ごと）の勾配: ∂L/∂a = ∂L/∂out * (a >= b), ∂L/∂b = ∂L/∂out * (a < b)
///
/// 注意: 等しい場合の勾配は慣例的に最初の入力に流します
#[derive(Debug)]
pub struct MaxBackward;

impl GradFn for MaxBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Max requires 2 inputs");
        // TODO: 後で実装（比較演算が必要）
        // 現時点では簡易実装として両方に勾配を流す
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// 定数との加算の勾配: ∂L/∂a = ∂L/∂out （定数項は勾配なし）
#[derive(Debug)]
pub struct AddConstBackward;

impl GradFn for AddConstBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "AddConst requires 1 input");
        vec![Some(grad_output.clone())]
    }
}

/// 定数との乗算の勾配: ∂L/∂a = ∂L/∂out * c
#[derive(Debug)]
pub struct MulConstBackward {
    pub constant: f32,
}

impl GradFn for MulConstBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "MulConst requires 1 input");
        vec![Some(grad_output * self.constant)]
    }
}

// === 数学関数の勾配関数 ===

/// Log2演算の勾配: ∂L/∂x = ∂L/∂out / (x * ln(2))
#[derive(Debug)]
pub struct Log2Backward;

impl GradFn for Log2Backward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Log2 requires 1 input");
        let x = &inputs[0];
        // ∂log2(x)/∂x = 1 / (x * ln(2))
        const INV_LN2: f32 = 1.0 / std::f32::consts::LN_2;
        vec![Some(grad_output / x * INV_LN2)]
    }
}

/// Exp2演算の勾配: ∂L/∂x = ∂L/∂out * 2^x * ln(2)
#[derive(Debug)]
pub struct Exp2Backward;

impl GradFn for Exp2Backward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Exp2 requires 1 input");
        let x = &inputs[0];
        // ∂2^x/∂x = 2^x * ln(2)
        const LN2: f32 = std::f32::consts::LN_2;
        vec![Some(grad_output * &x.exp2() * LN2)]
    }
}

/// Sin演算の勾配: ∂L/∂x = ∂L/∂out * cos(x)
#[derive(Debug)]
pub struct SinBackward;

impl GradFn for SinBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Sin requires 1 input");
        let x = &inputs[0];
        // ∂sin(x)/∂x = cos(x)
        vec![Some(grad_output * &x.cos())]
    }
}

/// Sqrt演算の勾配: ∂L/∂x = ∂L/∂out / (2 * sqrt(x))
#[derive(Debug)]
pub struct SqrtBackward;

impl GradFn for SqrtBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Sqrt requires 1 input");
        let x = &inputs[0];
        // ∂sqrt(x)/∂x = 1 / (2 * sqrt(x)) = 0.5 * rsqrt(x) * recip(x)
        // rsqrt(x) = 1 / sqrt(x) なので、 0.5 / sqrt(x)
        vec![Some(grad_output / &x.sqrt() * 0.5)]
    }
}

/// Pad演算の勾配: パディング部分を除去して元のサイズに戻す
///
/// grad_output から padding 部分を除去するslice操作
#[derive(Debug)]
pub struct PadBackward {
    pub padding: Vec<(usize, usize)>,
}

impl GradFn for PadBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Pad requires 1 input");

        let input = &inputs[0];
        let input_shape = input.data.view.shape();

        // パディングを除去する範囲を計算
        // 各軸について: [padding_before, padding_before + original_size]
        let ranges: Vec<(usize, usize)> = self
            .padding
            .iter()
            .enumerate()
            .map(|(i, (before, _after))| {
                // input_shape[i]は式なので、定数の場合のみ処理
                // TODO: 式の評価が必要な場合は対応が必要
                let size = match &input_shape[i] {
                    crate::graph::shape::Expr::Const(val) => *val as usize,
                    _ => panic!("PadBackward requires constant input shape"),
                };
                (*before, before + size)
            })
            .collect();

        vec![Some(grad_output.slice(ranges))]
    }
}

/// Slice演算の勾配: 元のサイズのゼロテンソルを作り、slice部分に勾配を配置
///
/// Padの逆操作として実装
#[derive(Debug)]
pub struct SliceBackward {
    pub ranges: Vec<(usize, usize)>,
}

impl GradFn for SliceBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Slice requires 1 input");

        let input = &inputs[0];
        let input_shape = input.data.view.shape();

        // Sliceの勾配は、元のサイズに戻すためにパディングを追加
        // 各軸について: padding_before = ranges[i].0, padding_after = input_shape[i] - ranges[i].1

        let padding: Vec<(usize, usize)> = self
            .ranges
            .iter()
            .enumerate()
            .map(|(i, (start, end))| {
                // input_shape[i]から実際のサイズを取得
                let input_size = match &input_shape[i] {
                    crate::graph::shape::Expr::Const(val) => *val as usize,
                    _ => {
                        // 式の場合は計算（start + (end - start) + padding_after = input_size）
                        // つまり padding_after = input_size - end
                        // しかしinput_sizeが式なので、endとの差分を計算する必要がある
                        // とりあえず、定数のみサポート
                        panic!("SliceBackward requires constant input shape")
                    }
                };
                let padding_before = *start;
                let padding_after = input_size - end;
                (padding_before, padding_after)
            })
            .collect();

        vec![Some(grad_output.pad(padding, 0.0))]
    }
}

// === 畳み込み演算の勾配関数 ===

/// Conv1d演算の勾配
///
/// TODO: 完全な勾配実装にはfold演算（col2im）が必要
/// 現在は簡易実装として panic! を返す
#[derive(Debug)]
#[allow(dead_code)]
pub struct Conv1dBackward {
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl GradFn for Conv1dBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Conv1d requires 2 inputs (input, kernel)");
        let input = &inputs[0];
        let kernel = &inputs[1];

        // 入力とカーネルのshape取得
        let input_shape = input.data.view.shape();
        let kernel_shape = kernel.data.view.shape();
        let grad_output_shape = grad_output.data.view.shape();

        // 定数のみサポート（動的shapeは未対応）
        let c_in = match &input_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv1d backward requires constant input shape"),
        };
        let _l_in = match &input_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv1d backward requires constant input shape"),
        };

        let kernel_size = match &kernel_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv1d backward requires constant kernel shape"),
        };

        // 入力に対する勾配: 転置畳み込み
        // この実装は単純化されており、stride=1, dilation=1, groups=1のみサポート
        if self.stride != 1 || self.dilation != 1 || self.groups != 1 {
            panic!("Conv1d backward currently only supports stride=1, dilation=1, groups=1");
        }

        // grad_outputをパディング
        // grad_output: (C_out, L'), 必要なパディング: kernel_size - 1
        let padding = kernel_size - 1;
        let grad_output_padded = grad_output.data.pad(vec![(0, 0), (padding, padding)], 0.0);

        // kernelを反転（conv → correlation変換）
        // kernel: (C_out, C_in, k) -> flip axis 2 -> (C_out, C_in, k)
        let kernel_flipped_view = kernel.data.view.clone().flip(2);
        let kernel_flipped = kernel
            .data
            .view(kernel_flipped_view)
            .view(kernel.data.view.clone().permute(vec![1, 0, 2])); // (C_in, C_out, k)

        // 転置畳み込み（simplified version）
        // grad_input = conv(grad_output_padded, kernel_flipped)
        let grad_input_data = Tensor::from_graph_node(grad_output_padded, false)
            .conv1d(&Tensor::from_graph_node(kernel_flipped, false), 1, 1, 1)
            .data;

        // カーネルに対する勾配: 相関計算
        // grad_kernel = correlate(input, grad_output)
        // input: (C_in, L), grad_output: (C_out, L')
        // unfold input -> (C_in, k, L')
        let input_unfolded = input.data.unfold1d(kernel_size, 1, 1, 1);

        // unsqueeze grad_output: (C_out, L') -> (C_out, 1, 1, L')
        let grad_out_tmp1 = grad_output.data.view.clone().unsqueeze(1);
        let grad_out_tmp2 = grad_out_tmp1.unsqueeze(2);
        let grad_output_expanded_view = grad_output.data.view(grad_out_tmp2);

        // unsqueeze input_unfolded: (C_in, k, L') -> (1, C_in, k, L')
        let input_unf_tmp = input_unfolded.view.clone().unsqueeze(0);
        let input_unfolded_expanded = input_unfolded.view(input_unf_tmp);

        // expand to common shape: (C_out, C_in, k, L')
        let c_out = match &kernel_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v,
            _ => panic!("Conv1d backward requires constant kernel shape"),
        };
        let common_shape = vec![
            crate::graph::shape::Expr::from(c_out),
            crate::graph::shape::Expr::from(c_in as isize),
            crate::graph::shape::Expr::from(kernel_size as isize),
            grad_output_shape[1].clone(),
        ];

        let grad_out_broadcasted = grad_output_expanded_view.expand(common_shape.clone());
        let input_unf_broadcasted = input_unfolded_expanded.expand(common_shape);

        // multiply and reduce over L'
        let grad_kernel_data = (&grad_out_broadcasted * &input_unf_broadcasted).reduce_sum(3); // reduce over L'

        vec![
            Some(Tensor::from_graph_node(grad_input_data, false)),
            Some(Tensor::from_graph_node(grad_kernel_data, false)),
        ]
    }
}

/// Conv2d演算の勾配
///
/// TODO: 完全な勾配実装にはfold演算（col2im）が必要
/// 現在は簡易実装として panic! を返す
#[derive(Debug)]
#[allow(dead_code)]
pub struct Conv2dBackward {
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl GradFn for Conv2dBackward {
    fn apply(&self, _grad_output: &Tensor, _inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        // TODO: 完全な勾配計算を実装
        panic!(
            "Conv2d backward is not yet implemented. fold/col2im operation is required for proper gradient computation."
        );
    }
}

/// Conv3d演算の勾配
///
/// TODO: 完全な勾配実装にはfold演算（col2im）が必要
/// 現在は簡易実装として panic! を返す
#[derive(Debug)]
#[allow(dead_code)]
pub struct Conv3dBackward {
    pub stride: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub groups: usize,
}

impl GradFn for Conv3dBackward {
    fn apply(&self, _grad_output: &Tensor, _inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        // TODO: 完全な勾配計算を実装
        panic!(
            "Conv3d backward is not yet implemented. fold/col2im operation is required for proper gradient computation."
        );
    }
}
