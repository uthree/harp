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
