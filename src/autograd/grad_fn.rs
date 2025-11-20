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
