//! 数学関数の勾配関数

use super::{GradFn, Tensor};

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
