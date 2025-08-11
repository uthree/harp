use super::{Tensor, TensorOp};

impl Tensor {
    pub(super) fn grad_fn(&self, grad: Tensor, srcs: &[Tensor]) -> Vec<Tensor> {
        match self.0.borrow().op {
            TensorOp::Add => vec![grad.clone(), grad],
            TensorOp::Sub => vec![grad.clone(), -grad],
            TensorOp::Mul => {
                let a = srcs[0].clone();
                let b = srcs[1].clone();
                vec![grad.clone() * b, grad * a]
            }
            TensorOp::Neg => vec![-grad],
            TensorOp::Recip => {
                let a = srcs[0].clone();
                let recip_a = a.recip();
                vec![-grad * recip_a.clone() * recip_a]
            }
            TensorOp::Sin => {
                let a = srcs[0].clone();
                vec![grad * a.cos()]
            }
            TensorOp::Exp2 => {
                // d/dx(2^x) = 2^x * ln(2)
                let ln_2 = Tensor::full(
                    grad.0.borrow().shape.clone(),
                    grad.0.borrow().dtype.clone(),
                    (2.0f32).ln().into(),
                    false,
                );
                vec![grad * self.clone() * ln_2]
            }
            TensorOp::Log2 => {
                // d/dx(log2(x)) = 1 / (x * ln(2))
                let a = srcs[0].clone();
                let ln_2 = Tensor::full(
                    grad.0.borrow().shape.clone(),
                    grad.0.borrow().dtype.clone(),
                    (2.0f32).ln().into(),
                    false,
                );
                vec![grad * (a * ln_2).recip()]
            }
            TensorOp::Sqrt => {
                // d/dx(sqrt(x)) = 1 / (2 * sqrt(x))
                let two = Tensor::full(
                    grad.0.borrow().shape.clone(),
                    grad.0.borrow().dtype.clone(),
                    2.0.into(),
                    false,
                );
                vec![grad * (two * self.clone()).recip()]
            }
            _ => vec![],
        }
    }
}
