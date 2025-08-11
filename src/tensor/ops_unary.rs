use super::{Tensor, TensorData, TensorOp};

macro_rules! impl_unary_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl std::ops::$trait for Tensor {
            type Output = Self;

            fn $method(self) -> Self::Output {
                let requires_grad = self.0.borrow().requires_grad;
                let shape = self.0.borrow().shape.clone();
                let dtype = self.0.borrow().dtype.clone();
                TensorData {
                    op: $op,
                    src: vec![self.clone()],
                    shape,
                    dtype,
                    buffer: None,
                    grad: None,
                    requires_grad,
                    backend: self.0.borrow().backend.clone(),
                }
                .into()
            }
        }
    };
}

impl_unary_op!(Neg, neg, TensorOp::Neg);

impl Tensor {
    pub fn recip(self) -> Self {
        let shape = self.0.borrow().shape.clone();
        let dtype = self.0.borrow().dtype.clone();
        let requires_grad = self.0.borrow().requires_grad;
        TensorData {
            op: TensorOp::Recip,
            src: vec![self.clone()],
            shape,
            dtype,
            buffer: None,
            grad: None,
            requires_grad,
            backend: self.0.borrow().backend.clone(),
        }
        .into()
    }
}
