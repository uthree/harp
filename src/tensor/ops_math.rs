use super::{Tensor, TensorData, TensorOp};

impl Tensor {
    pub fn sin(self) -> Self {
        let requires_grad = self.0.borrow().requires_grad;
        let shape = self.0.borrow().shape.clone();
        let dtype = self.0.borrow().dtype.clone();
        TensorData {
            op: TensorOp::Sin,
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

    pub fn cos(self) -> Self {
        let pi_over_2 = Tensor::full(
            self.0.borrow().shape.clone(),
            self.0.borrow().dtype.clone(),
            (std::f32::consts::PI / 2.0).into(),
            false,
        );
        (self + pi_over_2).sin()
    }

    pub fn exp2(self) -> Self {
        let requires_grad = self.0.borrow().requires_grad;
        let shape = self.0.borrow().shape.clone();
        let dtype = self.0.borrow().dtype.clone();
        TensorData {
            op: TensorOp::Exp2,
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

    pub fn log2(self) -> Self {
        let requires_grad = self.0.borrow().requires_grad;
        let shape = self.0.borrow().shape.clone();
        let dtype = self.0.borrow().dtype.clone();
        TensorData {
            op: TensorOp::Log2,
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

    pub fn sqrt(self) -> Self {
        let requires_grad = self.0.borrow().requires_grad;
        let shape = self.0.borrow().shape.clone();
        let dtype = self.0.borrow().dtype.clone();
        TensorData {
            op: TensorOp::Sqrt,
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
