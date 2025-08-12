use crate::ast::AstOp;
use super::{Tensor, TensorData, TensorOp};

impl Tensor {
    pub fn sum(&self, axis: usize) -> Tensor {
        let requires_grad = self.0.borrow().requires_grad;
        let mut shape = self.shape();
        if axis >= shape.len() {
            panic!("Axis {} is out of bounds for shape {:?}", axis, shape);
        }
        shape.remove(axis);
        let dtype = self.0.borrow().dtype.clone();
        TensorData::new(
            TensorOp::Reduce(AstOp::Add, axis),
            vec![self.clone()],
            shape,
            dtype,
            requires_grad,
            self.0.borrow().backend.clone(),
        )
        .into()
    }
}
