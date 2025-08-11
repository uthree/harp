use super::{Tensor, TensorData, TensorOp};

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl std::ops::$trait for Tensor {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                if self.0.borrow().shape != rhs.0.borrow().shape {
                    panic!(
                        "Shape mismatch for op {:?}: {:?} vs {:?}",
                        $op,
                        self.0.borrow().shape,
                        rhs.0.borrow().shape
                    );
                }
                let self_backend = &self.0.borrow().backend;
                let rhs_backend = &rhs.0.borrow().backend;
                if self_backend != rhs_backend {
                    panic!("Backends of tensors do not match");
                }
                if self.0.borrow().dtype != rhs.0.borrow().dtype {
                    panic!("Dtypes of tensors do not match");
                }
                let requires_grad = self.0.borrow().requires_grad || rhs.0.borrow().requires_grad;
                let shape = self.0.borrow().shape.clone();
                let dtype = self.0.borrow().dtype.clone();
                TensorData {
                    op: $op,
                    src: vec![self.clone(), rhs.clone()],
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

impl_binary_op!(Add, add, TensorOp::Add);
impl_binary_op!(Sub, sub, TensorOp::Sub);
impl_binary_op!(Mul, mul, TensorOp::Mul);

impl std::ops::Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

macro_rules! impl_binary_op_assign {
    ($trait:ident, $method:ident, $op_trait:ident, $op_method:ident) => {
        impl std::ops::$trait for Tensor {
            fn $method(&mut self, rhs: Self) {
                let new_tensor = std::ops::$op_trait::$op_method(self.clone(), rhs);
                self.0 = new_tensor.0;
            }
        }
    };
}

impl_binary_op_assign!(AddAssign, add_assign, Add, add);
impl_binary_op_assign!(SubAssign, sub_assign, Sub, sub);
impl_binary_op_assign!(MulAssign, mul_assign, Mul, mul);
impl_binary_op_assign!(DivAssign, div_assign, Div, div);
