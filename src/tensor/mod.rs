//! Provides the core `Tensor` struct and its associated operations.
//!
//! This module defines a lazily-evaluated tensor object that builds a computation
//! graph under the hood. The actual computation is deferred until the `forward()`
//! method is called. This allows for graph-level optimizations before execution.

use crate::ast::{Const, DType};
use crate::backend::Backend;
use crate::backend::c::CBackend;
use crate::cbuffer::CBuffer;
use crate::graph::Graph;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

thread_local! {
    static C_BACKEND: Rc<RefCell<CBackend>> = Rc::new(RefCell::new(CBackend::new()));
}

/// A buffer holding the tensor's actual data on a computation device.
pub enum TensorBuffer {
    /// A buffer managed by the C backend.
    C(CBuffer),
}

/// Represents the computation backend where tensor operations are executed.
#[derive(Clone)]
pub enum TensorBackend {
    /// The C backend, which compiles and runs operations as C code.
    C(Rc<RefCell<CBackend>>),
}

impl PartialEq for TensorBackend {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorBackend::C(a), TensorBackend::C(b)) => Rc::ptr_eq(a, b),
        }
    }
}

/// Defines the types of operations that can create a `Tensor`.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorOp {
    Rand,
    Full(Const),
    Add,
    Sub,
    Mul,
    Neg,
    Recip,
}

/// Contains the internal data and metadata for a `Tensor`.
pub struct TensorData {
    op: TensorOp,
    src: Vec<Tensor>,
    shape: Shape,
    dtype: DType,
    buffer: Option<TensorBuffer>,
    grad: Option<Tensor>,
    requires_grad: bool,
    backend: TensorBackend,
}

/// A type alias for the shape of a tensor, represented as a vector of dimensions.
pub type Shape = Vec<usize>;

/// The primary tensor structure.
#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    pub fn rand(shape: Shape, dtype: DType, requires_grad: bool) -> Self {
        TensorData {
            op: TensorOp::Rand,
            src: vec![],
            shape,
            dtype,
            buffer: None,
            grad: None,
            requires_grad,
            backend: backend("c"),
        }
        .into()
    }

    pub fn full(shape: Shape, dtype: DType, value: Const, requires_grad: bool) -> Self {
        TensorData {
            op: TensorOp::Full(value),
            src: vec![],
            shape,
            dtype,
            buffer: None,
            grad: None,
            requires_grad,
            backend: backend("c"),
        }
        .into()
    }

    pub fn ones(shape: Shape, dtype: DType, requires_grad: bool) -> Self {
        Self::full(shape, dtype, Const::from(1.0), requires_grad)
    }

    pub fn zeros(shape: Shape, dtype: DType, requires_grad: bool) -> Self {
        Self::full(shape, dtype, Const::from(0.0), requires_grad)
    }

    pub fn forward(&self) {
        if self.0.borrow().buffer.is_some() {
            return;
        }

        for s in &self.0.borrow().src {
            s.forward();
        }

        let graph = Graph::new();
        let data = self.0.borrow();

        let srcs: Vec<_> = data
            .src
            .iter()
            .map(|s| {
                let s_data = s.0.borrow();
                graph.input(
                    s_data.dtype.clone(),
                    s_data.shape.iter().map(|d| (*d).into()).collect(),
                )
            })
            .collect();

        let op = match data.op.clone() {
            TensorOp::Rand => graph.rand(
                data.dtype.clone(),
                data.shape.iter().map(|d| (*d).into()).collect(),
            ),
            TensorOp::Full(val) => {
                graph.full(val, data.shape.iter().map(|d| (*d).into()).collect())
            }
            TensorOp::Add => srcs[0] + srcs[1],
            TensorOp::Sub => srcs[0] - srcs[1],
            TensorOp::Mul => srcs[0] * srcs[1],
            TensorOp::Neg => -srcs[0],
            TensorOp::Recip => srcs[0].recip(),
        };

        op.as_output();

        let result_buffer = match data.backend.clone() {
            TensorBackend::C(b) => {
                TensorBuffer::C(b.borrow_mut().run(&graph).into_iter().last().unwrap())
            }
        };

        drop(data);
        self.0.borrow_mut().buffer = Some(result_buffer);
    }

    pub fn backward(&self) {
        if !self.0.borrow().requires_grad {
            return;
        }
        if self.0.borrow().shape.iter().product::<usize>() != 1 {
            panic!("backward() can only be called on a scalar tensor.");
        }

        let mut tape = Vec::new();
        self.build_tape(&mut tape, &mut HashSet::new());

        let mut grads: HashMap<*const TensorData, Tensor> = HashMap::new();
        grads.insert(
            self.0.as_ptr(),
            Tensor::ones(
                self.0.borrow().shape.clone(),
                self.0.borrow().dtype.clone(),
                false,
            ),
        );

        for tensor in tape.iter().rev() {
            let tensor_ptr = tensor.0.as_ptr() as *const TensorData;
            let grad = match grads.get(&tensor_ptr) {
                Some(g) => g.clone(),
                None => continue,
            };

            let (srcs, requires_grad) = {
                let data = tensor.0.borrow();
                (data.src.clone(), data.requires_grad)
            };

            if !requires_grad {
                continue;
            }

            let new_grads = tensor.grad_fn(grad, &srcs);

            for (src_tensor, grad_val) in srcs.iter().zip(new_grads) {
                let src_ptr = src_tensor.0.as_ptr() as *const TensorData;
                let entry = grads.entry(src_ptr).or_insert_with(|| {
                    Tensor::zeros(
                        src_tensor.0.borrow().shape.clone(),
                        src_tensor.0.borrow().dtype.clone(),
                        false,
                    )
                });
                *entry += grad_val;
            }
        }

        for (ptr, grad_tensor) in grads {
            let mut_ptr = ptr as *mut TensorData;
            unsafe {
                (*mut_ptr).grad = Some(grad_tensor);
            }
        }
    }

    fn grad_fn(&self, grad: Tensor, srcs: &[Tensor]) -> Vec<Tensor> {
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
            _ => vec![],
        }
    }

    fn build_tape(&self, tape: &mut Vec<Tensor>, visited: &mut HashSet<*const TensorData>) {
        let ptr = self.0.as_ptr() as *const TensorData;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        for src in &self.0.borrow().src {
            src.build_tape(tape, visited);
        }
        tape.push(self.clone());
    }

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

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl std::ops::$trait for Tensor {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
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

impl_binary_op!(Add, add, TensorOp::Add);
impl_binary_op!(Sub, sub, TensorOp::Sub);
impl_binary_op!(Mul, mul, TensorOp::Mul);
impl_unary_op!(Neg, neg, TensorOp::Neg);

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

impl From<TensorData> for Tensor {
    fn from(value: TensorData) -> Self {
        Tensor(Rc::new(RefCell::new(value)))
    }
}

pub fn backend(name: &str) -> TensorBackend {
    match name {
        "c" => C_BACKEND.with(|backend| TensorBackend::C(backend.clone())),
        _ => panic!("Unsupported backend: {}", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_tensor_creation() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        assert!(a.0.borrow().buffer.is_none());
        a.forward();
        assert!(a.0.borrow().buffer.is_some());

        let b = Tensor::zeros(vec![10, 20], DType::F32, false);
        b.forward();
        assert!(b.0.borrow().buffer.is_some());

        let c = Tensor::full(vec![10, 20], DType::F32, 5.0.into(), false);
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_add_forward() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = Tensor::ones(vec![10, 20], DType::F32, false);
        let c = a + b;
        assert!(c.0.borrow().buffer.is_none());
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_forward_is_lazy() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = Tensor::ones(vec![10, 20], DType::F32, false);
        let c = a + b;
        assert!(c.0.borrow().buffer.is_none());
    }

    #[test]
    fn test_backward_dependency() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = Tensor::ones(vec![10, 20], DType::F32, false);
        let c = a.clone() + b.clone();
        assert!(a.0.borrow().buffer.is_none());
        assert!(b.0.borrow().buffer.is_none());
        c.forward();
        assert!(a.0.borrow().buffer.is_some());
        assert!(b.0.borrow().buffer.is_some());
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_sub_forward() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = Tensor::ones(vec![10, 20], DType::F32, false);
        let c = a - b;
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_mul_forward() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = Tensor::ones(vec![10, 20], DType::F32, false);
        let c = a * b;
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_div_forward() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = Tensor::ones(vec![10, 20], DType::F32, false);
        let c = a / b;
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_neg_forward() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = -a;
        b.forward();
        assert!(b.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_add_assign() {
        let mut a = Tensor::ones(vec![10, 20], DType::F32, false);
        let a_original_ptr = Rc::as_ptr(&a.0);
        let b = Tensor::ones(vec![10, 20], DType::F32, false);
        a += b;
        let a_new_ptr = Rc::as_ptr(&a.0);
        assert_ne!(a_original_ptr, a_new_ptr);
        assert_eq!(a.0.borrow().op, TensorOp::Add);
        a.forward();
        assert!(a.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_simple_backward() {
        let a = Tensor::ones(vec![1], DType::F32, true);
        let b = Tensor::full(vec![1], DType::F32, 2.0.into(), true);
        let c = a.clone() * b.clone();
        c.backward();
        assert!(a.0.borrow().grad.is_some());
        assert!(b.0.borrow().grad.is_some());
    }

    #[test]
    fn test_grad_accumulation() {
        let a = Tensor::full(vec![1], DType::F32, 3.0.into(), true);
        let b = a.clone() * a.clone(); // y = a^2
        b.backward();
        assert!(a.0.borrow().grad.is_some());
    }

    #[test]
    fn test_complex_backward() {
        let a = Tensor::ones(vec![1], DType::F32, true);
        let b = Tensor::full(vec![1], DType::F32, 2.0.into(), true);
        let c = Tensor::full(vec![1], DType::F32, 3.0.into(), true);
        let z = a.clone() * b.clone() + c.clone();
        z.backward();
        assert!(a.0.borrow().grad.is_some());
        assert!(b.0.borrow().grad.is_some());
        assert!(c.0.borrow().grad.is_some());
    }
}
