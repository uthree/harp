//! Provides the core `Tensor` struct and its associated operations.
//!
//! This module defines a lazily-evaluated tensor object that builds a computation
//! graph under the hood. The actual computation is deferred until the `forward()`
//! method is called. This allows for graph-level optimizations before execution.

use crate::ast::DType;
use crate::backend::Backend;
use crate::backend::c::CBackend;
use crate::cbuffer::CBuffer;
use crate::graph::Graph;
use std::cell::RefCell;
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
    /// Represents an operation that creates a tensor with random values.
    Rand,
    /// Represents an element-wise addition operation.
    Add,
    /// Represents an element-wise subtraction operation.
    Sub,
    /// Represents an element-wise multiplication operation.
    Mul,
    /// Represents an element-wise negation operation.
    Neg,
    /// Represents an element-wise reciprocal operation.
    Recip,
}

/// Contains the internal data and metadata for a `Tensor`.
///
/// This structure holds all the information needed to compute the tensor's value,
/// including its operation, source tensors, shape, data type, and eventually, the
/// computed buffer.
pub struct TensorData {
    /// The operation that produces this tensor.
    op: TensorOp,
    /// A list of source tensors for the operation.
    src: Vec<Tensor>,
    /// The shape of the tensor.
    shape: Shape,
    /// The data type of the tensor.
    dtype: DType,
    /// The computed data, available after `forward()` is called.
    buffer: Option<TensorBuffer>,
    /// The gradient of this tensor, computed during backpropagation.
    grad: Option<Tensor>,
    /// A flag indicating whether this tensor requires a gradient.
    requires_grad: bool,
    /// The backend used for computation.
    backend: TensorBackend,
}

/// A type alias for the shape of a tensor, represented as a vector of dimensions.
pub type Shape = Vec<usize>;

/// The primary tensor structure.
///
/// `Tensor` represents a node in the computation graph. It holds a reference-counted
/// pointer to `TensorData`, which contains the actual tensor information and operation.
/// The operations are deferred until `forward()` is called (lazy evaluation).
#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    /// Creates a new tensor with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to create.
    /// * `dtype` - The data type of the tensor's elements.
    pub fn rand(shape: Shape, dtype: DType) -> Self {
        TensorData {
            op: TensorOp::Rand,
            src: vec![],
            shape,
            dtype,
            buffer: None,
            grad: None,
            requires_grad: false,
            backend: backend("c"),
        }
        .into()
    }

    /// Triggers the computation of the tensor's value.
    ///
    /// This method performs the actual computation defined by the tensor's operation.
    /// It recursively calls `forward()` on its source tensors to ensure all dependencies
    // are computed first. If the tensor's buffer has already been computed, this
    /// method does nothing. This enables lazy evaluation.
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
            TensorOp::Add => srcs[0].clone() + srcs[1].clone(),
            TensorOp::Sub => srcs[0].clone() - srcs[1].clone(),
            TensorOp::Mul => srcs[0].clone() * srcs[1].clone(),
            TensorOp::Neg => -srcs[0].clone(),
            TensorOp::Recip => srcs[0].clone().recip(),
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
    pub fn recip(self) -> Self {
        let shape = self.0.borrow().shape.clone();
        let dtype = self.0.borrow().dtype.clone();
        TensorData {
            op: TensorOp::Recip,
            src: vec![self.clone()],
            shape,
            dtype,
            buffer: None,
            grad: None,
            requires_grad: false,
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
                let shape = self.0.borrow().shape.clone();
                let dtype = self.0.borrow().dtype.clone();
                TensorData {
                    op: $op,
                    src: vec![self.clone(), rhs.clone()],
                    shape,
                    dtype,
                    buffer: None,
                    grad: None,
                    requires_grad: false,
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
                let shape = self.0.borrow().shape.clone();
                let dtype = self.0.borrow().dtype.clone();
                TensorData {
                    op: $op,
                    src: vec![self.clone()],
                    shape,
                    dtype,
                    buffer: None,
                    grad: None,
                    requires_grad: false,
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
    fn test_tensor_rand_forward() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        assert!(a.0.borrow().buffer.is_none());
        a.forward();
        assert!(a.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_add_forward() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        let b = Tensor::rand(vec![10, 20], DType::F32);
        let c = a + b;
        assert!(c.0.borrow().buffer.is_none());
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_forward_is_lazy() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        let b = Tensor::rand(vec![10, 20], DType::F32);
        let c = a + b;
        // Buffer should not be allocated before forward is called
        assert!(c.0.borrow().buffer.is_none());
    }

    #[test]
    fn test_backward_dependency() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        let b = Tensor::rand(vec![10, 20], DType::F32);
        let c = a.clone() + b.clone();

        // Buffers for a and b should be None initially
        assert!(a.0.borrow().buffer.is_none());
        assert!(b.0.borrow().buffer.is_none());

        // When forward is called on c, it should recursively call forward on a and b
        c.forward();

        // Now, buffers for a, b, and c should all be allocated
        assert!(a.0.borrow().buffer.is_some());
        assert!(b.0.borrow().buffer.is_some());
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_sub_forward() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        let b = Tensor::rand(vec![10, 20], DType::F32);
        let c = a - b;
        assert!(c.0.borrow().buffer.is_none());
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_mul_forward() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        let b = Tensor::rand(vec![10, 20], DType::F32);
        let c = a * b;
        assert!(c.0.borrow().buffer.is_none());
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_div_forward() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        let b = Tensor::rand(vec![10, 20], DType::F32);
        let c = a / b;
        assert!(c.0.borrow().buffer.is_none());
        c.forward();
        assert!(c.0.borrow().buffer.is_some());
    }

    #[test]
    fn test_tensor_neg_forward() {
        let a = Tensor::rand(vec![10, 20], DType::F32);
        let b = -a;
        assert!(b.0.borrow().buffer.is_none());
        b.forward();
        assert!(b.0.borrow().buffer.is_some());
    }
}
