//! Provides the core `Tensor` struct and its associated operations.
//!
//! This module defines a lazily-evaluated tensor object that builds a computation
//! graph under the hood. The actual computation is deferred until the `forward()`
//! method is called. This allows for graph-level optimizations before execution.

mod creation;
mod grad;
mod ops_binary;
mod ops_math;
mod ops_shape;
mod ops_unary;

use crate::ast::{Const, DType};
use crate::backend::Backend;
use crate::c::{CBackend, CBuffer};
use crate::graph::Graph;
use once_cell::sync::Lazy;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

static C_BACKEND: Lazy<CBackend> = Lazy::new(CBackend::new);

/// A buffer holding the tensor's actual data on a computation device.
pub enum TensorBuffer {
    /// A buffer managed by the C backend.
    C(CBuffer),
}

/// Represents the computation backend where tensor operations are executed.
#[derive(Clone)]
pub enum TensorBackend {
    /// The C backend, which compiles and runs operations as C code.
    C,
}

impl PartialEq for TensorBackend {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorBackend::C, TensorBackend::C) => true,
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
    Sin,
    Exp2,
    Log2,
    Sqrt,
    Permute(Vec<usize>),
    Reshape(Vec<usize>),
    Expand(Vec<usize>),
    Squeeze(usize),
    Unsqueeze(usize),
    Slice(Vec<(usize, usize)>),
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
            TensorOp::Sin => srcs[0].sin(),
            TensorOp::Exp2 => srcs[0].exp2(),
            TensorOp::Log2 => srcs[0].log2(),
            TensorOp::Sqrt => srcs[0].sqrt(),
            TensorOp::Permute(axes) => srcs[0].permute(axes),
            TensorOp::Reshape(shape) => srcs[0].reshape(shape.iter().map(|&d| d.into()).collect()),
            TensorOp::Expand(shape) => srcs[0].expand(shape.iter().map(|&d| d.into()).collect()),
            TensorOp::Squeeze(dim) => srcs[0].squeeze(dim),
            TensorOp::Unsqueeze(dim) => srcs[0].unsqueeze(dim),
            TensorOp::Slice(args) => srcs[0].slice(
                args.iter()
                    .map(|(s, e)| ((*s).into(), (*e).into()))
                    .collect(),
            ),
        };

        op.as_output();

        let result_buffer = match data.backend.clone() {
            TensorBackend::C => {
                let backend = &C_BACKEND;
                TensorBuffer::C(backend.run(&graph).into_iter().last().unwrap())
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
}

impl From<TensorData> for Tensor {
    fn from(value: TensorData) -> Self {
        Tensor(Rc::new(RefCell::new(value)))
    }
}

pub fn backend(name: &str) -> TensorBackend {
    match name {
        "c" => TensorBackend::C,
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

    #[test]
    fn test_math_functions_forward() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = a.sin();
        b.forward();
        assert!(b.0.borrow().buffer.is_some());

        // Chaining operations
        let c = Tensor::ones(vec![10, 20], DType::F32, false);
        let d = c.cos().sqrt().exp2().log2();
        d.forward();
        assert!(d.0.borrow().buffer.is_some());
    }

    #[test]
    #[should_panic(expected = "Shape mismatch for op")]
    fn test_add_shape_mismatch_panics() {
        let a = Tensor::ones(vec![10, 20], DType::F32, false);
        let b = Tensor::ones(vec![20, 10], DType::F32, false);
        let _ = a + b;
    }

    #[test]
    fn test_shape_ops() {
        let a = Tensor::ones(vec![1, 10, 20], DType::F32, false);

        // Permute
        let p = a.permute(vec![2, 0, 1]);
        assert_eq!(p.0.borrow().shape, vec![20, 1, 10]);
        assert_eq!(p.0.borrow().op, TensorOp::Permute(vec![2, 0, 1]));

        // Reshape
        let r = a.reshape(vec![200]);
        assert_eq!(r.0.borrow().shape, vec![200]);
        assert_eq!(r.0.borrow().op, TensorOp::Reshape(vec![200]));

        // Squeeze
        let s = a.squeeze(0);
        assert_eq!(s.0.borrow().shape, vec![10, 20]);
        assert_eq!(s.0.borrow().op, TensorOp::Squeeze(0));

        // Unsqueeze
        let u = s.unsqueeze(1);
        assert_eq!(u.0.borrow().shape, vec![10, 1, 20]);
        assert_eq!(u.0.borrow().op, TensorOp::Unsqueeze(1));

        // Slice
        let sl = a.slice(vec![(0, 1), (2, 5), (8, 12)]);
        assert_eq!(sl.0.borrow().shape, vec![1, 3, 4]);
        assert_eq!(
            sl.0.borrow().op,
            TensorOp::Slice(vec![(0, 1), (2, 5), (8, 12)])
        );
    }
}
