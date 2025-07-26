//! The core `Tensor` struct and related components.
//!
//! This module defines the central data structure of the library, `Tensor`, which
//! represents a multi-dimensional array. All operations on `Tensor`s are lazy,
//! meaning they build up a computation graph rather than executing immediately.
//!
//! The actual computation is triggered by calling the `.realize()` method, which
//! hands off the generated computation graph to the appropriate `Backend`.

use crate::backends::{Backend, Buffer};
use crate::context;
use crate::dot::ToDot;
use crate::dtype::{DType, IntoDType};
use crate::linearizer::Linearizer;
use crate::lowerizer::Lowerizer;
use crate::optimizer::Optimizer;
use crate::shapetracker::ShapeTracker;
use crate::uop::Op;
use log::debug;
use ndarray::ArrayD;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// Represents the operation that created a `Tensor`.
///
/// This enum tracks the origin of a `Tensor` within the computation graph.
#[derive(Debug, Clone)]
pub enum TensorOp {
    /// A source tensor, typically representing data loaded from memory.
    Load,
    /// A tensor created as the result of a unary operation.
    Unary(Op),
    /// A tensor created as the result of a binary operation.
    Binary(Op),
}

/// The internal, reference-counted implementation of a `Tensor`.
///
/// This struct holds all the metadata required to define a node in the computation graph,
/// such as the operation that created it, its sources (parent nodes), its shape,
/// and the backend responsible for its computation.
pub struct Tensor_<T> {
    /// The operation that produced this tensor.
    pub op: TensorOp,
    /// The source tensors (inputs) for the operation.
    pub src: Vec<Tensor<T>>,
    /// A `ShapeTracker` that manages the tensor's shape and memory layout.
    pub tracker: ShapeTracker,
    /// The data type of the tensor's elements.
    pub dtype: DType,
    /// The backend responsible for executing computations for this tensor.
    pub backend: Rc<dyn Backend>,
    /// A cached handle to the realized memory buffer, once computed.
    pub realized: RefCell<Option<Buffer>>,
    phantom: std::marker::PhantomData<T>,
}

/// A lazy, multi-dimensional array (tensor).
///
/// `Tensor` is the main user-facing struct. It's a lightweight handle (a reference-counted
/// pointer) to the underlying tensor data (`Tensor_`). Operations on `Tensor`s are
/// performed lazily, building up a computation graph.
///
/// To execute the computation and get the result, call the `.realize()` method.
#[derive(Clone)]
pub struct Tensor<T>(pub Rc<Tensor_<T>>);

impl<T: Clone + Default + 'static + IntoDType> Tensor<T> {
    /// Creates a new `Tensor`.
    ///
    /// This is the primary constructor used to build nodes in the computation graph.
    pub fn new(
        op: TensorOp,
        src: Vec<Tensor<T>>,
        tracker: ShapeTracker,
        dtype: DType,
        backend: Rc<dyn Backend>,
    ) -> Self {
        Self(Rc::new(Tensor_ {
            op,
            src,
            tracker,
            dtype,
            backend,
            realized: RefCell::new(None),
            phantom: std::marker::PhantomData,
        }))
    }

    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        let backend = context::backend("clang");
        let buffer = backend.alloc(
            data.len() * std::mem::size_of::<T>(),
            backend.clone(),
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                backend.get_buffer_ptr(buffer.id) as *mut T,
                data.len(),
            );
        }
        let tracker = ShapeTracker::new(shape.to_vec());
        let tensor = Self::new(
            TensorOp::Load,
            vec![],
            tracker,
            T::into_dtype(),
            backend,
        );
        *tensor.0.realized.borrow_mut() = Some(buffer);
        tensor
    }

    pub fn to_vec(&self) -> Vec<T> {
        let buffer = self.realize();
        let ptr = self.0.backend.get_buffer_ptr(buffer.id) as *const T;
        let len = self.0.tracker.shape().iter().product();
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    /// Triggers the computation of the tensor's value.
    ///
    /// This method walks the computation graph backwards from this tensor, generates
    /// an optimized intermediate representation (`UOp`), renders it to C code,
    /// compiles it, and executes it on the backend.
    ///
    /// The result is a `Buffer` handle, which points to the memory on the compute
    /// device (e.g., CPU) containing the tensor's data. Results are cached, so
    /// subsequent calls to `.realize()` on the same tensor will be fast.
    pub fn realize(&self) -> Buffer {
        if let Some(ref realized) = *self.0.realized.borrow() {
            debug!("Cache hit for tensor");
            return realized.clone();
        }
        debug!("Realizing tensor with op: {:?}", self.0.op);

        let result_var = match self.0.op {
            TensorOp::Load => {
                // This case should be handled by from_vec, but as a fallback, allocate.
                let size: usize =
                    self.0.tracker.shape().iter().product::<usize>() * self.0.dtype.size();
                debug!("Allocating new buffer for Load op with size: {size}");
                self.0.backend.alloc(size, self.0.backend.clone())
            }
            TensorOp::Unary(_) | TensorOp::Binary(_) => {
                let args: Vec<_> = self.0.src.iter().map(|t| t.realize()).collect();
                let size: usize =
                    self.0.tracker.shape().iter().product::<usize>() * self.0.dtype.size();
                let output_buffer = self.0.backend.alloc(size, self.0.backend.clone());
                let mut kernel_args = args;
                kernel_args.push(output_buffer.clone());

                let mut lowerizer: Lowerizer<T> = Lowerizer::new();
                let uop_graph = lowerizer.lower(self);
                debug!("Generated UOp graph: {uop_graph:?}");

                let optimizer = Optimizer::new();
                let optimized_uop_graph = optimizer.optimize(&uop_graph);
                debug!("Optimized UOp graph: {optimized_uop_graph:?}");

                let mut linearizer = Linearizer::new();
                let kernel = linearizer.linearize(&optimized_uop_graph, self.shape());
                let args_ref: Vec<&Buffer> = kernel_args.iter().collect();
                self.0.backend.compile_and_exec(&kernel, &args_ref, &[]);
                output_buffer
            }
        };

        *self.0.realized.borrow_mut() = Some(result_var.clone());
        result_var
    }

    pub fn shape(&self) -> &[usize] {
        self.0.tracker.shape()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_tracker = self.0.tracker.reshape(new_shape);
        Self::new(
            self.0.op.clone(),
            self.0.src.clone(),
            new_tracker,
            self.0.dtype.clone(),
            self.0.backend.clone(),
        )
    }

    fn lazy_unary_op(&self, op: Op) -> Self {
        Self::new(
            TensorOp::Unary(op),
            vec![self.clone()],
            self.0.tracker.clone(),
            self.0.dtype.clone(),
            self.0.backend.clone(),
        )
    }

    fn lazy_binary_op(op: Op, a: &Self, b: &Self) -> Self {
        assert!(
            Rc::ptr_eq(&a.0.backend, &b.0.backend),
            "Backends must be the same for binary operations"
        );
        Self::new(
            TensorOp::Binary(op),
            vec![a.clone(), b.clone()],
            a.0.tracker.clone(),
            a.0.dtype.clone(),
            a.0.backend.clone(),
        )
    }

    pub fn exp2(&self) -> Self {
        self.lazy_unary_op(Op::Exp2)
    }
    pub fn log2(&self) -> Self {
        self.lazy_unary_op(Op::Log2)
    }
    pub fn sqrt(&self) -> Self {
        self.lazy_unary_op(Op::Sqrt)
    }
    pub fn sin(&self) -> Self {
        self.lazy_unary_op(Op::Sin)
    }
}

impl<T: Clone + Default + 'static + IntoDType> Add for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Add, self, rhs)
    }
}

impl<T: Clone + Default + 'static + IntoDType> Add for Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T: Clone + Default + 'static + IntoDType> Sub for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Sub, self, rhs)
    }
}

impl<T: Clone + Default + 'static + IntoDType> Sub for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T: Clone + Default + 'static + IntoDType> Mul for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Mul, self, rhs)
    }
}

impl<T: Clone + Default + 'static + IntoDType> Mul for Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<T: Clone + Default + 'static + IntoDType> Div for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Div, self, rhs)
    }
}

impl<T: Clone + Default + 'static + IntoDType> Div for Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<T: Clone + Default + 'static + IntoDType> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Self::Output {
        self.lazy_unary_op(Op::Neg)
    }
}

impl<T: Clone + Default + 'static + IntoDType> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: Clone + Default + 'static + IntoDType> ToDot for Tensor<T> {
    fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph G {\n");
        dot.push_str("  node [shape=box];\n");
        let mut visited = FxHashSet::default();
        build_dot_tensor(self, &mut dot, &mut visited);
        dot.push_str("}\n");
        dot
    }
}

fn build_dot_tensor<T: Clone + Default + 'static + IntoDType>(
    tensor: &Tensor<T>,
    dot: &mut String,
    visited: &mut FxHashSet<*const Tensor_<T>>,
) {
    let ptr = Rc::as_ptr(&tensor.0);
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    let label = format!(
        "op: {:?}\nshape: {:?}\ndtype: {:?}",
        tensor.0.op,
        tensor.shape(),
        tensor.0.dtype
    )
    .replace('\n', "\\n");
    dot.push_str(&format!("  \"{ptr:p}\" [label=\"{label}\"];\n"));

    for src in &tensor.0.src {
        let src_ptr = Rc::as_ptr(&src.0);
        dot.push_str(&format!("  \"{src_ptr:p}\" -> \"{ptr:p}\";\n"));
        build_dot_tensor(src, dot, visited);
    }
}

impl<T: Clone + Default + 'static + IntoDType> From<ArrayD<T>> for Tensor<T> {
    fn from(arr: ArrayD<T>) -> Self {
        let shape = arr.shape().to_vec();
        let (data, _) = arr.into_raw_vec_and_offset();
        Tensor::from_vec(data, &shape)
    }
}

impl<T: Clone + Default + 'static + IntoDType> From<Tensor<T>> for ArrayD<T> {
    fn from(tensor: Tensor<T>) -> Self {
        let shape = tensor.shape().to_vec();
        let data = tensor.to_vec();
        ArrayD::from_shape_vec(shape, data).unwrap()
    }
}
