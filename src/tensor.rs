//! The core `Tensor` struct and related components.
//!
//! This module defines the central data structure of the library, `Tensor`, which
//! represents a multi-dimensional array. All operations on `Tensor`s are lazy,
//! meaning they build up a computation graph rather than executing immediately.
//!
//! The actual computation is triggered by calling the `.realize()` method, which
//! hands off the generated computation graph to the appropriate `Backend`.

use crate::autotuner::Configuration;
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
use rustc_hash::{FxHashSet, FxHashMap};
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

    /// Triggers the computation of the tensor's value using the default configuration.
    ///
    /// This is a convenience wrapper around `realize_with_config`.
    pub fn realize(&self) -> Buffer {
        self.realize_with_config(&Configuration::default())
    }

    /// Triggers the computation of the tensor's value with a specific configuration.
    ///
    /// This method walks the computation graph backwards from this tensor, generates
    /// an optimized intermediate representation (`UOp`), renders it to C code,
    /// compiles it, and executes it on the backend, all according to the provided
    /// `Configuration`.
    ///
    /// The result is a `Buffer` handle, which points to the memory on the compute
    /// device. Results are cached, so subsequent calls to `.realize()` on the same
    /// tensor will be fast unless the cache is cleared.
    pub fn realize_with_config(&self, config: &Configuration) -> Buffer {
        if let Some(ref realized) = *self.0.realized.borrow() {
            debug!("Cache hit for tensor");
            return realized.clone();
        }
        debug!("Realizing tensor with op: {:?}", self.0.op);

        let result_var = match self.0.op {
            TensorOp::Load => {
                let size: usize =
                    self.0.tracker.shape().iter().product::<usize>() * self.0.dtype.size();
                debug!("Allocating new buffer for Load op with size: {size}");
                self.0.backend.alloc(size, self.0.backend.clone())
            }
            TensorOp::Unary(_) | TensorOp::Binary(_) => {
                // Realize all source tensors first.
                self.0
                    .src
                    .iter()
                    .for_each(|t| _ = t.realize_with_config(config));

                let size: usize =
                    self.0.tracker.shape().iter().product::<usize>() * self.0.dtype.size();
                let output_buffer = self.0.backend.alloc(size, self.0.backend.clone());

                // The arguments to the kernel are all the leaf tensors (inputs) in the graph,
                // plus the output tensor itself.
                let leaf_tensors = self.get_leaf_tensors();
                let mut kernel_arg_tensors: Vec<&Tensor<T>> = leaf_tensors.iter().collect();
                kernel_arg_tensors.push(self);

                let kernel_args_bufs: Vec<Buffer> = kernel_arg_tensors
                    .iter()
                    .map(|t| {
                        if t.0.realized.borrow().is_some() {
                            t.0.realized.borrow().clone().unwrap()
                        } else {
                            // This must be the output buffer.
                            output_buffer.clone()
                        }
                    })
                    .collect();
                let kernel_args_bufs_ref: Vec<&Buffer> = kernel_args_bufs.iter().collect();

                let mut lowerizer = Lowerizer::new(&kernel_arg_tensors);
                let uop_graph = lowerizer.lower(self);
                debug!("Generated UOp graph: {uop_graph:?}");

                let optimizer = Optimizer::new(config);
                let optimized_uop_graph = optimizer.optimize(&uop_graph);
                debug!("Optimized UOp graph: {optimized_uop_graph:?}");

                let mut linearizer = Linearizer::new();
                let kernel = linearizer.linearize(&optimized_uop_graph, self.shape());

                self.0.backend.compile_and_exec(
                    &kernel,
                    &kernel_args_bufs_ref,
                    &[],
                    &config.backend_options,
                );
                output_buffer
            }
        };

        *self.0.realized.borrow_mut() = Some(result_var.clone());
        result_var
    }

    /// Clears the cached realized buffer for this tensor and its ancestors.
    ///
    /// This is necessary for the autotuner to re-run the computation with
    /// different configurations.
    pub fn clear_cache(&self) {
        let mut visited = FxHashSet::default();
        self.clear_cache_recursive(&mut visited);
    }

    fn clear_cache_recursive(&self, visited: &mut FxHashSet<*const Tensor_<T>>) {
        let ptr = Rc::as_ptr(&self.0);
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        *self.0.realized.borrow_mut() = None;

        for src in &self.0.src {
            src.clear_cache_recursive(visited);
        }
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

    /// Recursively collects all unique leaf tensors (those with `TensorOp::Load`)
    /// in the computation graph starting from this tensor.
    pub fn get_leaf_tensors(&self) -> Vec<Tensor<T>> {
        let mut leafs = FxHashMap::default();
        let mut visited = FxHashSet::default();
        self.collect_leaf_tensors_recursive(&mut leafs, &mut visited);
        leafs.into_values().collect()
    }

    fn collect_leaf_tensors_recursive(
        &self,
        leafs: &mut FxHashMap<*const Tensor_<T>, Tensor<T>>,
        visited: &mut FxHashSet<*const Tensor_<T>>,
    ) {
        let ptr = Rc::as_ptr(&self.0);
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        if let TensorOp::Load = self.0.op {
            leafs.insert(ptr, self.clone());
        }

        for src in &self.0.src {
            src.collect_leaf_tensors_recursive(leafs, visited);
        }
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
