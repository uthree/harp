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
use crate::dtype::DType;
use crate::linearizer::Linearizer;
use crate::lowerizer::Lowerizer;
use crate::optimizer::Optimizer;
use crate::shapetracker::ShapeTracker;
use crate::uop::Op;
use log::debug;
use ndarray::ArrayD;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::rc::Rc;
use std::time::Duration;

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
    /// A tensor created as the result of a reduce operation.
    Reduce(usize, Op),
    Constant(crate::dtype::Number),
}

/// The internal, reference-counted implementation of a `Tensor`.
///
/// This struct holds all the metadata required to define a node in the computation graph,
/// such as the operation that created it, its sources (parent nodes), its shape,
/// and the backend responsible for its computation.
pub struct Tensor_ {
    /// The operation that produced this tensor.
    pub op: TensorOp,
    /// The source tensors (inputs) for the operation.
    pub src: Vec<Tensor>,
    /// A `ShapeTracker` that manages the tensor's shape and memory layout.
    pub tracker: ShapeTracker,
    /// The data type of the tensor's elements.
    pub dtype: DType,
    /// The backend responsible for executing computations for this tensor.
    pub backend: Rc<dyn Backend>,
    /// A cached handle to the realized memory buffer, once computed.
    pub realized: RefCell<Option<Buffer>>,
}

/// A lazy, multi-dimensional array (tensor).
///
/// `Tensor` is the main user-facing struct. It's a lightweight handle (a reference-counted
/// pointer) to the underlying tensor data (`Tensor_`). Operations on `Tensor`s are
/// performed lazily, building up a computation graph.
///
/// To execute the computation and get the result, call the `.realize()` method.
#[derive(Clone)]
pub struct Tensor(pub Rc<Tensor_>);

impl Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    /// Creates a new `Tensor`.
    ///
    /// This is the primary constructor used to build nodes in the computation graph.
    pub fn new(
        op: TensorOp,
        src: Vec<Tensor>,
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
        }))
    }

    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let backend = context::backend("clang");
        let tracker = ShapeTracker::new(shape);
        let op = match dtype {
            DType::F32 => TensorOp::Constant(crate::dtype::Number::F32(0.0)),
            DType::F64 => TensorOp::Constant(crate::dtype::Number::F64(0.0)),
            DType::I32 => TensorOp::Constant(crate::dtype::Number::I32(0)),
            DType::I64 => TensorOp::Constant(crate::dtype::Number::I64(0)),
            _ => unimplemented!(),
        };
        Self::new(op, vec![], tracker, dtype, backend)
    }

    pub fn ones(shape: Vec<usize>, dtype: DType) -> Self {
        let backend = context::backend("clang");
        let tracker = ShapeTracker::new(shape);
        let op = match dtype {
            DType::F32 => TensorOp::Constant(crate::dtype::Number::F32(1.0)),
            DType::F64 => TensorOp::Constant(crate::dtype::Number::F64(1.0)),
            DType::I32 => TensorOp::Constant(crate::dtype::Number::I32(1)),
            DType::I64 => TensorOp::Constant(crate::dtype::Number::I64(1)),
            _ => unimplemented!(),
        };
        Self::new(op, vec![], tracker, dtype, backend)
    }

    /// Triggers the computation of the tensor's value using the default configuration.
    ///
    /// This is a convenience wrapper around `realize_with_config` that discards the duration.
    pub fn realize(&self) -> Buffer {
        self.realize_with_config(&Configuration::default()).0
    }

    /// Triggers the computation and returns the resulting buffer and execution duration.
    ///
    /// This method walks the computation graph, generates and optimizes the kernel,
    /// compiles it, and executes it. Results are cached.
    pub fn realize_with_config(&self, config: &Configuration) -> (Buffer, Duration) {
        if let Some(ref realized) = *self.realized.borrow() {
            debug!("Cache hit for tensor");
            return (realized.clone(), Duration::ZERO);
        }
        debug!("Realizing tensor with op: {:?}", self.op);

        let (result_var, exec_time) = match self.op {
            TensorOp::Load => {
                let size: usize =
                    self.tracker.shape().iter().product::<usize>() * self.dtype.size();
                debug!("Allocating new buffer for Load op with size: {size}");
                (
                    self.backend.alloc(size, self.backend.clone()),
                    Duration::ZERO,
                )
            }
            TensorOp::Unary(_)
            | TensorOp::Binary(_)
            | TensorOp::Reduce(_, _)
            | TensorOp::Constant(_) => {
                // Realize all source tensors first and accumulate their execution time.
                let mut total_exec_time = Duration::ZERO;
                self.src.iter().for_each(|t| {
                    let (_, exec_time) = t.realize_with_config(config);
                    total_exec_time += exec_time;
                });

                let size: usize =
                    self.tracker.shape().iter().product::<usize>() * self.dtype.size();
                let output_buffer = self.backend.alloc(size, self.backend.clone());

                let leaf_tensors = self.get_leaf_tensors();
                let mut kernel_arg_tensors: Vec<&Tensor> = leaf_tensors.iter().collect();
                kernel_arg_tensors.push(self);

                let kernel_args_bufs: Vec<Buffer> = kernel_arg_tensors
                    .iter()
                    .map(|t| {
                        if t.realized.borrow().is_some() {
                            t.realized.borrow().clone().unwrap()
                        } else {
                            output_buffer.clone()
                        }
                    })
                    .collect();
                let kernel_args_bufs_ref: Vec<&Buffer> = kernel_args_bufs.iter().collect();

                let mut lowerizer = Lowerizer::new(&kernel_arg_tensors);
                let uop_graph = lowerizer.lower(self);
                debug!("Generated UOp graph: {uop_graph:?}");

                let baseline_optimizer = Optimizer::new_baseline();
                let baseline_optimized_uop = baseline_optimizer.optimize(&uop_graph);
                debug!("After baseline optimization: {baseline_optimized_uop:?}");

                let tuning_optimizer = Optimizer::new_for_tuning(config);
                let final_optimized_uop = tuning_optimizer.optimize(&baseline_optimized_uop);
                debug!("After tuning optimization: {final_optimized_uop:?}");

                // --- Fusion-aware Linearization ---
                let use_counts = final_optimized_uop.get_use_counts();
                let mut linearizer = Linearizer::new(&use_counts);
                let kernel = linearizer.linearize(&final_optimized_uop, self.shape());

                let own_exec_time = self.backend.compile_and_exec(
                    &kernel,
                    &kernel_args_bufs_ref,
                    &[],
                    &config.backend_options,
                );
                (output_buffer, total_exec_time + own_exec_time)
            }
        };

        *self.realized.borrow_mut() = Some(result_var.clone());
        (result_var, exec_time)
    }

    /// Clears the cached realized buffer for this tensor and its ancestors.
    ///
    /// This is necessary for the autotuner to re-run the computation with
    /// different configurations.
    pub fn clear_cache(&self) {
        let mut visited = FxHashSet::default();
        self.clear_cache_recursive(&mut visited);
    }

    fn clear_cache_recursive(&self, visited: &mut FxHashSet<*const Tensor_>) {
        let ptr = Rc::as_ptr(&self.0);
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        *self.realized.borrow_mut() = None;

        for src in &self.src {
            src.clear_cache_recursive(visited);
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.tracker.shape()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_tracker = self.tracker.reshape(new_shape);
        Self::new(
            self.op.clone(),
            self.src.clone(),
            new_tracker,
            self.dtype.clone(),
            self.backend.clone(),
        )
    }

    fn lazy_unary_op(&self, op: Op) -> Self {
        Self::new(
            TensorOp::Unary(op),
            vec![self.clone()],
            self.tracker.clone(),
            self.dtype.clone(),
            self.backend.clone(),
        )
    }

    fn lazy_binary_op(op: Op, a: &Self, b: &Self) -> Self {
        assert!(
            Rc::ptr_eq(&a.backend, &b.backend),
            "Backends must be the same for binary operations"
        );
        Self::new(
            TensorOp::Binary(op),
            vec![a.clone(), b.clone()],
            a.tracker.clone(),
            a.dtype.clone(),
            a.backend.clone(),
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

    pub fn recip(&self) -> Self {
        self.lazy_unary_op(Op::Recip)
    }

    pub fn reduce(&self, axis: usize, op: Op) -> Self {
        let mut new_shape = self.shape().to_vec();
        new_shape.remove(axis);
        let new_tracker = ShapeTracker::new(new_shape);
        Self::new(
            TensorOp::Reduce(axis, op),
            vec![self.clone()],
            new_tracker,
            self.dtype.clone(),
            self.backend.clone(),
        )
    }

    pub fn sum(&self, axis: usize) -> Self {
        self.reduce(axis, Op::Add)
    }

    pub fn max(&self, rhs: &Self) -> Self {
        Self::lazy_binary_op(Op::Max, self, rhs)
    }

    /// Recursively collects all unique leaf tensors (those with `TensorOp::Load`)
    /// in the computation graph starting from this tensor.
    pub fn get_leaf_tensors(&self) -> Vec<Tensor> {
        let mut leafs = FxHashMap::default();
        let mut visited = FxHashSet::default();
        self.collect_leaf_tensors_recursive(&mut leafs, &mut visited);
        leafs.into_values().collect()
    }

    fn collect_leaf_tensors_recursive(
        &self,
        leafs: &mut FxHashMap<*const Tensor_, Tensor>,
        visited: &mut FxHashSet<*const Tensor_>,
    ) {
        let ptr = Rc::as_ptr(&self.0);
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        if let TensorOp::Load = self.op {
            leafs.insert(ptr, self.clone());
        }

        for src in &self.src {
            src.collect_leaf_tensors_recursive(leafs, visited);
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Add, self, rhs)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-rhs)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Mul, self, rhs)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
        self * &rhs.recip()
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        self.lazy_unary_op(Op::Neg)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl ToDot for Tensor {
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

fn build_dot_tensor(tensor: &Tensor, dot: &mut String, visited: &mut FxHashSet<*const Tensor_>) {
    let ptr = Rc::as_ptr(&tensor.0);
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    let label = format!(
        "op: {:?}\nshape: {:?}\ndtype: {:?}",
        tensor.op,
        tensor.shape(),
        tensor.dtype
    )
    .replace('\n', "\\n");
    dot.push_str(&format!("  \"{ptr:p}\" [label=\"{label}\"];\n"));

    for src in &tensor.src {
        let src_ptr = Rc::as_ptr(&src.0);
        dot.push_str(&format!("  \"{src_ptr:p}\" -> \"{ptr:p}\";\n"));
        build_dot_tensor(src, dot, visited);
    }
}

fn tensor_from_array<T: Clone>(arr: ArrayD<T>, dtype: DType) -> Tensor {
    let shape = arr.shape().to_vec();
    let backend = context::backend("clang");
    let buffer = backend.alloc(arr.len() * std::mem::size_of::<T>(), backend.clone());
    unsafe {
        std::ptr::copy_nonoverlapping(
            arr.as_ptr(),
            backend.get_buffer_ptr(buffer.id) as *mut T,
            arr.len(),
        );
    }
    let tracker = ShapeTracker::new(shape);
    let tensor = Tensor::new(TensorOp::Load, vec![], tracker, dtype, backend);
    *tensor.0.realized.borrow_mut() = Some(buffer);
    tensor
}

fn array_from_tensor<T: Clone>(tensor: &Tensor) -> ArrayD<T> {
    let buffer = tensor.realize();
    let ptr = tensor.backend.get_buffer_ptr(buffer.id) as *const T;
    let len = tensor.shape().iter().product();
    let data = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    ArrayD::from_shape_vec(tensor.shape().to_vec(), data).unwrap()
}

impl From<ArrayD<f32>> for Tensor {
    fn from(arr: ArrayD<f32>) -> Self {
        tensor_from_array(arr, DType::F32)
    }
}

impl From<Tensor> for ArrayD<f32> {
    fn from(tensor: Tensor) -> Self {
        assert_eq!(tensor.dtype, DType::F32, "DType mismatch");
        array_from_tensor(&tensor)
    }
}

impl From<ArrayD<i64>> for Tensor {
    fn from(arr: ArrayD<i64>) -> Self {
        tensor_from_array(arr, DType::I64)
    }
}

impl From<Tensor> for ArrayD<i64> {
    fn from(tensor: Tensor) -> Self {
        assert_eq!(tensor.dtype, DType::I64, "DType mismatch");
        array_from_tensor(&tensor)
    }
}
