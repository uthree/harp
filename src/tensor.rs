//! The core `Tensor` struct and related components.
//!
//! This module defines the central data structure of the library, `Tensor`, which
//! represents a multi-dimensional array. All operations on `Tensor`s are lazy,
//! meaning they build up a computation graph rather than executing immediately.
//!
//! The actual computation is triggered by calling the `.realize()` method, which
//! hands off the generated computation graph to the appropriate `Backend`.

use crate::backends::{Backend, Buffer};
use crate::dot::ToDot;
use crate::dtype::DType;
use crate::linearizer::Linearizer;
use crate::optimizer::Optimizer;
use crate::shapetracker::ShapeTracker;
use crate::uop::{Op, UOp};
use log::debug;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;

/// Represents the operation that created a `Tensor`.
///
/// This enum tracks the origin of a `Tensor` within the computation graph.
#[derive(Debug, Clone)]
pub enum TensorOp {
    /// A source tensor, typically representing data loaded from memory.
    Load,
    /// A tensor created as the result of a binary operation.
    Binary(Op),
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
                let size: usize =
                    self.0.tracker.shape().iter().product::<usize>() * self.0.dtype.size();
                debug!("Allocating new buffer for Load op with size: {size}");
                self.0.backend.alloc(size, self.0.backend.clone())
            }
            TensorOp::Binary(_) => {
                let args: Vec<_> = self.0.src.iter().map(|t| t.realize()).collect();
                let size: usize =
                    self.0.tracker.shape().iter().product::<usize>() * self.0.dtype.size();
                let output_buffer = self.0.backend.alloc(size, self.0.backend.clone());
                let mut kernel_args = args;
                kernel_args.push(output_buffer.clone());

                let uop_graph = self.to_uop();
                debug!("Generated UOp graph: {uop_graph:?}");

                let optimizer = Optimizer::new();
                let optimized_uop_graph = optimizer.optimize(&uop_graph);
                debug!("Optimized UOp graph: {optimized_uop_graph:?}");

                let mut linearizer = Linearizer::new();
                let kernel = linearizer.linearize(&optimized_uop_graph, self.0.tracker.shape());
                let args_ref: Vec<&Buffer> = kernel_args.iter().collect();
                self.0.backend.compile_and_exec(&kernel, &args_ref);
                output_buffer
            }
        };

        *self.0.realized.borrow_mut() = Some(result_var.clone());
        result_var
    }

    fn to_uop(&self) -> UOp {
        fn build_uop_graph(
            tensor: &Tensor,
            arg_map: &mut std::collections::HashMap<*const Tensor_, UOp>,
            loop_var: &UOp,
        ) -> UOp {
            if let Some(uop) = arg_map.get(&Rc::as_ptr(&tensor.0)) {
                return uop.clone();
            }

            let uop = match &tensor.0.op {
                TensorOp::Load => {
                    let buffer =
                        UOp::var(&format!("data{}", arg_map.len()), tensor.0.dtype.clone());
                    let idx = tensor.0.tracker.expr_node(loop_var);
                    UOp::new(Op::Load, tensor.0.dtype.clone(), vec![buffer, idx])
                }
                TensorOp::Binary(op) => {
                    let lhs = build_uop_graph(&tensor.0.src[0], arg_map, loop_var);
                    let rhs = build_uop_graph(&tensor.0.src[1], arg_map, loop_var);
                    UOp::new(op.clone(), tensor.0.dtype.clone(), vec![lhs, rhs])
                }
            };
            arg_map.insert(Rc::as_ptr(&tensor.0), uop.clone());
            uop
        }

        let mut arg_map = std::collections::HashMap::new();
        let loop_var = UOp::var("i", DType::U64);
        let result_expr = build_uop_graph(self, &mut arg_map, &loop_var);

        let out_idx = arg_map.len();
        let output_buffer = UOp::var(&format!("data{out_idx}"), self.0.dtype.clone());
        let idx = self.0.tracker.expr_node(&loop_var);
        

        UOp::new(
            Op::Store,
            DType::Unit,
            vec![output_buffer, idx, result_expr],
        )
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

    fn lazy_binary_op(op: Op, a: &Self, b: &Self) -> Self {
        assert!(Rc::ptr_eq(&a.0.backend, &b.0.backend));
        Self::new(
            TensorOp::Binary(op),
            vec![a.clone(), b.clone()],
            a.0.tracker.clone(),
            a.0.dtype.clone(),
            a.0.backend.clone(),
        )
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
        tensor.0.op,
        tensor.0.tracker.shape(),
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
