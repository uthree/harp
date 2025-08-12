//! Provides the core `Tensor` struct and its associated operations.
//!
//! This module defines a lazily-evaluated tensor object that builds a computation
//! graph under the hood. The actual computation is deferred until the `forward()`
//! method is called. This allows for graph-level optimizations before execution.

pub mod creation;
mod grad;
pub mod op_conversion;
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
use std::sync::atomic::{AtomicUsize, Ordering};

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
static C_BACKEND: Lazy<CBackend> = Lazy::new(CBackend::new);

/// A buffer holding the tensor's actual data on a computation device.
#[derive(Debug, Clone)]
pub enum TensorBuffer {
    /// A buffer managed by the C backend.
    C(CBuffer),
}

/// Represents the computation backend where tensor operations are executed.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorBackend {
    /// The C backend, which compiles and runs operations as C code.
    C,
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

impl TensorData {
    pub fn new(
        op: TensorOp,
        src: Vec<Tensor>,
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
        backend: TensorBackend,
    ) -> Self {
        Self {
            id: 0, // Will be set by From<TensorData>
            op,
            src,
            shape,
            dtype,
            buffer: None,
            grad: None,
            requires_grad,
            backend,
        }
    }
}

/// Contains the internal data and metadata for a `Tensor`.
#[derive(Debug)]
pub struct TensorData {
    pub id: usize,
    pub op: TensorOp,
    pub src: Vec<Tensor>,
    pub shape: Shape,
    pub dtype: DType,
    pub buffer: Option<TensorBuffer>,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
    pub backend: TensorBackend,
}

/// A type alias for the shape of a tensor, represented as a vector of dimensions.
pub type Shape = Vec<usize>;

/// The primary tensor structure.
#[derive(Clone, Debug)]
pub struct Tensor(pub Rc<RefCell<TensorData>>);

impl Tensor {
    pub fn id(&self) -> usize {
        self.0.borrow().id
    }

    pub fn forward(&self) {
        // If buffer is already computed, do nothing.
        if self.0.borrow().buffer.is_some() {
            return;
        }

        // Build the computation graph from the final tensor.
        let (graph, tensor_to_node) = Graph::from_tensor(self);

        // This logic is simplified. A robust implementation would handle user-provided inputs.
        let input_buffers = vec![];

        // Execute the entire graph.
        let result_buffers = C_BACKEND.execute(&graph, input_buffers, vec![]);

        // Assign the resulting buffers back to all tensors in the graph.
        let all_tensors = self.all_tensors();
        for tensor in all_tensors {
            if let Some(node_id) = tensor_to_node.get(&tensor.id())
                && let Some(buffer) = result_buffers.get(node_id) {
                    tensor.0.borrow_mut().buffer = Some(TensorBuffer::C(buffer.clone()));
                }
        }
    }

    // Helper function to collect all unique tensors in the computation graph.
    fn all_tensors(&self) -> Vec<Tensor> {
        let mut visited = HashSet::new();
        let mut all = vec![];
        let mut queue = vec![self.clone()];
        visited.insert(self.id());

        while let Some(tensor) = queue.pop() {
            all.push(tensor.clone());
            for src in &tensor.0.borrow().src {
                if visited.insert(src.id()) {
                    queue.push(src.clone());
                }
            }
        }
        all
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
            self.0.as_ptr() as *const TensorData,
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
    fn from(mut value: TensorData) -> Self {
        // Assign a unique ID to every tensor created.
        if value.id == 0 {
            value.id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        }
        Tensor(Rc::new(RefCell::new(value)))
    }
}

pub fn backend(name: &str) -> TensorBackend {
    match name {
        "c" => TensorBackend::C,
        _ => panic!("Unsupported backend: {}", name),
    }
}
