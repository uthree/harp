use crate::node::{self, Node};
use crate::op::{Load, OpAdd, OpDiv, OpMul, OpSub, Operator, Reduce};
use dyn_clone::clone_box;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::Arc;

/// A multi-dimensional array.
///
/// `Tensor` is a lightweight, reference-counted wrapper around the core
/// computation graph (`TensorData`). Cloning a `Tensor` is cheap.
#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<TensorData>,
}

/// The internal representation of a `Tensor`'s computation.
///
/// It holds the operator that produces the tensor's value and a list
/// of source (input) tensors.
pub struct TensorData {
    pub op: Box<dyn Operator>,
    pub src: Vec<Tensor>,
}

/// Tracks the shape and indexing of a `Tensor`.
///
/// This is crucial for compiling the high-level tensor graph into a
/// scalar `Node` graph. It resolves multi-dimensional indexing into
/// linear memory offsets.
pub struct ShapeTracker {
    /// The size of each dimension (e.g., `[4, 3]` for a 4x3 matrix).
    pub dims: Vec<Rc<Node>>,
    /// The mathematical expression to convert a multi-dimensional index
    /// into a linear memory offset.
    pub index_expr: Vec<Rc<Node>>,
}

impl Tensor {
    /// Creates a new "leaf" tensor that represents loading data.
    pub fn new_load() -> Self {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(Load),
                src: vec![],
            }),
        }
    }

    /// Creates a new `Reduce` tensor.
    pub fn reduce(self, op: impl Operator + 'static, axis: usize) -> Self {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(Reduce {
                    op: Box::new(op),
                    axis,
                }),
                src: vec![self],
            }),
        }
    }

    /// Performs a sum reduction along a specified axis.
    pub fn sum(self, axis: usize) -> Self {
        self.reduce(OpAdd, axis)
    }

    /// Compiles the tensor's computation graph into a traditional Node graph.
    /// This process resolves tensor indexing into scalar operations.
    pub fn compile(&self, shape_tracker: &ShapeTracker) -> Rc<Node> {
        let op = &self.data.op;
        let src = &self.data.src;

        match op.name() {
            "Load" => {
                // For a Load op, we use the ShapeTracker's index expression.
                // This expression calculates the final memory offset.
                // We assume the expression is a single node for now.
                assert_eq!(shape_tracker.index_expr.len(), 1, "Complex index expressions not yet supported");
                let index_node = &shape_tracker.index_expr[0];
                Rc::new(node::Node::from(Arc::new(node::NodeData {
                    op: Box::new(Load),
                    src: vec![(**index_node).clone()],
                })))
            }
            "OpAdd" | "OpSub" | "OpMul" | "OpDiv" => {
                // For binary ops, compile the sources recursively and combine them.
                let left = src[0].compile(shape_tracker);
                let right = src[1].compile(shape_tracker);
                let new_op = clone_box(&**op); // Clone the operator
                Rc::new(node::Node::from(Arc::new(node::NodeData {
                    op: new_op,
                    src: vec![(*left).clone(), (*right).clone()],
                })))
            }
            _ => todo!("Compile not implemented for op: {}", op.name()),
        }
    }
}

// --- Operator Overloads ---

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpAdd),
                src: vec![self, rhs],
            }),
        }
    }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpSub),
                src: vec![self, rhs],
            }),
        }
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpMul),
                src: vec![self, rhs],
            }),
        }
    }
}

impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpDiv),
                src: vec![self, rhs],
            }),
        }
    }
}