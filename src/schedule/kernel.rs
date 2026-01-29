//! Fused kernel representation.

use crate::dtype::{DType, ScalarValue};
use crate::ops::Ops;
use crate::shape::Shape;

/// Source for a fused operation.
#[derive(Debug, Clone)]
pub enum FusedSource {
    /// Input buffer at given index
    Input(usize),
    /// Result of a previous operation in the chain (by index)
    PrevOp(usize),
    /// A constant scalar value
    Constant(ScalarValue),
}

/// A single operation within a fused kernel.
#[derive(Debug, Clone)]
pub struct FusedOp {
    /// The operation type
    pub op: Ops,
    /// Sources for this operation
    pub sources: Vec<FusedSource>,
    /// Output dtype for this operation
    pub dtype: DType,
}

impl FusedOp {
    /// Create a new fused operation.
    pub fn new(op: Ops, sources: Vec<FusedSource>, dtype: DType) -> Self {
        Self { op, sources, dtype }
    }
}

/// Input specification for a fused kernel.
#[derive(Debug, Clone)]
pub struct KernelInput {
    /// Index in the buffer list
    pub buffer_index: usize,
    /// Data type of the input
    pub dtype: DType,
    /// Shape of the input
    pub shape: Shape,
}

impl KernelInput {
    /// Create a new kernel input.
    pub fn new(buffer_index: usize, dtype: DType, shape: Shape) -> Self {
        Self {
            buffer_index,
            dtype,
            shape,
        }
    }
}

/// A fused kernel represents multiple operations combined into a single GPU kernel.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// Unique name for this kernel
    pub name: String,
    /// Chain of operations to execute (in order)
    pub ops_chain: Vec<FusedOp>,
    /// Inputs to this kernel
    pub inputs: Vec<KernelInput>,
    /// Output shape
    pub output_shape: Shape,
    /// Output data type
    pub output_dtype: DType,
}

impl FusedKernel {
    /// Create a new fused kernel.
    pub fn new(
        name: String,
        ops_chain: Vec<FusedOp>,
        inputs: Vec<KernelInput>,
        output_shape: Shape,
        output_dtype: DType,
    ) -> Self {
        Self {
            name,
            ops_chain,
            inputs,
            output_shape,
            output_dtype,
        }
    }

    /// Returns true if this kernel has multiple fused operations.
    pub fn is_fused(&self) -> bool {
        self.ops_chain.len() > 1
    }

    /// Returns the number of elements in the output.
    pub fn output_numel(&self) -> usize {
        self.output_shape.numel()
    }
}
