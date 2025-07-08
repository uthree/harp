use crate::tensor::TensorData;
use crate::dtype::DType;
use std::fmt::Debug;

/// Base trait for all operators in the computation graph.
///
/// Every operation that can be a node in the graph must implement this trait.
/// It requires `Debug` for logging and `Send + Sync` for thread safety.
pub trait Operator: Debug + Send + Sync {
    /// Returns a reference to `self` as a `dyn Any` trait object.
    /// This allows downcasting to concrete operator types.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Returns the data type of the output tensor produced by this operator.
    fn output_dtype(&self) -> DType;
}

// --- Sub-traits for categorization ---

/// Marker trait for special operators that don't fit into other categories.
pub trait SpecialOp: Operator {}

/// Marker trait for unary operators (operations with one input).
pub trait UnaryOp: Operator {}

/// Marker trait for binary operators (operations with two inputs).
pub trait BinaryOp: Operator {}

/// Trait for reduction operators that operate along a specific dimension.
pub trait ReduceOp: Operator {
    /// Returns the dimension along which the reduction is performed.
    fn dim(&self) -> usize;
}

/// Marker trait for movement operators that change the layout or view of a tensor.
pub trait MovementOp: Operator {}

// --- Operator Implementations ---

// --- Special Operators ---
/// Represents an input node in the computation graph.
/// This operator does not perform any computation but serves as a placeholder for external data.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub dtype: DType,
}
impl Operator for Input {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
}
impl SpecialOp for Input {}

/// Represents a constant value in the computation graph.
/// This operator holds a fixed `TensorData` value.
#[derive(Debug, Clone)]
pub struct Const {
    /// The constant tensor data.
    pub data: TensorData,
}

impl Operator for Const {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        self.data.dtype
    }
}

/// Represents a type casting operation.
#[derive(Debug, Clone, Copy)]
pub struct Cast {
    pub dtype: DType,
}

impl Operator for Cast {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
}

// --- Unary Operators ---

/// Represents the base-2 exponential function (2^x).
#[derive(Debug, Default, Clone, Copy)]
pub struct Exp2;
impl Operator for Exp2 {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl UnaryOp for Exp2 {}

/// Represents the base-2 logarithm function (log2(x)).
#[derive(Debug, Default, Clone, Copy)]
pub struct Log2;
impl Operator for Log2 {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl UnaryOp for Log2 {}

/// Represents the sine function (sin(x)).
#[derive(Debug, Default, Clone, Copy)]
pub struct Sin;
impl Operator for Sin {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl UnaryOp for Sin {}

/// Represents the square root function (sqrt(x)).
#[derive(Debug, Default, Clone, Copy)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl UnaryOp for Sqrt {}

/// Represents the reciprocal function (1/x).
#[derive(Debug, Default, Clone, Copy)]
pub struct Recip;
impl Operator for Recip {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl UnaryOp for Recip {}

// --- Binary Operators ---

/// Represents the addition operation (x + y).
#[derive(Debug, Default, Clone, Copy)]
pub struct Add;
impl Operator for Add {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl BinaryOp for Add {}

/// Represents the multiplication operation (x * y).
#[derive(Debug, Default, Clone, Copy)]
pub struct Mul;
impl Operator for Mul {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl BinaryOp for Mul {}

/// Represents the remainder operation (x % y).
#[derive(Debug, Default, Clone, Copy)]
pub struct Rem;
impl Operator for Rem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl BinaryOp for Rem {}

/// Represents the less than comparison operation (x < y).
/// Returns 1.0 if true, 0.0 if false.
#[derive(Debug, Default, Clone, Copy)]
pub struct LessThan;
impl Operator for LessThan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl BinaryOp for LessThan {}

// --- Reduce Operators ---

/// Represents a sum reduction operation along a specified dimension.
#[derive(Debug, Clone, Copy)]
pub struct SumReduce {
    /// The dimension along which the sum reduction is performed.
    pub dim: usize,
}
impl Operator for SumReduce {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl ReduceOp for SumReduce {
    fn dim(&self) -> usize {
        self.dim
    }
}

/// Represents a maximum reduction operation along a specified dimension.
#[derive(Debug, Clone, Copy)]
pub struct MaxReduce {
    /// The dimension along which the maximum reduction is performed.
    pub dim: usize,
}
impl Operator for MaxReduce {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl ReduceOp for MaxReduce {
    fn dim(&self) -> usize {
        self.dim
    }
}

// --- Movement Operators ---

/// Represents a contiguous memory layout operation.
/// This operator ensures the tensor data is stored contiguously in memory.
#[derive(Debug, Default, Clone, Copy)]
pub struct Contiguous;
impl Operator for Contiguous {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_dtype(&self) -> DType {
        DType::F32 // Assuming float output for now
    }
}
impl MovementOp for Contiguous {}
