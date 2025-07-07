use crate::tensor::TensorData;
use std::fmt::Debug;

// Base trait for all operators
pub trait Operator: Debug + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
}

// --- Sub-traits for categorization ---

// Special operators
pub trait SpecialOp: Operator {}

// Unary operators
pub trait UnaryOp: Operator {}

// Binary operators
pub trait BinaryOp: Operator {}

// Reduce operators that act on dimensions
pub trait ReduceOp: Operator {
    fn dim(&self) -> usize;
}

// Movement operators
pub trait MovementOp: Operator {}

// --- Operator Implementations ---

// --- Special Operators ---
#[derive(Debug, Default, Clone, Copy)]
pub struct Input;
impl Operator for Input {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl SpecialOp for Input {}

// --- Unary Operators ---

#[derive(Debug, Default, Clone, Copy)]
pub struct Exp2;
impl Operator for Exp2 {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl UnaryOp for Exp2 {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Log2;
impl Operator for Log2 {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl UnaryOp for Log2 {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Sin;
impl Operator for Sin {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl UnaryOp for Sin {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl UnaryOp for Sqrt {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Recip;
impl Operator for Recip {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl UnaryOp for Recip {}

// --- Binary Operators ---

#[derive(Debug, Default, Clone, Copy)]
pub struct Add;
impl Operator for Add {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl BinaryOp for Add {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Mul;
impl Operator for Mul {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl BinaryOp for Mul {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Rem;
impl Operator for Rem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl BinaryOp for Rem {}

#[derive(Debug, Default, Clone, Copy)]
pub struct LessThan;
impl Operator for LessThan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl BinaryOp for LessThan {}

// --- Reduce Operators ---

#[derive(Debug, Clone, Copy)]
pub struct SumReduce {
    pub dim: usize,
}
impl Operator for SumReduce {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl ReduceOp for SumReduce {
    fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaxReduce {
    pub dim: usize,
}
impl Operator for MaxReduce {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl ReduceOp for MaxReduce {
    fn dim(&self) -> usize {
        self.dim
    }
}

// --- Movement Operators ---

#[derive(Debug, Default, Clone, Copy)]
pub struct Contiguous;
impl Operator for Contiguous {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl MovementOp for Contiguous {}

#[derive(Debug, Clone)]
pub struct Const {
    pub data: TensorData,
}

impl Operator for Const {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
