//! Operation definitions for the computation graph.

use std::fmt;

/// All supported operations in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ops {
    // Buffer operations
    Load,
    Store,
    Const,

    // Unary operations
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Recip,

    // Binary operations
    Add,
    Sub,
    Mul,
    Div,
    Max,
    CmpLt,
    CmpEq,

    // Ternary operations
    Where,

    // Reduce operations
    Sum,
    ReduceMax,

    // Movement operations
    Reshape,
    Expand,
    Permute,
    Pad,
    Shrink,
    Stride,

    // Cast operation
    Cast,
}

/// Reduce operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
}

impl fmt::Display for Ops {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ops::Load => write!(f, "LOAD"),
            Ops::Store => write!(f, "STORE"),
            Ops::Const => write!(f, "CONST"),
            Ops::Neg => write!(f, "NEG"),
            Ops::Exp => write!(f, "EXP"),
            Ops::Log => write!(f, "LOG"),
            Ops::Sqrt => write!(f, "SQRT"),
            Ops::Sin => write!(f, "SIN"),
            Ops::Cos => write!(f, "COS"),
            Ops::Recip => write!(f, "RECIP"),
            Ops::Add => write!(f, "ADD"),
            Ops::Sub => write!(f, "SUB"),
            Ops::Mul => write!(f, "MUL"),
            Ops::Div => write!(f, "DIV"),
            Ops::Max => write!(f, "MAX"),
            Ops::CmpLt => write!(f, "CMPLT"),
            Ops::CmpEq => write!(f, "CMPEQ"),
            Ops::Where => write!(f, "WHERE"),
            Ops::Sum => write!(f, "SUM"),
            Ops::ReduceMax => write!(f, "REDUCEMAX"),
            Ops::Reshape => write!(f, "RESHAPE"),
            Ops::Expand => write!(f, "EXPAND"),
            Ops::Permute => write!(f, "PERMUTE"),
            Ops::Pad => write!(f, "PAD"),
            Ops::Shrink => write!(f, "SHRINK"),
            Ops::Stride => write!(f, "STRIDE"),
            Ops::Cast => write!(f, "CAST"),
        }
    }
}

impl Ops {
    /// Returns true if this is a unary operation.
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            Ops::Neg | Ops::Exp | Ops::Log | Ops::Sqrt | Ops::Sin | Ops::Cos | Ops::Recip
        )
    }

    /// Returns true if this is a binary operation.
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Ops::Add | Ops::Sub | Ops::Mul | Ops::Div | Ops::Max | Ops::CmpLt | Ops::CmpEq
        )
    }

    /// Returns true if this is a reduce operation.
    pub fn is_reduce(&self) -> bool {
        matches!(self, Ops::Sum | Ops::ReduceMax)
    }

    /// Returns true if this is a movement operation.
    pub fn is_movement(&self) -> bool {
        matches!(
            self,
            Ops::Reshape | Ops::Expand | Ops::Permute | Ops::Pad | Ops::Shrink | Ops::Stride
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ops_classification() {
        assert!(Ops::Neg.is_unary());
        assert!(Ops::Add.is_binary());
        assert!(Ops::Sum.is_reduce());
        assert!(Ops::Reshape.is_movement());
    }
}
