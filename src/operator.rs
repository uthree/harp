use std::fmt::Debug;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Operator {
    // Input node
    Input,

    // Unary Operators
    Recip,
    Sin,
    Exp2,
    Log2,
    Sqrt,

    // Binary Operators
    Add,
    Mul,
    Rem,
    LessThan,

    // Dimensional Operators
    SumReduce(usize),
    MaxReduce(usize),
    Contiguous,
}
