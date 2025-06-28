use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Operator {
    // Input node
    Input,

    // unary
    Recip,
    Sin,
    Log2,
    Exp2,
    Sqrt,

    // binary
    Add,
    Mul,
    Mod,
    LessThan,

    // dimensional
    Sum(usize),
    CumSum(usize),
    Prod(usize),
    CumProd(usize),
    Max(usize),
    ArgMax(usize),

    // shape control
    Contiguous,
}

#[derive(Debug, Clone)]
pub struct Node {
    operator: Operator,
}

#[derive(Debug, Clone)]
pub struct Graph {}
