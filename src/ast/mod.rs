mod helper;
mod op;

pub use helper::*;

#[derive(Debug, Clone)]
pub struct AstNode {
    op: AstOp,
    dtype: DType,
}

#[derive(Debug, Clone)]
pub enum AstOp {
    Const(ConstValue),
    Add(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Max(Box<AstNode>, Box<AstNode>),
    Idiv(Box<AstNode>, Box<AstNode>), // integer division
    Rem(Box<AstNode>, Box<AstNode>),  // remainder
    Neg(Box<AstNode>),
    Recip(Box<AstNode>),
    Sqrt(Box<AstNode>),
    Sin(Box<AstNode>),
    Log2(Box<AstNode>),
    Exp2(Box<AstNode>),

    // TODO: add fields
    Load,
    Store,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Unknown, // unknown type, placeholder
    None,    // void
    Isize,   // signed integer
    Usize,   // unsigned integer
    F32,     // 32-bit float
    Bool,    // bool,

    Ptr(Box<DType>),        // pointer of some type
    Vec(Box<DType>, usize), // fixed-size vector for simd operation
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Isize(isize),
    Usize(usize),
    F32(f32),
    Bool(bool),
}
