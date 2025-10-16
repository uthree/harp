mod helper;

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
    Neg(Box<AstNode>),
    Recip(Box<AstNode>),
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
    Vec(Box<DType>, usize), // fixed-size vector
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Isize(isize),
    Usize(usize),
    F32(f32),
    Bool(bool),
}
