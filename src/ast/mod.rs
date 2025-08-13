#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,   // float
    Usize, // size_t

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    F32(f32),
    Usize(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstOp {
    Const(Const),
    Add,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AstNode {
    op: AstOp,
    args: Vec<AstNode>,
    dtype: DType,
}
