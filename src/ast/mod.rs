use crate::graph::shape::Expr as ShapeExpr;
pub mod pattern;

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,   // float
    Usize, // size_t
    Isize, // ssize_t
    Void,

    Ptr(Box<Self>, ShapeExpr), // pointer
    Vec(Box<Self>, usize),     // fixed-size array (for SIMD vectorization)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstLiteral {
    F32(f32),
    Usize(usize),
    Isize(isize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    Const(ConstLiteral), // constant value
    Var(String),         // get value from variable
    Cast(DType),         // convert another type

    // numeric ops
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    Neg(Box<Self>),
    Recip(Box<Self>),
    Sin(Box<Self>),
    Sqrt(Box<Self>),
    Log2(Box<Self>),
    Exp2(Box<Self>),
    CallFunction(Vec<Self>),

    // statements
    Range {
        // Forループ
        counter_name: String, // ループカウンタの変数名
        max: Box<Self>,       // ループ回数
        body: Vec<Self>,
    },

    Declare {
        name: String,
        dtype: DType,
        constant: bool,
    }, // declare new (local) variable

    Drop(String), // drop (local) variable explicitly

    Barrier,

    // for pattern matching
    Capture(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    name: String,
    body: Vec<AstNode>,
    // TODO: arguments, return values
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    functions: Vec<Function>,
    entry_point: String,
}
