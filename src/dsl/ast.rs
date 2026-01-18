//! DSL Abstract Syntax Tree types

use std::fmt;

/// A complete DSL program containing multiple graph definitions
#[derive(Debug, Clone)]
pub struct DslProgram {
    pub graphs: Vec<GraphDef>,
}

/// A single graph definition
#[derive(Debug, Clone)]
pub struct GraphDef {
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub return_type: TypeSpec,
    pub body: Vec<Statement>,
    pub return_expr: DslExpr,
}

/// Parameter declaration
#[derive(Debug, Clone)]
pub struct ParamDecl {
    pub name: String,
    pub type_spec: TypeSpec,
}

/// Type specification: dtype + shape
#[derive(Debug, Clone)]
pub struct TypeSpec {
    pub dtype: DType,
    pub shape: Vec<ShapeDim>,
}

/// Data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

/// Shape dimension: either static (concrete value) or dynamic (named)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim {
    Static(i64),
    Dynamic(String),
}

impl fmt::Display for ShapeDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeDim::Static(n) => write!(f, "{}", n),
            ShapeDim::Dynamic(name) => write!(f, "{}", name),
        }
    }
}

/// Statement (only let binding for now)
#[derive(Debug, Clone)]
pub enum Statement {
    Let { name: String, value: DslExpr },
}

/// DSL expression
#[derive(Debug, Clone)]
pub enum DslExpr {
    /// Variable reference
    Var(String),
    /// Numeric literal
    Literal(f64),
    /// Binary operation
    BinaryOp {
        op: BinOp,
        lhs: Box<DslExpr>,
        rhs: Box<DslExpr>,
    },
    /// Unary operation
    UnaryOp {
        op: UnaryOp,
        operand: Box<DslExpr>,
    },
    /// Function call
    FuncCall {
        name: String,
        args: Vec<FuncArg>,
    },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    // Comparison
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
        }
    }
}

/// Function argument
#[derive(Debug, Clone)]
pub enum FuncArg {
    /// Positional expression argument
    Positional(DslExpr),
    /// Named argument with expression value
    Named { name: String, value: DslExpr },
    /// Named argument with shape value
    NamedShape { name: String, shape: Vec<ShapeDim> },
    /// Positional shape argument (e.g., expand(x, [32, 64]))
    Shape(Vec<ShapeDim>),
}
