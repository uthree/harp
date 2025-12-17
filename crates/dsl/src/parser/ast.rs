//! DSL Abstract Syntax Tree definitions

use harp::graph::DType;

/// A module containing multiple graph definitions
#[derive(Debug, Clone)]
pub struct DslModule {
    pub graphs: Vec<DslGraph>,
}

/// A graph definition
#[derive(Debug, Clone)]
pub struct DslGraph {
    /// Graph name
    pub name: String,
    /// Shape variables with default values: (name, default_value)
    /// All shape variables must have default values
    pub shape_vars: Vec<(String, isize)>,
    /// Input parameters
    pub inputs: Vec<DslParam>,
    /// Output parameters
    pub outputs: Vec<DslParam>,
    /// Body statements
    pub body: Vec<DslStatement>,
}

/// A parameter (input or output)
#[derive(Debug, Clone)]
pub struct DslParam {
    /// Parameter name
    pub name: String,
    /// Data type
    pub dtype: DslDType,
    /// Shape as expressions
    pub shape: Vec<ShapeExpr>,
}

/// Data type in DSL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DslDType {
    F32,
    I32,
    Bool,
}

impl From<DslDType> for DType {
    fn from(dtype: DslDType) -> Self {
        match dtype {
            DslDType::F32 => DType::F32,
            DslDType::I32 => DType::I32,
            DslDType::Bool => DType::Bool,
        }
    }
}

/// Shape expression (dimension size)
#[derive(Debug, Clone)]
pub enum ShapeExpr {
    /// Constant integer
    Const(isize),
    /// Variable reference
    Var(String),
    /// Binary operation
    BinOp {
        op: ShapeBinOp,
        lhs: Box<ShapeExpr>,
        rhs: Box<ShapeExpr>,
    },
}

/// Shape binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// Statement in graph body
#[derive(Debug, Clone)]
pub enum DslStatement {
    /// Simple assignment: x = expr
    Assign { name: String, value: DslExpr },
    /// Tuple assignment: (a, b) = expr
    /// Used for multi-output subgraph calls
    TupleAssign { names: Vec<String>, value: DslExpr },
    /// Return statement: return a or return a, b, c
    /// Required at the end of each graph to specify outputs
    Return { names: Vec<String> },
}

/// Expression
#[derive(Debug, Clone)]
pub enum DslExpr {
    /// Variable reference
    Var(String),
    /// Integer literal
    IntLit(i64),
    /// Float literal
    FloatLit(f64),
    /// Binary operation
    BinOp {
        op: DslBinOp,
        lhs: Box<DslExpr>,
        rhs: Box<DslExpr>,
    },
    /// Unary operation
    UnaryOp {
        op: DslUnaryOp,
        operand: Box<DslExpr>,
    },
    /// Method call: a.sum(1)
    MethodCall {
        receiver: Box<DslExpr>,
        method: String,
        args: Vec<DslArg>,
    },
    /// Function call: matmul(a, b)
    FunctionCall { name: String, args: Vec<DslArg> },
    /// Fused elementwise: fused(a, b, c) { a * b + c }
    FusedElementwise {
        inputs: Vec<String>,
        expr: Box<DslExpr>,
    },
    /// Fused reduce: fused_reduce(a, b, axis: 1, op: sum) { a * b }
    FusedReduce {
        inputs: Vec<String>,
        axis: usize,
        op: ReduceOpKind,
        expr: Box<DslExpr>,
    },
    /// Fused cumulative: fused_cumulative(a, axis: 0, op: sum) { a * a }
    FusedCumulative {
        inputs: Vec<String>,
        axis: usize,
        op: CumulativeOpKind,
        expr: Box<DslExpr>,
    },
    /// Array literal: [a, b, c]
    ArrayLit(Vec<DslExpr>),
    /// Index access: a[0]
    Index {
        base: Box<DslExpr>,
        indices: Vec<DslExpr>,
    },
}

/// Argument to function/method call
#[derive(Debug, Clone)]
pub enum DslArg {
    /// Positional argument
    Positional(DslExpr),
    /// Named argument: axis: 1
    Named { name: String, value: DslArgValue },
    /// Array argument: [a, b, c]
    Array(Vec<DslExpr>),
}

/// Value of a named argument
#[derive(Debug, Clone)]
pub enum DslArgValue {
    Expr(DslExpr),
    Ident(String),
    Array(Vec<DslExpr>),
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DslBinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DslUnaryOp {
    Neg,
    Not,
}

/// Reduce operation kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOpKind {
    Sum,
    Prod,
    Max,
}

/// Cumulative operation kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeOpKind {
    Sum,
    Prod,
}

impl DslExpr {
    /// Check if this expression is a simple variable reference
    pub fn as_var(&self) -> Option<&str> {
        match self {
            DslExpr::Var(name) => Some(name),
            _ => None,
        }
    }

    /// Check if this expression is an integer literal
    pub fn as_int(&self) -> Option<i64> {
        match self {
            DslExpr::IntLit(v) => Some(*v),
            _ => None,
        }
    }
}

impl ShapeExpr {
    /// Convert to Harp's Expr
    pub fn to_harp_expr(&self) -> harp::graph::shape::Expr {
        use harp::graph::shape::Expr;
        match self {
            ShapeExpr::Const(v) => Expr::Const(*v),
            ShapeExpr::Var(name) => Expr::Var(name.clone()),
            ShapeExpr::BinOp { op, lhs, rhs } => {
                let l = lhs.to_harp_expr();
                let r = rhs.to_harp_expr();
                match op {
                    ShapeBinOp::Add => l + r,
                    ShapeBinOp::Sub => l - r,
                    ShapeBinOp::Mul => l * r,
                    ShapeBinOp::Div => l / r,
                    ShapeBinOp::Rem => l % r,
                }
            }
        }
    }
}
