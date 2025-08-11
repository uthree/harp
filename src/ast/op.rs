use crate::ast::dtype::{Const, DType};
use std::hash::{Hash, Hasher};

/// Represents an operation in the Abstract Syntax Tree.
///
/// The philosophy here is that `AstOp` should represent the operation itself,
/// while the operands (source nodes) are consistently stored in the `src`
/// field of the `AstNode`. This makes the tree structure more uniform.
///
/// Also, based on the design philosophy that the number of types of operator sets should be as small as possible,
/// for example, subtraction (a-b) is expressed as (a+neg(b)).
#[derive(Debug, Clone)]
pub enum AstOp {
    // --- Placeholders ---
    /// A placeholder for pattern matching in graph rewriting.
    Capture(usize, DType),

    // --- Literals and Variables ---
    /// A constant value.
    Const(Const),
    /// A variable, identified by its name.
    Var(String),

    // --- Unary Operations ---
    Neg,
    Recip,
    Sin,
    Sqrt,
    Log2,
    Exp2,
    /// Casts a value to a different data type. `src[0]` is the value to cast.
    Cast(DType),

    // --- Binary Operations ---
    Add,
    Sub,
    Mul,
    Max,
    Rem,
    LessThan,

    // --- Other Operators ---
    /// Packs multiple values into a tuple.
    Pack,
    /// Takes the n-th element from a tuple.
    Index(usize),

    // --- Statements and Control Flow ---
    /// A collection of top-level function definitions.
    Program,
    /// A block of statements. The statements are stored in the `src` field of the `AstNode`.
    Block,
    /// Assigns a value. `src[0]` is the destination, `src[1]` is the source.
    Assign,
    /// Declares a variable and assigns a value to it. `src[0]` is the value.
    Declare {
        name: String,
        dtype: DType,
    },
    /// Stores a value at a memory location. `src[0]` is the destination address, `src[1]` is the value.
    Store,
    /// Dereferences a pointer. `src[0]` is the address to dereference.
    Deref,
    /// Represents an indexed access into a buffer. `src[0]` is the buffer, `src[1]` is the index.
    BufferIndex,
    /// Represents a for-loop. `src[0]` is the maximum value (exclusive), and the rest of `src` is the loop body.
    Range {
        loop_var: String,
    },
    /// Represents a function definition. The function body is in `src`.
    Func {
        name: String,
        args: Vec<(String, DType)>,
    },
    /// Represents a function call. Arguments are stored in the `src` field.
    Call(String),
}

impl PartialEq for AstOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Capture(l0, l1), Self::Capture(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Const(l0), Self::Const(r0)) => l0 == r0,
            (Self::Var(l0), Self::Var(r0)) => l0 == r0,
            (Self::Cast(l0), Self::Cast(r0)) => l0 == r0,
            (Self::Index(l0), Self::Index(r0)) => l0 == r0,
            (
                Self::Declare {
                    name: l_name,
                    dtype: l_dtype,
                },
                Self::Declare {
                    name: r_name,
                    dtype: r_dtype,
                },
            ) => l_name == r_name && l_dtype == r_dtype,
            (
                Self::Range {
                    loop_var: l_loop_var,
                },
                Self::Range {
                    loop_var: r_loop_var,
                },
            ) => l_loop_var == r_loop_var,
            (
                Self::Func {
                    name: l_name,
                    args: l_args,
                },
                Self::Func {
                    name: r_name,
                    args: r_args,
                },
            ) => l_name == r_name && l_args == r_args,
            (Self::Call(l0), Self::Call(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Eq for AstOp {}

impl Hash for AstOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            AstOp::Capture(id, dtype) => {
                id.hash(state);
                dtype.hash(state);
            }
            AstOp::Const(c) => c.hash(state),
            AstOp::Var(name) => name.hash(state),
            AstOp::Cast(dtype) => dtype.hash(state),
            AstOp::Index(i) => i.hash(state),
            AstOp::Declare { name, dtype } => {
                name.hash(state);
                dtype.hash(state);
            }
            AstOp::Range { loop_var } => loop_var.hash(state),
            AstOp::Func { name, args } => {
                name.hash(state);
                args.hash(state);
            }
            AstOp::Call(name) => name.hash(state),
            // Ops with no data
            _ => {}
        }
    }
}
