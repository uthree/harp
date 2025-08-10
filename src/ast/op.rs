use crate::ast::dtype::{Const, DType};

/// Represents an operation in the Abstract Syntax Tree.
///
/// The philosophy here is that `AstOp` should represent the operation itself,
/// while the operands (source nodes) are consistently stored in the `src`
/// field of the `AstNode`. This makes the tree structure more uniform.
///
/// Also, based on the design philosophy that the number of types of operator sets should be as small as possible,
/// for example, subtraction (a-b) is expressed as (a+neg(b)).
#[derive(Debug, Clone, PartialEq)]
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
    Cos,
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
