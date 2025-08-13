use std::cell::Cell;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::ast::{dtype::DType, op::AstOp};

thread_local! {
    static NEXT_ID: Cell<usize> = const { Cell::new(0) };
}

/// Generates a unique ID for each `AstNode`.
fn next_id() -> usize {
    NEXT_ID.with(|cell| {
        let id = cell.get();
        cell.set(id + 1);
        id
    })
}

/// The fundamental building block of the Abstract Syntax Tree.
///
/// An `AstNode` represents a single operation, value, or statement in the
/// computation. It has a unique ID, an operation type (`Op`), a list of
/// source nodes (`src`), and a data type (`dtype`).
///
/// # Examples
///
/// ```
/// use harp::ast::{AstNode, DType};
///
/// // Create an AST for `a + 1.0`
/// let a = AstNode::var("a").with_type(DType::F32);
/// let b: AstNode = 1.0f32.into();
/// let c = a + b;
///
/// // The resulting AST node represents the addition.
/// assert_eq!(c.op, harp::ast::AstOp::Add);
/// assert_eq!(c.src.len(), 2);
/// ```
#[derive(Clone)]
pub struct AstNode {
    pub id: usize,
    pub op: AstOp,
    pub src: Vec<AstNode>,
    pub dtype: DType,
}

impl fmt::Debug for AstNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A format that doesn't include the id, to be used for hashing/keys.
        // For a more detailed debug view, we might need another method.
        f.debug_struct("AstNode")
            .field("op", &self.op)
            .field("src", &self.src)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl PartialEq for AstNode {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.src == other.src && self.dtype == other.dtype
    }
}

impl Eq for AstNode {}

impl Hash for AstNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        self.src.hash(state);
        self.dtype.hash(state);
    }
}

impl AstNode {
    /// Creates a new `AstNode`.
    pub fn new(op: AstOp, src: Vec<AstNode>, dtype: DType) -> Self {
        Self {
            id: next_id(),
            op,
            src,
            dtype,
        }
    }

    /// Creates a new `Capture` node for pattern matching.
    pub fn capture(id: usize, dtype: DType) -> Self {
        Self::new(AstOp::Capture(id, dtype.clone()), vec![], dtype)
    }

    /// Creates a new `Var` node.
    pub fn var(name: &str) -> Self {
        Self::new(AstOp::Var(name.to_string()), vec![], DType::Any)
    }

    /// Associates a data type with the node.
    pub fn with_type(self, dtype: DType) -> Self {
        Self::new(self.op, self.src, dtype)
    }

    /// Creates a `Cast` node to convert the data type of this node.
    pub fn cast(self, dtype: DType) -> Self {
        if let AstOp::Const(c) = self.op {
            let new_const = c.cast(dtype.clone());
            return AstNode::from(new_const);
        }
        Self::new(AstOp::Cast(dtype.clone()), vec![self], dtype)
    }

    /// Returns a new `AstNode` with the same `id`, `op`, and `dtype`, but with new `src` nodes.
    pub fn with_src(self, src: Vec<AstNode>) -> Self {
        Self {
            id: self.id,
            op: self.op,
            src,
            dtype: self.dtype,
        }
    }

    pub fn replace_node(&self, target: &AstNode, replacement: &AstNode) -> AstNode {
        if self == target {
            return replacement.clone();
        }

        let new_src: Vec<AstNode> = self
            .src
            .iter()
            .map(|child| child.replace_node(target, replacement))
            .collect();

        // If no children were changed, we can return the original node to avoid unnecessary clones.
        if new_src.iter().zip(self.src.iter()).all(|(a, b)| a == b) {
            return self.clone();
        }

        AstNode::new(self.op.clone(), new_src, self.dtype.clone())
    }
}
