//! # Operator Module
//!
//! Defines the `Operator` trait and the various operator types used in the graph.

use crate::dtype::DType;
use crate::node::Node;
use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;

// --- Main Operator Trait ---
pub trait Operator: Debug + DynClone + Any {
    fn as_any(&self) -> &dyn Any;
    fn name(&self) -> &'static str;
}
dyn_clone::clone_trait_object!(Operator);

impl PartialEq for dyn Operator {
    fn eq(&self, other: &Self) -> bool {
        self.as_any().type_id() == other.as_any().type_id()
    }
}
impl Eq for dyn Operator {}

// --- Operator Traits ---

/// Marker trait for operators that can be used to build a Tensor graph.
pub trait TensorOperator: Operator {}

/// Marker trait for operators that are applied element-wise.
pub trait Elementwise: Operator {}

/// Trait for binary operators that have an identity element.
pub trait HasIdentityElement: Operator {
    fn identity_element() -> Node;
}

/// Marker trait for primitive operators that can be compiled.
pub trait PrimitiveOp: Operator {}

/// Trait for fused operators that can be decomposed into primitive operators.
/// This is mainly used to represent fused operators for acceleration,
/// allowing them to be broken down into a graph of primitive operations.
pub trait FusedOp: Operator {
    /// Decomposes the fused operator into a subgraph of primitive operators.
    ///
    /// # Arguments
    ///
    /// * `operands` - The operands of the fused operator.
    ///
    /// # Returns
    ///
    /// The root node of the subgraph representing the decomposition.
    fn fallback(&self, operands: &[Node]) -> Node;
}

pub trait UnaryOp: Operator {}
pub trait BinaryOp: Operator {}
pub trait CommutativeOp: BinaryOp {}

// --- Operator Structs ---

macro_rules! def_operators {
    ($($name:ident),*) => {
        $(
            #[derive(Debug, Clone, PartialEq, Eq)]
            pub struct $name;
        )*
    };
}

macro_rules! impl_operator {
    ($($name:ident),*) => {
        $(
            impl Operator for $name {
                fn as_any(&self) -> &dyn Any { self }
                fn name(&self) -> &'static str { stringify!($name) }
            }
        )*
    };
}

// Define all operator structs
def_operators!(
    OpAdd, OpSub, OpMul, OpDiv, OpRem, Load, Store, Recip, Wildcard, Sin, Exp2, Log2, Sqrt, Max,
    Sink, Loop, Reshape, OpUniform, OpRandn
);
impl_operator!(
    OpAdd, OpSub, OpMul, OpDiv, OpRem, Load, Store, Recip, Wildcard, Sin, Exp2, Log2, Sqrt, Max,
    Sink, Loop, Reshape, OpUniform, OpRandn
);

// --- Specialized Operator Structs ---

#[derive(Debug, Clone)]
pub struct Const(pub Box<dyn DType>);
impl Operator for Const {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Const"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capture(pub String);
impl Operator for Capture {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Capture"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable(pub String);
impl Operator for Variable {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Variable"
    }
}

#[derive(Debug, Clone)]
pub struct Cast(pub Box<dyn DType>);
impl Operator for Cast {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Cast"
    }
}

#[derive(Debug, Clone)]
pub struct Reduce {
    pub op: Box<dyn Operator>, // In a real scenario, this would have trait bounds
    pub axis: usize,
}
impl Operator for Reduce {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Reduce"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permute {
    pub order: Vec<usize>,
}
impl Operator for Permute {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Permute"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expand {
    pub shape: Vec<u64>,
}
impl Operator for Expand {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Expand"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice {
    pub args: Vec<(u64, u64)>,
}
impl Operator for Slice {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Slice"
    }
}

// --- Trait Implementations for Operators ---

// PrimitiveOps
impl PrimitiveOp for OpAdd {}
impl PrimitiveOp for OpMul {}
impl PrimitiveOp for OpRem {}
impl PrimitiveOp for Load {}
impl PrimitiveOp for Store {}
impl PrimitiveOp for Recip {}
impl PrimitiveOp for Sin {}
impl PrimitiveOp for Exp2 {}
impl PrimitiveOp for Log2 {}
impl PrimitiveOp for Sqrt {}
impl PrimitiveOp for Const {}
impl PrimitiveOp for Cast {}
impl PrimitiveOp for Max {}
impl PrimitiveOp for Variable {}
impl PrimitiveOp for OpUniform {}
impl PrimitiveOp for OpRandn {}

// FusedOps
impl FusedOp for OpSub {
    fn fallback(&self, operands: &[Node]) -> Node {
        assert_eq!(operands.len(), 2, "OpSub expects 2 operands");
        operands[0].clone() - operands[1].clone()
    }
}

impl FusedOp for OpDiv {
    fn fallback(&self, operands: &[Node]) -> Node {
        assert_eq!(operands.len(), 2, "OpDiv expects 2 operands");
        operands[0].clone() / operands[1].clone()
    }
}

// TensorOperators
impl TensorOperator for OpAdd {}
impl TensorOperator for OpSub {}
impl TensorOperator for OpMul {}
impl TensorOperator for OpDiv {}
impl TensorOperator for Reshape {}
impl TensorOperator for Permute {}
impl TensorOperator for Expand {}
impl TensorOperator for Slice {}
impl TensorOperator for OpUniform {}
impl TensorOperator for OpRandn {}
// Add other ops that can be used in tensor graphs...
impl TensorOperator for Sin {}
impl TensorOperator for Exp2 {}
impl TensorOperator for Log2 {}
impl TensorOperator for Sqrt {}
impl TensorOperator for Max {}
impl TensorOperator for Reduce {}

// Binary & Commutative
impl BinaryOp for OpAdd {}
impl CommutativeOp for OpAdd {}
impl BinaryOp for OpSub {}
impl BinaryOp for OpMul {}
impl CommutativeOp for OpMul {}
impl BinaryOp for OpDiv {}
impl BinaryOp for Max {}
impl CommutativeOp for Max {}

// Unary
impl UnaryOp for Recip {}
impl UnaryOp for Sin {}
impl UnaryOp for Exp2 {}
impl UnaryOp for Log2 {}
impl UnaryOp for Sqrt {}
impl UnaryOp for Cast {}
impl UnaryOp for Reshape {}
impl UnaryOp for Permute {}
impl UnaryOp for Expand {}
impl UnaryOp for Slice {}

// Elementwise
impl Elementwise for OpAdd {}
impl Elementwise for OpSub {}
impl Elementwise for OpMul {}
impl Elementwise for OpDiv {}
impl Elementwise for Sin {}
impl Elementwise for Exp2 {}
impl Elementwise for Log2 {}
impl Elementwise for Sqrt {}
impl Elementwise for Max {}

// HasIdentityElement
impl HasIdentityElement for OpAdd {
    fn identity_element() -> Node {
        crate::node::constant(0.0f64)
    }
}
impl HasIdentityElement for OpMul {
    fn identity_element() -> Node {
        crate::node::constant(1.0f64)
    }
}
impl HasIdentityElement for Max {
    fn identity_element() -> Node {
        crate::node::constant(f64::NEG_INFINITY)
    }
}
