//! # Operator Module
//!
//! Defines the `Operator` trait and the various operator types used in the graph.

use crate::dtype::DType;
use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;

// --- Operator Trait ---
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

def_operators!(
    OpAdd, OpMul, OpRem, Load, Store, Recip, Wildcard, Sin, Exp2, Log2, Sqrt
);
impl_operator!(
    OpAdd, OpMul, OpRem, Load, Store, Recip, Wildcard, Sin, Exp2, Log2, Sqrt
);

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

// --- Marker Traits for Operators ---
pub trait UnaryOp: Operator {}
pub trait BinaryOp: Operator {}
pub trait CommutativeOp: BinaryOp {}

// --- Implement Marker Traits ---
impl BinaryOp for OpAdd {}
impl CommutativeOp for OpAdd {}
impl BinaryOp for OpMul {}
impl CommutativeOp for OpMul {}

impl UnaryOp for Recip {}
impl UnaryOp for Sin {}
impl UnaryOp for Exp2 {}
impl UnaryOp for Log2 {}
impl UnaryOp for Sqrt {}
impl UnaryOp for Cast {}
