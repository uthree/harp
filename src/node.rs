use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

// --- DType system (no changes) ---
pub trait DType: Debug + DynClone {}
impl DType for f32 {}
impl DType for f64 {}
impl DType for i32 {}
impl DType for i64 {}
dyn_clone::clone_trait_object!(DType);

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
macro_rules! def_operator {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name;
        impl Operator for $name {
            fn as_any(&self) -> &dyn Any { self }
            fn name(&self) -> &'static str { stringify!($name) }
        }
    };
}
def_operator!(Add);
def_operator!(Sub);
def_operator!(Mul);
def_operator!(Div);
def_operator!(Load);
def_operator!(Store);

#[derive(Debug, Clone)]
pub struct Const(pub Box<dyn DType>);
impl Operator for Const {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &'static str { "Const" }
}

// --- Marker Traits for Operators ---
pub trait UnaryOp: Operator {}
pub trait BinaryOp: Operator {}
pub trait CommutativeOp: BinaryOp {}

// --- Implement Marker Traits ---
impl BinaryOp for Add {}
impl CommutativeOp for Add {}
impl BinaryOp for Mul {}
impl CommutativeOp for Mul {}

impl BinaryOp for Sub {}
impl BinaryOp for Div {}

// --- Node Struct ---
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Box<dyn Operator>,
    pub src: Vec<Arc<Self>>,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        // Compare operators by their type and sources.
        // Dereference the boxes to compare the `dyn Operator` trait objects.
        &*self.op == &*other.op && self.src == other.src
    }
}
impl Eq for Node {}