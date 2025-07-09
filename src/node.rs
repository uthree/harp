use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

// --- DType System ---
pub trait DType: Debug + DynClone {}
dyn_clone::clone_trait_object!(DType);

// --- Marker Traits for DTypes ---
pub trait Float: DType {}
pub trait Integer: DType {}
pub trait UnsignedInteger: Integer {}
pub trait SignedInteger: Integer {}

// --- DType Implementations ---
macro_rules! impl_dtype {
    ($($t:ty),*) => { $( impl DType for $t {} )* };
}
macro_rules! impl_float {
    ($($t:ty),*) => { $( impl Float for $t {} )* };
}
macro_rules! impl_integer {
    ($($t:ty),*) => { $( impl Integer for $t {} )* };
}
macro_rules! impl_unsigned {
    ($($t:ty),*) => { $( impl UnsignedInteger for $t {} )* };
}
macro_rules! impl_signed {
    ($($t:ty),*) => { $( impl SignedInteger for $t {} )* };
}

// Implement for Floats
impl_dtype!(f32, f64);
impl_float!(f32, f64);

// Implement for Unsigned Integers
impl_dtype!(u8, u16, u32, u64, usize);
impl_integer!(u8, u16, u32, u64, usize);
impl_unsigned!(u8, u16, u32, u64, usize);

// Implement for Signed Integers
impl_dtype!(i8, i16, i32, i64, isize);
impl_integer!(i8, i16, i32, i64, isize);
impl_signed!(i8, i16, i32, i64, isize);

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

def_operators!(Add, Mul, Load, Store, Recip);
impl_operator!(Add, Mul, Load, Store, Recip);

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

// --- Marker Traits for Operators ---
pub trait UnaryOp: Operator {}
pub trait BinaryOp: Operator {}
pub trait CommutativeOp: BinaryOp {}

// --- Implement Marker Traits ---
impl BinaryOp for Add {}
impl CommutativeOp for Add {}
impl BinaryOp for Mul {}
impl CommutativeOp for Mul {}

impl UnaryOp for Recip {}

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
