use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;

/// A trait for data types that can be used in graph nodes.
///
/// This trait provides the basic building blocks for type safety within the graph.
pub trait DType: Debug + DynClone + Any {
    /// Returns the value as `&dyn Any`, allowing for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns the name of the concrete type.
    fn type_name(&self) -> &'static str;
}
dyn_clone::clone_trait_object!(DType);

// --- Marker Traits for DTypes ---

/// A marker trait for floating-point types.
pub trait Float: DType {}
/// A marker trait for integer types.
pub trait Integer: DType {}
/// A marker trait for unsigned integer types.
pub trait UnsignedInteger: Integer {}
/// A marker trait for signed integer types.
pub trait SignedInteger: Integer {}

// --- DType Implementations ---

/// A macro to implement the `DType` trait for given types.
macro_rules! impl_dtype {
    ($($t:ty),*) => {
        $(
            impl DType for $t {
                fn as_any(&self) -> &dyn Any { self }
                fn type_name(&self) -> &'static str {
                    std::any::type_name::<Self>()
                }
            }
        )*
    };
}

/// A macro to implement the `Float` marker trait.
macro_rules! impl_float {
    ($($t:ty),*) => { $( impl Float for $t {} )* };
}

/// A macro to implement the `Integer` marker trait.
macro_rules! impl_integer {
    ($($t:ty),*) => { $( impl Integer for $t {} )* };
}

/// A macro to implement the `UnsignedInteger` marker trait.
macro_rules! impl_unsigned {
    ($($t:ty),*) => { $( impl UnsignedInteger for $t {} )* };
}

/// A macro to implement the `SignedInteger` marker trait.
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
