use std::any::TypeId;
use std::fmt;

/// A trait representing a data type.
pub trait DType: 'static + Send + Sync + fmt::Debug + fmt::Display {
    /// Returns the `TypeId` of the implementing type.
    fn id(&self) -> TypeId;
    /// Returns the name of the data type.
    fn name(&self) -> &'static str;
}

// Using a macro to reduce boilerplate for defining DType structs and implementing the trait.
macro_rules! define_dtype {
    ($($struct_name:ident),*) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            pub struct $struct_name;

            impl DType for $struct_name {
                fn id(&self) -> TypeId {
                    TypeId::of::<$struct_name>()
                }

                fn name(&self) -> &'static str {
                    stringify!($struct_name)
                }
            }

            impl fmt::Display for $struct_name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", self.name())
                }
            }
        )*
    };
}

// Define all the data type structs.
define_dtype!(
    Bool, I8, U8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64
);

// Static instances of each dtype.
pub const BOOL_DTYPE: &dyn DType = &Bool;
pub const I8_DTYPE: &dyn DType = &I8;
pub const U8_DTYPE: &dyn DType = &U8;
pub const I16_DTYPE: &dyn DType = &I16;
pub const U16_DTYPE: &dyn DType = &U16;
pub const I32_DTYPE: &dyn DType = &I32;
pub const U32_DTYPE: &dyn DType = &U32;
pub const I64_DTYPE: &dyn DType = &I64;
pub const U64_DTYPE: &dyn DType = &U64;
pub const F16_DTYPE: &dyn DType = &F16;
pub const BF16_DTYPE: &dyn DType = &BF16;
pub const F32_DTYPE: &dyn DType = &F32;
pub const F64_DTYPE: &dyn DType = &F64;
