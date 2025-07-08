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

/// Represents a scalar value of any supported data type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Bool(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    // F16(f16),
    // BF16(bf16),
    F32(f32),
    F64(f64),
}

impl Scalar {
    /// Returns the `DType` of the scalar value.
    pub fn dtype(&self) -> &'static dyn DType {
        match self {
            Scalar::Bool(_) => BOOL_DTYPE,
            Scalar::I8(_) => I8_DTYPE,
            Scalar::U8(_) => U8_DTYPE,
            Scalar::I16(_) => I16_DTYPE,
            Scalar::U16(_) => U16_DTYPE,
            Scalar::I32(_) => I32_DTYPE,
            Scalar::U32(_) => U32_DTYPE,
            Scalar::I64(_) => I64_DTYPE,
            Scalar::U64(_) => U64_DTYPE,
            // Scalar::F16(_) => F16_DTYPE,
            // Scalar::BF16(_) => BF16_DTYPE,
            Scalar::F32(_) => F32_DTYPE,
            Scalar::F64(_) => F64_DTYPE,
        }
    }
}