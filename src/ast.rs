#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    Const(Const),

    // unary ops
    Neg(Box<Self>),
    Recip(Box<Self>),
    Sin(Box<Self>),
    Sqrt(Box<Self>),

    // binary ops
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),

    // Statements
    Loop {}, // for loop
    If {},
    IfElse {},
    Declare {}, // declare variable
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Ptr(Box<Self>),
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Const {
    F32(f32),
    F64(f64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

macro_rules! impl_dtype {
    ($variant: ident, $num_type: ident) => {
        impl From<$num_type> for Const {
            fn from(v: $num_type) -> Self {
                Const::$variant(v)
            }
        }

        impl From<$num_type> for Ast {
            fn from(v: $num_type) -> Self {
                Ast::Const(Const::from(v))
            }
        }
    };
}

impl_dtype!(F32, f32);
impl_dtype!(F64, f64);
impl_dtype!(I8, i8);
impl_dtype!(I16, i16);
impl_dtype!(I32, i32);
impl_dtype!(I64, i64);
impl_dtype!(U8, u8);
impl_dtype!(U16, u16);
impl_dtype!(U32, u32);
impl_dtype!(U64, u64);

impl Const {
    pub fn dtype(&self) -> DType {
        match *self {
            Const::F32(_) => DType::F32,
            Const::F64(_) => DType::F64,
            Const::I8(_) => DType::I8,
            Const::I16(_) => DType::I16,
            Const::I32(_) => DType::I32,
            Const::I64(_) => DType::I64,
            Const::U8(_) => DType::U8,
            Const::U16(_) => DType::U16,
            Const::U32(_) => DType::U32,
            Const::U64(_) => DType::U64,
        }
    }
}
