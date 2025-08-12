use std::any::TypeId;
use std::hash::{Hash, Hasher};

use crate::ast::node::AstNode;

/// Represents the data type of a value in the AST.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    USize,
    /// Represents a void or empty type.
    Void,
    /// A pointer to another type.
    Ptr(Box<Self>),
    /// An array of a type.
    /// In C programming language, pointer of first element.
    Array(Box<Self>, Box<AstNode>),
    /// A tuple of types.
    Tuple(Vec<Self>),
    // --- Types for pattern matching ---
    /// Matches any type.
    Any,
    /// Matches any natural number type (unsigned integers).
    Natural,
    /// Matches any integer type (signed or unsigned).
    Integer,
    /// Matches any real number type (floats).
    Real,
}

impl DType {
    pub fn from_type<T: 'static>() -> Self {
        let type_id = TypeId::of::<T>();
        if type_id == TypeId::of::<f32>() {
            DType::F32
        } else if type_id == TypeId::of::<f64>() {
            DType::F64
        } else if type_id == TypeId::of::<i8>() {
            DType::I8
        } else if type_id == TypeId::of::<i16>() {
            DType::I16
        } else if type_id == TypeId::of::<i32>() {
            DType::I32
        } else if type_id == TypeId::of::<i64>() {
            DType::I64
        } else if type_id == TypeId::of::<u8>() {
            DType::U8
        } else if type_id == TypeId::of::<u16>() {
            DType::U16
        } else if type_id == TypeId::of::<u32>() {
            DType::U32
        } else if type_id == TypeId::of::<u64>() {
            DType::U64
        } else if type_id == TypeId::of::<usize>() {
            DType::USize
        } else {
            panic!("Unsupported type");
        }
    }

    /// Converts the `DType` to a `TypeId`.
    ///
    /// # Panics
    ///
    /// Panics if the `DType` cannot be represented as a concrete Rust type,
    /// such as `Any`, `Void`, or `Ptr`.
    pub fn to_type_id(&self) -> TypeId {
        match self {
            DType::F32 => TypeId::of::<f32>(),
            DType::F64 => TypeId::of::<f64>(),
            DType::I8 => TypeId::of::<i8>(),
            DType::I16 => TypeId::of::<i16>(),
            DType::I32 => TypeId::of::<i32>(),
            DType::I64 => TypeId::of::<i64>(),
            DType::U8 => TypeId::of::<u8>(),
            DType::U16 => TypeId::of::<u16>(),
            DType::U32 => TypeId::of::<u32>(),
            DType::U64 => TypeId::of::<u64>(),
            DType::USize => TypeId::of::<usize>(),
            _ => panic!("Cannot convert {self:?} to TypeId"),
        }
    }

    /// Returns `true` if the type is a real number (float).
    pub fn is_real(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::Real)
    }

    /// Returns `true` if the type is a natural number (unsigned integer).
    pub fn is_natural(&self) -> bool {
        matches!(
            self,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::USize | DType::Natural
        )
    }

    /// Returns `true` if the type is an integer (signed or unsigned).
    pub fn is_integer(&self) -> bool {
        self.is_natural()
            || matches!(
                self,
                DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::Integer
            )
    }

    /// Returns the size of the data type in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::USize => std::mem::size_of::<usize>(),
            DType::Ptr(_) => std::mem::size_of::<*const ()>(),
            DType::Array(..) => std::mem::size_of::<*const ()>(),
            _ => panic!("Cannot get size of {self:?}"),
        }
    }

    /// Checks if this type matches another type, considering pattern matching types.
    pub fn matches(&self, other: &DType) -> bool {
        if self == other {
            return true;
        }
        match self {
            DType::Any => true,
            DType::Real => other.is_real(),
            DType::Natural => other.is_natural(),
            DType::Integer => other.is_integer(),
            DType::Ptr(a) => {
                if let DType::Ptr(b) = other {
                    a.matches(b)
                } else {
                    false
                }
            }
            DType::Array(a, ..) => {
                if let DType::Array(b, ..) = other {
                    a.matches(b)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// Represents a constant literal value.
#[derive(Debug, Clone, Copy)]
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
    USize(usize),
}

impl PartialEq for Const {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(l), Self::F32(r)) => l.to_bits() == r.to_bits(),
            (Self::F64(l), Self::F64(r)) => l.to_bits() == r.to_bits(),
            (Self::I8(l), Self::I8(r)) => l == r,
            (Self::I16(l), Self::I16(r)) => l == r,
            (Self::I32(l), Self::I32(r)) => l == r,
            (Self::I64(l), Self::I64(r)) => l == r,
            (Self::U8(l), Self::U8(r)) => l == r,
            (Self::U16(l), Self::U16(r)) => l == r,
            (Self::U32(l), Self::U32(r)) => l == r,
            (Self::U64(l), Self::U64(r)) => l == r,
            (Self::USize(l), Self::USize(r)) => l == r,
            _ => false,
        }
    }
}

impl Eq for Const {}

impl Hash for Const {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Const::F32(v) => v.to_bits().hash(state),
            Const::F64(v) => v.to_bits().hash(state),
            Const::I8(v) => v.hash(state),
            Const::I16(v) => v.hash(state),
            Const::I32(v) => v.hash(state),
            Const::I64(v) => v.hash(state),
            Const::U8(v) => v.hash(state),
            Const::U16(v) => v.hash(state),
            Const::U32(v) => v.hash(state),
            Const::U64(v) => v.hash(state),
            Const::USize(v) => v.hash(state),
        }
    }
}

impl Const {
    pub fn to_usize(&self) -> Option<usize> {
        match *self {
            Const::I8(v) => Some(v as usize),
            Const::I16(v) => Some(v as usize),
            Const::I32(v) => Some(v as usize),
            Const::I64(v) => Some(v as usize),
            Const::U8(v) => Some(v as usize),
            Const::U16(v) => Some(v as usize),
            Const::U32(v) => Some(v as usize),
            Const::U64(v) => Some(v as usize),
            Const::USize(v) => Some(v),
            _ => None, // Floats cannot be safely converted to usize
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Const::F32(v) => *v == 0.0,
            Const::F64(v) => *v == 0.0,
            Const::I8(v) => *v == 0,
            Const::I16(v) => *v == 0,
            Const::I32(v) => *v == 0,
            Const::I64(v) => *v == 0,
            Const::U8(v) => *v == 0,
            Const::U16(v) => *v == 0,
            Const::U32(v) => *v == 0,
            Const::U64(v) => *v == 0,
            Const::USize(v) => *v == 0,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            Const::F32(v) => *v == 1.0,
            Const::F64(v) => *v == 1.0,
            Const::I8(v) => *v == 1,
            Const::I16(v) => *v == 1,
            Const::I32(v) => *v == 1,
            Const::I64(v) => *v == 1,
            Const::U8(v) => *v == 1,
            Const::U16(v) => *v == 1,
            Const::U32(v) => *v == 1,
            Const::U64(v) => *v == 1,
            Const::USize(v) => *v == 1,
        }
    }
}

macro_rules! impl_dtype {
    ($variant: ident, $num_type: ident) => {
        impl From<$num_type> for Const {
            fn from(v: $num_type) -> Self {
                Const::$variant(v)
            }
        }

        impl From<$num_type> for AstNode {
            fn from(v: $num_type) -> Self {
                let c = Const::$variant(v);
                AstNode::new(super::op::AstOp::Const(c), vec![], c.dtype())
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
impl_dtype!(USize, usize);

impl Const {
    pub fn cast(&self, target_dtype: DType) -> Self {
        match self {
            Const::F32(v) => match target_dtype {
                DType::F32 => Const::F32(*v),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from F32 to {:?}", target_dtype),
            },
            Const::F64(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from F64 to {:?}", target_dtype),
            },
            Const::I8(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from I8 to {:?}", target_dtype),
            },
            Const::I16(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from I16 to {:?}", target_dtype),
            },
            Const::I32(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from I32 to {:?}", target_dtype),
            },
            Const::I64(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from I64 to {:?}", target_dtype),
            },
            Const::U8(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from U8 to {:?}", target_dtype),
            },
            Const::U16(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from U16 to {:?}", target_dtype),
            },
            Const::U32(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from U32 to {:?}", target_dtype),
            },
            Const::U64(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v),
                DType::USize => Const::USize(*v as usize),
                _ => panic!("Unsupported cast from U64 to {:?}", target_dtype),
            },
            Const::USize(v) => match target_dtype {
                DType::F32 => Const::F32(*v as f32),
                DType::F64 => Const::F64(*v as f64),
                DType::I8 => Const::I8(*v as i8),
                DType::I16 => Const::I16(*v as i16),
                DType::I32 => Const::I32(*v as i32),
                DType::I64 => Const::I64(*v as i64),
                DType::U8 => Const::U8(*v as u8),
                DType::U16 => Const::U16(*v as u16),
                DType::U32 => Const::U32(*v as u32),
                DType::U64 => Const::U64(*v as u64),
                DType::USize => Const::USize(*v),
                _ => panic!("Unsupported cast from USize to {:?}", target_dtype),
            },
        }
    }
}

impl From<Const> for AstNode {
    fn from(c: Const) -> Self {
        AstNode::new(super::op::AstOp::Const(c), vec![], c.dtype())
    }
}

impl Const {
    /// Returns the `DType` corresponding to the constant value.
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
            Const::USize(_) => DType::USize,
        }
    }
}
