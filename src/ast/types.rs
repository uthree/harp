//! データ型とリテラルの定義

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ============================================================================
// TensorDType trait - Maps Rust types to DType
// ============================================================================

/// Trait for types that can be used as tensor element types.
///
/// This trait provides a compile-time mapping from Rust types to DType values,
/// enabling type-safe tensor operations.
///
/// # Example
///
/// ```ignore
/// use eclat::TensorDType;
///
/// assert_eq!(f32::DTYPE, DType::F32);
/// assert_eq!(i32::DTYPE, DType::I32);
/// ```
pub trait TensorDType: Clone + Debug + Send + Sync + 'static {
    /// The corresponding DType for this Rust type
    const DTYPE: DType;

    /// Convert a value of this type to a Literal
    fn to_literal(value: Self) -> Literal;

    /// Get the zero value for this type
    fn zero() -> Self;

    /// Get the one value for this type
    fn one() -> Self;
}

// Macro for implementing TensorDType for numeric types
macro_rules! impl_tensor_dtype {
    // For types with simple zero/one literals (integers)
    ($ty:ty, $dtype:ident, $lit:ident, $zero:expr, $one:expr) => {
        impl TensorDType for $ty {
            const DTYPE: DType = DType::$dtype;
            fn to_literal(value: Self) -> Literal {
                Literal::$lit(value)
            }
            fn zero() -> Self {
                $zero
            }
            fn one() -> Self {
                $one
            }
        }
    };
}

impl_tensor_dtype!(bool, Bool, Bool, false, true);
impl_tensor_dtype!(i8, I8, I8, 0, 1);
impl_tensor_dtype!(i16, I16, I16, 0, 1);
impl_tensor_dtype!(i32, I32, I32, 0, 1);
impl_tensor_dtype!(i64, I64, I64, 0, 1);
impl_tensor_dtype!(u8, U8, U8, 0, 1);
impl_tensor_dtype!(u16, U16, U16, 0, 1);
impl_tensor_dtype!(u32, U32, U32, 0, 1);
impl_tensor_dtype!(u64, U64, U64, 0, 1);
impl_tensor_dtype!(f16, F16, F16, f16::ZERO, f16::ONE);
impl_tensor_dtype!(bf16, BF16, BF16, bf16::ZERO, bf16::ONE);
impl_tensor_dtype!(f32, F32, F32, 0.0, 1.0);
impl_tensor_dtype!(f64, F64, F64, 0.0, 1.0);

// ============================================================================
// AddressSpace enum
// ============================================================================

/// メモリアドレス空間を表す列挙型
/// GPU カーネルにおけるメモリの種類を区別する
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum AddressSpace {
    /// グローバルメモリ（デフォルト）
    /// GPU の VRAM 上のメモリで、全スレッドからアクセス可能
    #[default]
    Global,
    /// 共有メモリ（シェアードメモリ）
    /// スレッドグループ内で共有される高速メモリ
    Shared,
    /// ローカルメモリ（プライベートメモリ）
    /// 各スレッドに固有のメモリ
    Local,
    /// 定数メモリ
    /// 読み取り専用でキャッシュされるメモリ
    Constant,
}

// ============================================================================
// DType enum
// ============================================================================

/// ASTノードの型を表す列挙型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DType {
    // Void (no value, used for function return types)
    Void,

    // Boolean
    Bool,

    // Signed integers
    I8,
    I16,
    I32,
    I64,

    // Unsigned integers
    U8,
    U16,
    U32,
    U64,

    // Floating point
    F16,
    BF16,
    F32,
    F64,

    // Index type: バックエンドによって自動的に決定される整数型
    // GPU/CPUの最適なインデックス型に変換される（通常はi32またはi64）
    Int,

    // Composite types
    /// メモリバッファへのポインタ型
    /// 第1引数: ポイント先の型
    /// 第2引数: アドレス空間（Global, Shared, Local, Constant）
    Ptr(Box<DType>, AddressSpace),
    Vec(Box<DType>, usize), // fixed size vector for SIMD
    Tuple(Vec<DType>),

    #[default]
    Unknown,
}

/// リテラル値
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Bool(bool),

    // Signed integers
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),

    // Unsigned integers
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),

    // Floating point
    F16(f16),
    BF16(bf16),
    F32(f32),
    F64(f64),
}

// ============================================================================
// Conversion from numeric types to Literal
// ============================================================================

// Macro for implementing From<T> for Literal
macro_rules! impl_from_for_literal {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(
            impl From<$ty> for Literal {
                fn from(value: $ty) -> Self {
                    Literal::$variant(value)
                }
            }
        )*
    };
}

impl_from_for_literal!(
    bool => Bool,
    i8 => I8,
    i16 => I16,
    i32 => I32,
    i64 => I64,
    u8 => U8,
    u16 => U16,
    u32 => U32,
    u64 => U64,
    f16 => F16,
    bf16 => BF16,
    f32 => F32,
    f64 => F64,
);

impl From<usize> for Literal {
    fn from(value: usize) -> Self {
        // usizeはインデックス計算で符号付き演算が必要な場合が多いためI64に変換
        Literal::I64(value as i64)
    }
}

impl From<isize> for Literal {
    fn from(value: isize) -> Self {
        Literal::I64(value as i64)
    }
}

// ============================================================================
// Literal methods
// ============================================================================

impl Literal {
    /// Get the DType of this literal
    pub fn dtype(&self) -> DType {
        match self {
            Literal::Bool(_) => DType::Bool,
            Literal::I8(_) => DType::I8,
            Literal::I16(_) => DType::I16,
            Literal::I32(_) => DType::I32,
            Literal::I64(_) => DType::I64,
            Literal::U8(_) => DType::U8,
            Literal::U16(_) => DType::U16,
            Literal::U32(_) => DType::U32,
            Literal::U64(_) => DType::U64,
            Literal::F16(_) => DType::F16,
            Literal::BF16(_) => DType::BF16,
            Literal::F32(_) => DType::F32,
            Literal::F64(_) => DType::F64,
        }
    }

    /// ブールリテラルを bool として取得
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Literal::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// 整数リテラルを i64 として取得（符号付き整数のみ）
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Literal::I8(v) => Some(*v as i64),
            Literal::I16(v) => Some(*v as i64),
            Literal::I32(v) => Some(*v as i64),
            Literal::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// 整数リテラルを u64 として取得（符号なし整数のみ）
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Literal::U8(v) => Some(*v as u64),
            Literal::U16(v) => Some(*v as u64),
            Literal::U32(v) => Some(*v as u64),
            Literal::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// 任意の整数リテラルを i64 として取得
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Literal::I8(v) => Some(*v as i64),
            Literal::I16(v) => Some(*v as i64),
            Literal::I32(v) => Some(*v as i64),
            Literal::I64(v) => Some(*v),
            Literal::U8(v) => Some(*v as i64),
            Literal::U16(v) => Some(*v as i64),
            Literal::U32(v) => Some(*v as i64),
            Literal::U64(v) => (*v).try_into().ok(),
            _ => None,
        }
    }

    /// 整数リテラルを usize として取得
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Literal::I8(v) => (*v).try_into().ok(),
            Literal::I16(v) => (*v).try_into().ok(),
            Literal::I32(v) => (*v).try_into().ok(),
            Literal::I64(v) => (*v).try_into().ok(),
            Literal::U8(v) => Some(*v as usize),
            Literal::U16(v) => Some(*v as usize),
            Literal::U32(v) => Some(*v as usize),
            Literal::U64(v) => (*v).try_into().ok(),
            _ => None,
        }
    }

    /// 浮動小数点リテラルを f64 として取得
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Literal::F16(v) => Some(v.to_f64()),
            Literal::BF16(v) => Some(v.to_f64()),
            Literal::F32(v) => Some(*v as f64),
            Literal::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Check if the literal is zero
    pub fn is_zero(&self) -> bool {
        match self {
            Literal::Bool(v) => !*v,
            Literal::I8(v) => *v == 0,
            Literal::I16(v) => *v == 0,
            Literal::I32(v) => *v == 0,
            Literal::I64(v) => *v == 0,
            Literal::U8(v) => *v == 0,
            Literal::U16(v) => *v == 0,
            Literal::U32(v) => *v == 0,
            Literal::U64(v) => *v == 0,
            Literal::F16(v) => *v == f16::ZERO,
            Literal::BF16(v) => *v == bf16::ZERO,
            Literal::F32(v) => *v == 0.0,
            Literal::F64(v) => *v == 0.0,
        }
    }

    /// Check if the literal is one
    pub fn is_one(&self) -> bool {
        match self {
            Literal::Bool(v) => *v,
            Literal::I8(v) => *v == 1,
            Literal::I16(v) => *v == 1,
            Literal::I32(v) => *v == 1,
            Literal::I64(v) => *v == 1,
            Literal::U8(v) => *v == 1,
            Literal::U16(v) => *v == 1,
            Literal::U32(v) => *v == 1,
            Literal::U64(v) => *v == 1,
            Literal::F16(v) => *v == f16::ONE,
            Literal::BF16(v) => *v == bf16::ONE,
            Literal::F32(v) => *v == 1.0,
            Literal::F64(v) => *v == 1.0,
        }
    }
}

// ============================================================================
// DType methods
// ============================================================================

impl DType {
    /// バイト単位でのサイズを取得
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Void => 0,
            DType::Bool => 1,
            DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 => 8,
            DType::Int => std::mem::size_of::<isize>(), // Platform-dependent
            DType::Ptr(_, _) => std::mem::size_of::<usize>(),
            DType::Vec(elem_type, size) => elem_type.size_in_bytes() * size,
            DType::Tuple(types) => types.iter().map(|t| t.size_in_bytes()).sum(),
            DType::Unknown => 0,
        }
    }

    /// Convert this type to a vector type with the given size
    pub fn to_vec(&self, size: usize) -> DType {
        DType::Vec(Box::new(self.clone()), size)
    }

    /// Convert this type to a pointer type with Global address space
    pub fn to_ptr(&self) -> DType {
        DType::Ptr(Box::new(self.clone()), AddressSpace::Global)
    }

    /// Convert this type to a pointer type with specified address space
    pub fn to_ptr_with_space(&self, space: AddressSpace) -> DType {
        DType::Ptr(Box::new(self.clone()), space)
    }

    /// If this is a Vec type, return the element type and size
    pub fn from_vec(&self) -> Option<(&DType, usize)> {
        match self {
            DType::Vec(elem_type, size) => Some((elem_type.as_ref(), *size)),
            _ => None,
        }
    }

    /// If this is a Ptr type, return the pointee type
    pub fn from_ptr(&self) -> Option<&DType> {
        match self {
            DType::Ptr(pointee, _) => Some(pointee.as_ref()),
            _ => None,
        }
    }

    /// If this is a Ptr type, return the address space
    pub fn address_space(&self) -> Option<&AddressSpace> {
        match self {
            DType::Ptr(_, space) => Some(space),
            _ => None,
        }
    }

    /// Get the element type if this is a Vec, otherwise return self
    pub fn element_type(&self) -> &DType {
        match self {
            DType::Vec(elem_type, _) => elem_type.as_ref(),
            _ => self,
        }
    }

    /// Get the pointee type if this is a Ptr, otherwise return self
    pub fn deref_type(&self) -> &DType {
        match self {
            DType::Ptr(pointee, _) => pointee.as_ref(),
            _ => self,
        }
    }

    /// Check if this is a Vec type
    pub fn is_vec(&self) -> bool {
        matches!(self, DType::Vec(..))
    }

    /// Check if this is a Ptr type
    pub fn is_ptr(&self) -> bool {
        matches!(self, DType::Ptr(_, _))
    }

    /// Check if this is a shared memory pointer
    pub fn is_shared_ptr(&self) -> bool {
        matches!(self, DType::Ptr(_, AddressSpace::Shared))
    }

    /// Check if this is a global memory pointer
    pub fn is_global_ptr(&self) -> bool {
        matches!(self, DType::Ptr(_, AddressSpace::Global))
    }

    /// Check if this is a Bool type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Check if this is a signed integer type
    pub fn is_signed_integer(&self) -> bool {
        matches!(
            self,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::Int
        )
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }

    /// Check if this is any integer type (signed or unsigned)
    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    /// Check if this is a numeric type (integer or float)
    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Check if the type is known (not Unknown)
    pub fn is_known(&self) -> bool {
        !matches!(self, DType::Unknown)
    }

    /// Get bit width of the type (for numeric types)
    pub fn bit_width(&self) -> Option<usize> {
        match self {
            DType::Bool => Some(1),
            DType::I8 | DType::U8 => Some(8),
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => Some(16),
            DType::I32 | DType::U32 | DType::F32 => Some(32),
            DType::I64 | DType::U64 | DType::F64 => Some(64),
            DType::Int => Some(std::mem::size_of::<isize>() * 8),
            _ => None,
        }
    }

    /// Create a zero literal for this type
    pub fn zero(&self) -> Option<Literal> {
        match self {
            DType::Bool => Some(Literal::Bool(false)),
            DType::I8 => Some(Literal::I8(0)),
            DType::I16 => Some(Literal::I16(0)),
            DType::I32 => Some(Literal::I32(0)),
            DType::I64 | DType::Int => Some(Literal::I64(0)),
            DType::U8 => Some(Literal::U8(0)),
            DType::U16 => Some(Literal::U16(0)),
            DType::U32 => Some(Literal::U32(0)),
            DType::U64 => Some(Literal::U64(0)),
            DType::F16 => Some(Literal::F16(f16::ZERO)),
            DType::BF16 => Some(Literal::BF16(bf16::ZERO)),
            DType::F32 => Some(Literal::F32(0.0)),
            DType::F64 => Some(Literal::F64(0.0)),
            _ => None,
        }
    }

    /// Create a one literal for this type
    pub fn one(&self) -> Option<Literal> {
        match self {
            DType::Bool => Some(Literal::Bool(true)),
            DType::I8 => Some(Literal::I8(1)),
            DType::I16 => Some(Literal::I16(1)),
            DType::I32 => Some(Literal::I32(1)),
            DType::I64 | DType::Int => Some(Literal::I64(1)),
            DType::U8 => Some(Literal::U8(1)),
            DType::U16 => Some(Literal::U16(1)),
            DType::U32 => Some(Literal::U32(1)),
            DType::U64 => Some(Literal::U64(1)),
            DType::F16 => Some(Literal::F16(f16::ONE)),
            DType::BF16 => Some(Literal::BF16(bf16::ONE)),
            DType::F32 => Some(Literal::F32(1.0)),
            DType::F64 => Some(Literal::F64(1.0)),
            _ => None,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_literal() {
        let lit_true = Literal::Bool(true);
        let lit_false = Literal::Bool(false);

        assert_eq!(lit_true.dtype(), DType::Bool);
        assert_eq!(lit_false.dtype(), DType::Bool);
        assert_eq!(lit_true.as_bool(), Some(true));
        assert_eq!(lit_false.as_bool(), Some(false));
    }

    #[test]
    fn test_integer_literals() {
        assert_eq!(Literal::I8(42).dtype(), DType::I8);
        assert_eq!(Literal::I16(42).dtype(), DType::I16);
        assert_eq!(Literal::I32(42).dtype(), DType::I32);
        assert_eq!(Literal::I64(42).dtype(), DType::I64);
        assert_eq!(Literal::U8(42).dtype(), DType::U8);
        assert_eq!(Literal::U16(42).dtype(), DType::U16);
        assert_eq!(Literal::U32(42).dtype(), DType::U32);
        assert_eq!(Literal::U64(42).dtype(), DType::U64);
    }

    #[test]
    fn test_float_literals() {
        assert_eq!(Literal::F32(2.5).dtype(), DType::F32);
        assert_eq!(Literal::F64(2.5).dtype(), DType::F64);
    }

    #[test]
    fn test_literal_conversions() {
        let lit: Literal = 42i8.into();
        assert_eq!(lit, Literal::I8(42));

        let lit: Literal = 42i64.into();
        assert_eq!(lit, Literal::I64(42));

        let lit: Literal = 42u64.into();
        assert_eq!(lit, Literal::U64(42));

        let lit: Literal = 2.5f64.into();
        assert_eq!(lit, Literal::F64(2.5));
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Bool.size_in_bytes(), 1);
        assert_eq!(DType::I8.size_in_bytes(), 1);
        assert_eq!(DType::I16.size_in_bytes(), 2);
        assert_eq!(DType::I32.size_in_bytes(), 4);
        assert_eq!(DType::I64.size_in_bytes(), 8);
        assert_eq!(DType::U8.size_in_bytes(), 1);
        assert_eq!(DType::U16.size_in_bytes(), 2);
        assert_eq!(DType::U32.size_in_bytes(), 4);
        assert_eq!(DType::U64.size_in_bytes(), 8);
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F64.size_in_bytes(), 8);
    }

    #[test]
    fn test_dtype_is_methods() {
        assert!(DType::Bool.is_bool());
        assert!(DType::I32.is_signed_integer());
        assert!(DType::U32.is_unsigned_integer());
        assert!(DType::I32.is_integer());
        assert!(DType::U32.is_integer());
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(DType::Int.is_signed_integer());
    }

    #[test]
    fn test_dtype_bit_width() {
        assert_eq!(DType::I8.bit_width(), Some(8));
        assert_eq!(DType::I16.bit_width(), Some(16));
        assert_eq!(DType::I32.bit_width(), Some(32));
        assert_eq!(DType::I64.bit_width(), Some(64));
        assert_eq!(DType::F32.bit_width(), Some(32));
        assert_eq!(DType::F64.bit_width(), Some(64));
    }

    #[test]
    fn test_literal_is_zero_one() {
        assert!(Literal::I32(0).is_zero());
        assert!(!Literal::I32(1).is_zero());
        assert!(Literal::I32(1).is_one());
        assert!(!Literal::I32(0).is_one());
        assert!(Literal::F32(0.0).is_zero());
        assert!(Literal::F32(1.0).is_one());
    }

    #[test]
    fn test_dtype_zero_one() {
        assert_eq!(DType::I32.zero(), Some(Literal::I32(0)));
        assert_eq!(DType::I32.one(), Some(Literal::I32(1)));
        assert_eq!(DType::F64.zero(), Some(Literal::F64(0.0)));
        assert_eq!(DType::F64.one(), Some(Literal::F64(1.0)));
    }

    #[test]
    fn test_tensor_dtype_trait() {
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(i32::DTYPE, DType::I32);
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(i32::zero(), 0);
        assert_eq!(i32::one(), 1);
    }
}
