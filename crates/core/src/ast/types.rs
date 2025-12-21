//! データ型とリテラルの定義

/// ASTノードの型を表す列挙型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,            // boolean (internally represented as u8 or i8: 0 = false, non-zero = true)
    Int,             // integer (isize, for array indexing and loop counters)
    I32,             // 32-bit signed integer (for data arrays)
    F32,             // float
    Ptr(Box<DType>), // pointer for memory buffer, 値を渡す時は参照を渡す。
    Vec(Box<DType>, usize), // fixed size vector for SIMD, 値は渡す時にコピーされる
    Tuple(Vec<DType>),
    Unknown,
    // TODO: 将来的にf16とか対応させたい
}

/// リテラル値
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Bool(bool),
    Int(isize),
    I32(i32),
    F32(f32),
}

// Conversion from numeric types to Literal
impl From<bool> for Literal {
    fn from(value: bool) -> Self {
        Literal::Bool(value)
    }
}

impl From<f32> for Literal {
    fn from(value: f32) -> Self {
        Literal::F32(value)
    }
}

impl From<isize> for Literal {
    fn from(value: isize) -> Self {
        Literal::Int(value)
    }
}

impl From<i32> for Literal {
    fn from(value: i32) -> Self {
        Literal::I32(value)
    }
}

impl From<usize> for Literal {
    fn from(value: usize) -> Self {
        Literal::Int(value as isize)
    }
}

impl Literal {
    /// Get the DType of this literal
    pub fn dtype(&self) -> DType {
        match self {
            Literal::Bool(_) => DType::Bool,
            Literal::F32(_) => DType::F32,
            Literal::Int(_) => DType::Int,
            Literal::I32(_) => DType::I32,
        }
    }

    /// ブールリテラルを bool として取得
    ///
    /// Bool を bool に変換して返します。
    /// Bool でない場合は None を返します。
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Literal::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// 整数リテラルを isize として取得
    ///
    /// Int または I32 を isize に変換して返します。
    /// Bool, F32 の場合は None を返します。
    pub fn as_isize(&self) -> Option<isize> {
        match self {
            Literal::Int(v) => Some(*v),
            Literal::I32(v) => Some(*v as isize),
            Literal::Bool(_) | Literal::F32(_) => None,
        }
    }

    /// 整数リテラルを usize として取得
    ///
    /// Int または I32 を usize に変換して返します。
    /// Bool, F32 または負の値の場合は None を返します。
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Literal::Int(v) => (*v).try_into().ok(),
            Literal::I32(v) => (*v).try_into().ok(),
            Literal::Bool(_) | Literal::F32(_) => None,
        }
    }
}

impl DType {
    /// バイト単位でのサイズを取得
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Bool => 1, // u8として表現
            DType::Int => std::mem::size_of::<isize>(),
            DType::I32 => std::mem::size_of::<i32>(), // 4 bytes
            DType::F32 => std::mem::size_of::<f32>(),
            DType::Ptr(_) => std::mem::size_of::<usize>(), // ポインタはusizeと同じサイズ
            DType::Vec(elem_type, size) => elem_type.size_in_bytes() * size,
            DType::Tuple(types) => types.iter().map(|t| t.size_in_bytes()).sum(),
            DType::Unknown => 0,
        }
    }

    /// Convert this type to a vector type with the given size
    pub fn to_vec(&self, size: usize) -> DType {
        DType::Vec(Box::new(self.clone()), size)
    }

    /// Convert this type to a pointer type
    pub fn to_ptr(&self) -> DType {
        DType::Ptr(Box::new(self.clone()))
    }

    /// If this is a Vec type, return the element type and size
    /// Returns None if this is not a Vec type
    pub fn from_vec(&self) -> Option<(&DType, usize)> {
        match self {
            DType::Vec(elem_type, size) => Some((elem_type.as_ref(), *size)),
            _ => None,
        }
    }

    /// If this is a Ptr type, return the pointee type
    /// Returns None if this is not a Ptr type
    pub fn from_ptr(&self) -> Option<&DType> {
        match self {
            DType::Ptr(pointee) => Some(pointee.as_ref()),
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
            DType::Ptr(pointee) => pointee.as_ref(),
            _ => self,
        }
    }

    /// Check if this is a Vec type
    pub fn is_vec(&self) -> bool {
        matches!(self, DType::Vec(_, _))
    }

    /// Check if this is a Ptr type
    pub fn is_ptr(&self) -> bool {
        matches!(self, DType::Ptr(_))
    }

    /// Check if this is a Bool type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }
}

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
    fn test_bool_literal_from_conversion() {
        let lit: Literal = true.into();
        assert_eq!(lit, Literal::Bool(true));

        let lit: Literal = false.into();
        assert_eq!(lit, Literal::Bool(false));
    }

    #[test]
    fn test_bool_dtype_size() {
        assert_eq!(DType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn test_bool_dtype_is_bool() {
        assert!(DType::Bool.is_bool());
        assert!(!DType::F32.is_bool());
        assert!(!DType::Int.is_bool());
    }

    #[test]
    fn test_bool_ptr_and_vec() {
        let bool_ptr = DType::Bool.to_ptr();
        assert_eq!(bool_ptr, DType::Ptr(Box::new(DType::Bool)));
        assert_eq!(bool_ptr.deref_type(), &DType::Bool);

        let bool_vec = DType::Bool.to_vec(4);
        assert_eq!(bool_vec, DType::Vec(Box::new(DType::Bool), 4));
        assert_eq!(bool_vec.element_type(), &DType::Bool);
    }

    #[test]
    fn test_literal_as_isize_with_bool() {
        let bool_lit = Literal::Bool(true);
        assert_eq!(bool_lit.as_isize(), None);
        assert_eq!(bool_lit.as_usize(), None);
    }
}
