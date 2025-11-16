//! データ型とリテラルの定義

/// ASTノードの型を表す列挙型
#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Int,                    // integer (for array indexing and general computation)
    F32,                    // float
    Ptr(Box<DType>),        // pointer for memory buffer, 値を渡す時は参照を渡す。
    Vec(Box<DType>, usize), // fixed size vector for SIMD, 値は渡す時にコピーされる
    Tuple(Vec<DType>),
    Unknown,
    // TODO: boolなどの追加
    // TODO: 将来的にf16とか対応させたい
}

/// リテラル値
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(isize),
    F32(f32),
}

// Conversion from numeric types to Literal
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

impl From<usize> for Literal {
    fn from(value: usize) -> Self {
        Literal::Int(value as isize)
    }
}

impl Literal {
    /// Get the DType of this literal
    pub fn dtype(&self) -> DType {
        match self {
            Literal::F32(_) => DType::F32,
            Literal::Int(_) => DType::Int,
        }
    }

    /// 整数リテラルを isize として取得
    ///
    /// Int を isize に変換して返します。
    /// F32 の場合は None を返します。
    pub fn as_isize(&self) -> Option<isize> {
        match self {
            Literal::Int(v) => Some(*v),
            Literal::F32(_) => None,
        }
    }

    /// 整数リテラルを usize として取得
    ///
    /// Int を usize に変換して返します。
    /// F32 または負の Int の場合は None を返します。
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Literal::Int(v) => (*v).try_into().ok(),
            Literal::F32(_) => None,
        }
    }
}

impl DType {
    /// バイト単位でのサイズを取得
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Int => std::mem::size_of::<isize>(),
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
}
