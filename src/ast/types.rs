use std::fmt;

#[derive(Debug, Clone, PartialEq, Default, Eq, Hash)]
pub enum DType {
    #[default]
    F32, // float
    Usize, // size_t
    Isize, // ssize_t
    Bool,  // boolean
    Void,

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array (for SIMD vectorization)
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::Usize => write!(f, "Usize"),
            DType::Isize => write!(f, "Isize"),
            DType::Bool => write!(f, "Bool"),
            DType::Void => write!(f, "Void"),
            DType::Ptr(inner) => write!(f, "Ptr<{}>", inner),
            DType::Vec(inner, size) => write!(f, "Vec<{}, {}>", inner, size),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstLiteral {
    F32(f32),
    Usize(usize),
    Isize(isize),
    Bool(bool),
}

// f32はEqを実装していないので手動でEqとHashを実装
impl Eq for ConstLiteral {}

impl std::hash::Hash for ConstLiteral {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ConstLiteral::F32(f) => {
                0u8.hash(state);
                f.to_bits().hash(state);
            }
            ConstLiteral::Usize(u) => {
                1u8.hash(state);
                u.hash(state);
            }
            ConstLiteral::Isize(i) => {
                2u8.hash(state);
                i.hash(state);
            }
            ConstLiteral::Bool(b) => {
                3u8.hash(state);
                b.hash(state);
            }
        }
    }
}
