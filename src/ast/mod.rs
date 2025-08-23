#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    F32,   // float
    Usize, // size_t
    Isize, // ssize_t
    Void,

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array (for SIMD vectorization)

    Any, // for pattern matching
}
