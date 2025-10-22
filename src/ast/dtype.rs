#[derive(Debug, Clone)]
pub enum DType {
    Isize,                  // signed integer
    Usize,                  // unsigned integer (for array indexing)
    F32,                    // float
    Ptr(Box<DType>),        // pointer for memory buffer
    Vec(Box<DType>, usize), // fixed size vector for SIMD
    Tuple(Vec<DType>),
    Unknown,
}
