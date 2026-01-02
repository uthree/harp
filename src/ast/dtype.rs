#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum DType {
    f32,
    f64,
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    bool,
    Int,                    // integer for indexing
    Vec(Box<DType>, usize), // fixed vector, for SIMD computation
    Ptr(Box<DType>),        // for buffer
}
