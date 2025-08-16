use crate::ast::DType;
use crate::backend::Buffer;

#[derive(Debug)]
pub struct CBuffer {
    /// Raw pointer to the allocated memory.
    pub ptr: *mut c_void,
    /// The shape of the buffer.
    pub shape: Vec<usize>,
    /// The data type of the elements.
    pub dtype: DType,
}
