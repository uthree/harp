use crate::ast::DType;
use crate::backend::Buffer;
use libc::c_void;

/// A buffer allocated on the C side.
#[derive(Debug)]
pub struct CBuffer {
    /// Raw pointer to the allocated memory.
    pub ptr: *mut c_void,
    /// The shape of the buffer.
    pub shape: Vec<usize>,
    /// The data type of the elements.
    pub dtype: DType,
}

impl Buffer for CBuffer {
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        let byte_size = self.shape.iter().product::<usize>() * self.dtype.size_in_bytes();
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, byte_size) }
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn allocate(dtype: DType, shape: Vec<usize>) -> Self {
        let byte_size = shape.iter().product::<usize>() * dtype.size_in_bytes();
        let ptr = unsafe { libc::malloc(byte_size) };
        if ptr.is_null() {
            panic!("Failed to allocate memory for CBuffer");
        }
        CBuffer { ptr, shape, dtype }
    }
}

impl Drop for CBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr);
            }
        }
    }
}
