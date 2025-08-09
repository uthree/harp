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

impl CBuffer {
    pub fn allocate(dtype: DType, shape: Vec<usize>) -> Self {
        let byte_size = shape.iter().product::<usize>() * dtype.size_in_bytes();
        let ptr = unsafe { libc::malloc(byte_size) };
        if ptr.is_null() {
            panic!("Failed to allocate memory for CBuffer");
        }
        CBuffer { ptr, shape, dtype }
    }

    pub fn from_slice<T: Clone + 'static>(data: &[T]) -> Self {
        let dtype = DType::from_type::<T>();
        let shape = vec![data.len()];
        let buffer = Self::allocate(dtype, shape);
        let slice = unsafe { std::slice::from_raw_parts_mut(buffer.ptr as *mut T, data.len()) };
        slice.clone_from_slice(data);
        buffer
    }

    pub fn as_slice<T>(&self) -> &[T] {
        let len = self.shape.iter().product();
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, len) }
    }
}

impl Clone for CBuffer {
    fn clone(&self) -> Self {
        let mut new_buffer = Self::allocate(self.dtype.clone(), self.shape.clone());
        new_buffer.as_mut_bytes().copy_from_slice(self.as_bytes());
        new_buffer
    }
}

impl Buffer for CBuffer {
    fn as_bytes(&self) -> &[u8] {
        let byte_size = self.shape.iter().product::<usize>() * self.dtype.size_in_bytes();
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, byte_size) }
    }

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
