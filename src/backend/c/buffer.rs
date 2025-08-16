use crate::ast::DType;
use crate::backend::Buffer;
use std::ffi::c_void;

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
    /// Creates a new buffer from a slice of data.
    pub fn from_slice<T>(data: &[T], shape: &[usize], dtype: DType) -> Self {
        let buffer = Self::allocate(dtype, shape.to_vec());
        let size = std::mem::size_of_val(data);
        unsafe {
            libc::memcpy(buffer.ptr, data.as_ptr() as *const c_void, size);
        }
        buffer
    }

    /// Copies the buffer data to a new `Vec`.
    pub fn to_vec<T>(&self) -> Vec<T> {
        let numel = self.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(numel);
        let size = numel * std::mem::size_of::<T>();
        unsafe {
            vec.set_len(numel);
            libc::memcpy(vec.as_mut_ptr() as *mut c_void, self.ptr, size);
        }
        vec
    }
}

impl Buffer for CBuffer {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn allocate(dtype: DType, shape: Vec<usize>) -> Self {
        let numel = shape.iter().product::<usize>();
        let size = numel * dtype.size();
        let ptr = unsafe { libc::malloc(size) };
        if ptr.is_null() {
            panic!("Failed to allocate memory for CBuffer");
        }
        CBuffer { ptr, shape, dtype }
    }
}

impl Drop for CBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::free(self.ptr);
        }
    }
}

impl DType {
    /// Returns the size of the data type in bytes.
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::Isize => std::mem::size_of::<isize>(),
            DType::Usize => std::mem::size_of::<usize>(),
            // Pointers are assumed to be 64-bit for now.
            DType::Ptr(_) => std::mem::size_of::<*const c_void>(),
            _ => unimplemented!("Size for dtype {:?} is not implemented", self),
        }
    }
}
